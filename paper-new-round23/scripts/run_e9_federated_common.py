#!/usr/bin/env python3
"""Shared helpers for E9 federated multi-client experiment runners."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from round23_runtime_utils import PAPER_NEW_ROUND19_ROOT, deep_merge, load_yaml_with_inherits
from round23_federated_utils import load_partition_manifest, resolve_dataset_train_path

if str(PAPER_NEW_ROUND19_ROOT) not in sys.path:
    sys.path.insert(0, str(PAPER_NEW_ROUND19_ROOT))

from build_e9_federated_partitions import build_partition_artifact
from paper_new_selector.eval_bridge import run_eval
from paper_new_selector.thesis_bridge import load_text_samples, resolve_output_root, resolve_repo_root, write_json
from thesis_platform.core.schemas import Sample
from thesis_platform.data.partition import partition_samples
from thesis_platform.evaluation.downstream_eval import export_synthetic_corpus


E9_METHOD_DISPLAY_NAMES = {
    "e9_round23": "round23",
    "e9_round19": "round19",
    "e9_pretext": "PrE-Text",
}
E9_DEFAULT_TOTAL_PROMPT_BUDGET = 32
E9_DEFAULT_VALIDATION_RATIO = 0.0
E9_IMBALANCE_TEMPLATE_8 = [0.24, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06]
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FederatedSettings:
    federated_setting: str
    num_clients: int
    split_mode: str
    imbalance_mode: str
    partition_manifest: str
    total_prompt_budget: int
    validation_ratio: float
    per_client_prompt_budget: tuple[int, ...]


def load_federated_settings(config_path: str | Path) -> tuple[dict[str, Any], FederatedSettings]:
    cfg = load_yaml_with_inherits(config_path)
    fed_cfg = dict(cfg.get("e9_federated", {}))
    settings = FederatedSettings(
        federated_setting=str(fed_cfg.get("federated_setting", "f4_uniform")),
        num_clients=int(fed_cfg.get("num_clients", 4)),
        split_mode=str(fed_cfg.get("split_mode", "uniform")).strip().lower(),
        imbalance_mode=str(fed_cfg.get("imbalance_mode", "none")).strip().lower(),
        partition_manifest=str(fed_cfg.get("partition_manifest", "")).strip(),
        total_prompt_budget=int(fed_cfg.get("total_prompt_budget", E9_DEFAULT_TOTAL_PROMPT_BUDGET)),
        validation_ratio=float(fed_cfg.get("validation_ratio", E9_DEFAULT_VALIDATION_RATIO)),
        per_client_prompt_budget=tuple(int(value) for value in fed_cfg.get("per_client_prompt_budget", [])),
    )
    return cfg, settings


def resolve_repo_relative_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (REPO_ROOT / candidate).resolve()


def ensure_partition_manifest(
    config_path: str | Path,
    *,
    settings: FederatedSettings,
    seed: int,
) -> tuple[Path, dict[str, Any]]:
    if not settings.partition_manifest:
        raise ValueError("e9_federated.partition_manifest is required for E9 experiments.")
    manifest_path = resolve_repo_relative_path(settings.partition_manifest)
    if manifest_path.exists():
        return manifest_path, load_partition_manifest(manifest_path)

    dataset_name, train_path = resolve_dataset_train_path(config_path)
    build_partition_artifact(
        dataset_name=dataset_name,
        train_path=train_path,
        num_clients=settings.num_clients,
        split_mode=settings.split_mode,
        seed=seed,
        output_dir=manifest_path.parent,
    )
    return manifest_path, load_partition_manifest(manifest_path)


def resolve_partition_clients(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    clients = manifest.get("clients", [])
    if not isinstance(clients, list) or not clients:
        raise ValueError("Partition manifest must contain a non-empty clients list.")
    resolved_clients: list[dict[str, Any]] = []
    for client in clients:
        row = dict(client)
        row["train_path"] = str(resolve_repo_relative_path(str(client["train_path"])))
        resolved_clients.append(row)
    return resolved_clients


def _stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _sample_length(sample: Sample) -> int:
    return len(sample.render_text().split())


def _build_noniid_groups(samples: list[Sample], *, num_clients: int) -> list[list[Sample]]:
    ordered = sorted(
        samples,
        key=lambda sample: (
            _sample_length(sample),
            _stable_hash(sample.render_text()),
        ),
    )
    if not ordered:
        return [[] for _ in range(num_clients)]
    group_size = int(math.ceil(len(ordered) / float(max(1, num_clients))))
    groups = [ordered[index : index + group_size] for index in range(0, len(ordered), group_size)]
    while len(groups) < num_clients:
        groups.append([])
    return groups[:num_clients]


def _rebalance_to_targets(groups: list[list[Sample]], targets: list[int]) -> list[list[Sample]]:
    flat = [sample for group in groups for sample in group]
    rebalanced: list[list[Sample]] = []
    cursor = 0
    for target in targets:
        rebalanced.append(flat[cursor : cursor + target])
        cursor += target
    if cursor < len(flat):
        rebalanced[-1].extend(flat[cursor:])
    return rebalanced


def _imbalance_targets(total_count: int, *, num_clients: int) -> list[int]:
    if num_clients == 8:
        ratios = list(E9_IMBALANCE_TEMPLATE_8)
    else:
        base = [1.0 / float(num_clients)] * num_clients
        decay = 0.6
        weights = [decay**idx for idx in range(num_clients)]
        total_weight = sum(weights)
        ratios = [weight / total_weight for weight in weights]
        if len(base) == len(ratios):
            pass
    raw = [ratio * total_count for ratio in ratios]
    targets = [max(1, int(value)) for value in raw]
    diff = total_count - sum(targets)
    index = 0
    while diff != 0 and targets:
        slot = index % len(targets)
        if diff > 0:
            targets[slot] += 1
            diff -= 1
        elif targets[slot] > 1:
            targets[slot] -= 1
            diff += 1
        index += 1
    return targets


def build_client_partitions(
    samples: list[Sample],
    *,
    settings: FederatedSettings,
    seed: int,
) -> list[list[Sample]]:
    if settings.split_mode == "uniform":
        partitions = partition_samples(
            samples,
            num_clients=settings.num_clients,
            max_samples_per_client=max(1, int(math.ceil(len(samples) / max(1, settings.num_clients)))),
            validation_ratio=settings.validation_ratio,
            seed=seed,
            strategy="shuffle_round_robin",
        )
        return [list(part["train"] or part["all"]) for part in partitions]

    groups = _build_noniid_groups(samples, num_clients=settings.num_clients)
    if (
        settings.split_mode == "imbalance_noniid"
        or settings.imbalance_mode in {"long_tail", "fixed_tail_v1"}
    ):
        groups = _rebalance_to_targets(groups, _imbalance_targets(len(samples), num_clients=settings.num_clients))

    client_buckets: list[list[Sample]] = []
    for idx, bucket in enumerate(groups):
        client_samples: list[Sample] = []
        for local_idx, sample in enumerate(bucket):
            client_samples.append(
                Sample(
                    sample_id=f"client_{idx}_train_{local_idx}",
                    client_id=f"client_{idx:03d}",
                    round_id=sample.round_id,
                    source=sample.source,
                    dataset_name=sample.dataset_name,
                    task_type=sample.task_type,
                    text=sample.text,
                    instruction=sample.instruction,
                    response=sample.response,
                    label=sample.label,
                    meta=dict(sample.meta),
                )
            )
        client_buckets.append(client_samples)
    while len(client_buckets) < settings.num_clients:
        client_buckets.append([])
    return client_buckets


def sample_to_record(sample: Sample) -> dict[str, Any]:
    record: dict[str, Any] = {"text": sample.text}
    if sample.instruction is not None:
        record["instruction"] = sample.instruction
    if sample.response is not None:
        record["response"] = sample.response
    if sample.label is not None:
        record["label"] = sample.label
    if sample.meta:
        record["meta"] = dict(sample.meta)
    return record


def write_client_train_json(path: Path, samples: list[Sample]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([sample_to_record(sample) for sample in samples], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def allocate_client_prompt_budget(total_prompt_budget: int, num_clients: int) -> list[int]:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    base = total_prompt_budget // num_clients
    remainder = total_prompt_budget % num_clients
    budgets = [max(1, base) for _ in range(num_clients)]
    for idx in range(remainder):
        budgets[idx] += 1
    return budgets


def resolve_client_prompt_budgets(settings: FederatedSettings) -> list[int]:
    if settings.per_client_prompt_budget:
        budgets = [int(value) for value in settings.per_client_prompt_budget]
        if len(budgets) != settings.num_clients:
            raise ValueError(
                f"per_client_prompt_budget length {len(budgets)} != num_clients {settings.num_clients}"
            )
        return budgets
    return allocate_client_prompt_budget(settings.total_prompt_budget, settings.num_clients)


def write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    return path


def build_client_config_payload(
    *,
    original_config_path: str | Path,
    client_output_root: Path,
    client_train_path: Path,
    prompt_budget: int,
    method: str,
    reference_budget: int = 20,
) -> dict[str, Any]:
    original_cfg = load_yaml_with_inherits(original_config_path)
    override: dict[str, Any] = {
        "paths": {
            "output_root": str(client_output_root.resolve()),
        },
        "data": {
            "train_path": str(client_train_path.resolve()),
        },
        "bootstrap": {
            "num_prompts": int(prompt_budget),
        },
        "eval": {
            "enabled": False,
            "mode": "pretext_small",
        },
    }
    if method == "e9_pretext":
        override["pipeline"] = {"stage1_mode": "expand_private"}
        override["selector"] = {
            "seed_top_k": int(reference_budget),
            "seed_budget_rule": {"enabled": False, "mode": "hierarchical_shape_routing"},
        }
        override["round23_controller"] = {"enabled": False}
    elif method == "e9_round19":
        override["selector"] = {
            "seed_top_k": int(reference_budget),
            "seed_budget_rule": {
                "enabled": True,
                "mode": "hierarchical_shape_routing",
            },
        }
        override["round23_controller"] = {"enabled": False}
    elif method == "e9_round23":
        override["selector"] = {
            "seed_top_k": int(reference_budget),
            "seed_budget_rule": {"enabled": False, "mode": "hierarchical_shape_routing"},
        }
        override["round23_controller"] = {
            "enabled": True,
            "reference_budget": int(reference_budget),
            "action_space": [-2, -1, 0, 1, 2],
        }
    else:
        raise ValueError(f"Unsupported E9 method: {method}")
    return deep_merge(original_cfg, override)


def export_client_synthetic_texts(summary: dict[str, Any], *, client_output_root: Path) -> tuple[list[str], Path]:
    synthetic_texts = [str(text) for text in summary.get("stage2", {}).get("synthetic_outputs", []) if str(text).strip()]
    stage2_dir = client_output_root / "stage2"
    corpus_path = export_synthetic_corpus(synthetic_texts, output_dir=stage2_dir, filename="llama7b_text_syn.json")
    return synthetic_texts, corpus_path


def run_client_pipeline_subprocess(
    config_path: Path,
    *,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    summary_path = config_path.parent / "pipeline_summary.json"
    log_path = config_path.parent / "pipeline_run.log"
    child_code = "\n".join(
        [
            "import json",
            "import sys",
            "from pathlib import Path",
            f"sys.path.insert(0, {str(PAPER_NEW_ROUND19_ROOT.resolve())!r})",
            "from paper_new_selector.pipeline import run_pipeline",
            f"config_path = Path({str(config_path)!r})",
            f"summary_path = Path({str(summary_path)!r})",
            "summary = run_pipeline(config_path, validate_only=False)",
            "summary_path.write_text(json.dumps(summary, ensure_ascii=False, default=str), encoding='utf-8')",
        ]
    )
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            [sys.executable, "-c", child_code],
            cwd=str(REPO_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            text=True,
        )
    if completed.returncode != 0:
        log_excerpt = ""
        if log_path.exists():
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            log_excerpt = "\n".join(lines[-40:])
        raise RuntimeError(
            "Client pipeline subprocess failed"
            + (f" (see {log_path})" if log_path else "")
            + (f": {log_excerpt}" if log_excerpt else "")
        )
    if not summary_path.exists():
        raise FileNotFoundError(f"Client pipeline summary was not written: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def write_aggregated_synthetic_texts(path: Path, texts: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(texts), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_server_eval_config_payload(
    *,
    original_config_path: str | Path,
    server_output_root: Path,
) -> dict[str, Any]:
    original_cfg = load_yaml_with_inherits(original_config_path)
    override = {
        "paths": {
            "output_root": str(server_output_root.resolve()),
        },
        "eval": {
            "enabled": True,
            "mode": "pretext_small",
        },
    }
    return deep_merge(original_cfg, override)


def write_partition_manifest(
    path: Path,
    *,
    settings: FederatedSettings,
    client_rows: list[dict[str, Any]],
) -> Path:
    payload = {
        "federated_setting": settings.federated_setting,
        "num_clients": settings.num_clients,
        "split_mode": settings.split_mode,
        "imbalance_mode": settings.imbalance_mode,
        "clients": client_rows,
    }
    write_json(path, payload)
    return path


def build_federated_sidecar(
    *,
    output_root: Path,
    experiment_id: str,
    method: str,
    settings: FederatedSettings,
    partition_manifest_path: Path,
    client_rows: list[dict[str, Any]],
    aggregated_texts: list[str],
    eval_summary: dict[str, Any],
    reference_budget: int,
    controller_bundle: str = "",
    model_metadata: dict[str, Any] | None = None,
) -> Path:
    deduped_count = 0
    canonical_corpus_path = str(eval_summary.get("canonical_synthetic_corpus_path", "")).strip()
    if canonical_corpus_path:
        candidate = Path(canonical_corpus_path)
        if candidate.exists():
            try:
                deduped_count = len(json.loads(candidate.read_text(encoding="utf-8")))
            except Exception:
                deduped_count = 0
    predicted_delta_values = [
        int(row["predicted_delta_k"])
        for row in client_rows
        if row.get("status") == "success" and row.get("predicted_delta_k") not in (None, "")
    ]
    predicted_budget_values = [
        int(row["predicted_target_budget"])
        for row in client_rows
        if row.get("status") == "success" and row.get("predicted_target_budget") not in (None, "")
    ]
    sidecar = {
        "budget_policy_type": "e9_federated",
        "experiment_id": experiment_id,
        "method": method,
        "method_display_name": E9_METHOD_DISPLAY_NAMES.get(method, method),
        "federated_setting": settings.federated_setting,
        "num_clients": settings.num_clients,
        "split_mode": settings.split_mode,
        "imbalance_mode": settings.imbalance_mode,
        "controller_scope": "all6" if method == "e9_round23" else "",
        "bundle_version": controller_bundle,
        "learner_family": str((model_metadata or {}).get("learner_family", "")),
        "model_family": str((model_metadata or {}).get("model_family", "")),
        "feature_version": str((model_metadata or {}).get("feature_version", "")),
        "target_mode": str((model_metadata or {}).get("target_mode", "")),
        "target_field": str((model_metadata or {}).get("target_field", "")),
        "reference_budget": int(reference_budget),
        "predicted_delta_k": json.dumps(predicted_delta_values, ensure_ascii=False) if predicted_delta_values else "",
        "predicted_target_budget": (
            json.dumps(predicted_budget_values, ensure_ascii=False) if predicted_budget_values else ""
        ),
        "partition_manifest_path": str(partition_manifest_path),
        "client_success_count": sum(1 for row in client_rows if row.get("status") == "success"),
        "client_failure_count": sum(1 for row in client_rows if row.get("status") != "success"),
        "aggregated_synthetic_count": len(aggregated_texts),
        "aggregated_synthetic_count_deduped": deduped_count,
        "runtime_artifacts": {
            "eval_summary": eval_summary,
            "clients": client_rows,
        },
    }
    sidecar_path = output_root / f"{experiment_id}_federated_runtime.json"
    sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
    return sidecar_path


def resolve_full_dataset_bundle(config_path: str | Path) -> dict[str, Any]:
    return load_text_samples(config_path)


def run_server_eval(
    *,
    synthetic_texts: list[str],
    server_config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    return run_eval(synthetic_texts=synthetic_texts, config_path=server_config_path, output_dir=output_dir)


def resolve_experiment_id(config_path: str | Path) -> str:
    cfg = load_yaml_with_inherits(config_path)
    return str(cfg.get("meta", {}).get("experiment_id", Path(config_path).stem))


def resolve_reference_budget(config_path: str | Path) -> int:
    cfg = load_yaml_with_inherits(config_path)
    runtime_cfg = dict(cfg.get("round23_controller", {}))
    return int(runtime_cfg.get("reference_budget", cfg.get("selector", {}).get("seed_top_k", 20)))


def resolve_output_root_for_config(config_path: str | Path) -> Path:
    return resolve_output_root(config_path)
