from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class Round19ReplayError(RuntimeError):
    pass


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


REPO_ROOT = resolve_repo_root()
ROUND19_ROOT = (REPO_ROOT / "paper-new-round19").resolve()
ROUND22_ROOT = (REPO_ROOT / "paper-new-round22").resolve()
ROUND19_BASE_CONFIG = (
    ROUND19_ROOT / "configs" / "experiments" / "single_node_tuning_round19" / "_base_selector_tuning_round19.yaml"
)

if str(ROUND19_ROOT) not in sys.path:
    sys.path.insert(0, str(ROUND19_ROOT))

from common import BUDGETS, as_float, as_int, read_jsonl  # noqa: E402
from paper_new_selector.budget_calibration import compute_budget_cost  # type: ignore  # noqa: E402
from paper_new_selector.hierarchical_budget import resolve_hierarchical_budget  # type: ignore  # noqa: E402
from paper_new_selector.redundancy import cosine_similarity  # type: ignore  # noqa: E402
from paper_new_selector.thesis_bridge import build_embedder_from_config, load_text_samples  # type: ignore  # noqa: E402


@dataclass
class ExplicitRound19ReplayPolicy:
    context_to_budget: dict[str, int]
    provenance: str = "explicit_mapping"

    def predict(self, context_row: dict[str, Any]) -> int:
        context_id = str(context_row["context_id"])
        if context_id not in self.context_to_budget:
            raise Round19ReplayError(f"Missing round19 replay label for context_id={context_id}")
        return int(self.context_to_budget[context_id])


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _percentile_nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(float(v) for v in values)
    if percentile <= 0:
        return float(sorted_values[0])
    if percentile >= 100:
        return float(sorted_values[-1])
    rank = int(math.ceil((float(percentile) / 100.0) * len(sorted_values)))
    return float(sorted_values[max(0, rank - 1)])


def compute_selected_redundancy_score(*, selected_vectors: list[list[float]]) -> float:
    if len(selected_vectors) < 2:
        return 0.0
    similarities: list[float] = []
    for left_index in range(len(selected_vectors)):
        for right_index in range(left_index + 1, len(selected_vectors)):
            similarities.append(
                cosine_similarity(
                    selected_vectors[left_index],
                    selected_vectors[right_index],
                )
            )
    return _mean(similarities)


def compute_selected_coverage_score(
    *,
    private_vectors: list[list[float]],
    selected_vectors: list[list[float]],
) -> dict[str, float]:
    if not private_vectors or not selected_vectors:
        return {"coverage_mean": 0.0, "coverage_p25": 0.0, "coverage_min": 0.0}
    coverage_values = [
        max(cosine_similarity(private_vector, selected) for selected in selected_vectors)
        for private_vector in private_vectors
    ]
    return {
        "coverage_mean": _mean(coverage_values),
        "coverage_p25": _percentile_nearest_rank(coverage_values, 25),
        "coverage_min": float(min(coverage_values)),
    }


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_with_inherits(config_path: Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    inherits = data.get("inherits", []) or []
    merged: dict[str, Any] = {}
    for include in inherits:
        include_path = (config_path.parent / str(include)).resolve()
        merged = deep_merge(merged, load_yaml_with_inherits(include_path))
    return deep_merge(merged, data)


def load_round19_rule_cfg() -> dict[str, Any]:
    merged = load_yaml_with_inherits(ROUND19_BASE_CONFIG)
    selector_cfg = dict(merged.get("selector", {}))
    rule_cfg = dict(selector_cfg.get("seed_budget_rule", {}))
    if not rule_cfg:
        raise Round19ReplayError(f"Failed to load round19 seed_budget_rule from {ROUND19_BASE_CONFIG}")
    return rule_cfg


def normalize_round22_path(raw_path: str) -> Path:
    normalized = str(raw_path).replace("\\", "/")
    marker = "/paper-new-round22/"
    if marker in normalized:
        relative = normalized.split(marker, 1)[1]
        candidate = (ROUND22_ROOT / relative).resolve()
        if candidate.exists():
            return candidate
    if normalized.startswith("paper-new-round22/"):
        candidate = (REPO_ROOT / normalized).resolve()
        if candidate.exists():
            return candidate
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Could not normalize round22 path: {raw_path}")


def build_private_dataset_cache() -> dict[str, dict[str, Any]]:
    return {}


def get_embedder(config_path: Path, cache: dict[str, Any]) -> Any:
    key = str(config_path.resolve())
    if key not in cache:
        cache[key] = build_embedder_from_config(config_path)
    return cache[key]


def private_dataset_artifacts(
    *,
    dataset_name: str,
    config_path: Path,
    dataset_cache: dict[str, dict[str, Any]],
    embedder_cache: dict[str, Any],
) -> dict[str, Any]:
    if dataset_name in dataset_cache:
        return dataset_cache[dataset_name]
    sample_bundle = load_text_samples(config_path)
    private_texts = [sample.render_text() for sample in sample_bundle["train_samples"]]
    private_lengths = [len(text.split()) for text in private_texts]
    embedder = get_embedder(config_path, embedder_cache)
    private_vectors = [
        list(map(float, vector))
        for vector in embedder.embed_texts(private_texts)
    ]
    artifacts = {
        "private_texts": private_texts,
        "private_lengths": private_lengths,
        "private_vectors": private_vectors,
    }
    dataset_cache[dataset_name] = artifacts
    return artifacts


def _read_stage1_summary(output_root: Path) -> dict[str, Any]:
    stage1_summary_path = output_root / "stage1_summary.json"
    if not stage1_summary_path.exists():
        raise FileNotFoundError(f"Missing stage1_summary.json under {output_root}")
    return json.loads(stage1_summary_path.read_text(encoding="utf-8"))


def _metrics_from_action_row(
    row: dict[str, Any],
    *,
    dataset_cache: dict[str, dict[str, Any]],
    embedder_cache: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    budget = as_int(row["action_budget"])
    output_root = normalize_round22_path(str(row["output_root"]))
    config_path = normalize_round22_path(str(row["config_path"]))
    stage1_summary = _read_stage1_summary(output_root)
    decision = dict(stage1_summary.get("decision", {}))
    candidate_records = list(decision.get("candidate_records", []))
    selected_indices = [int(index) for index in decision.get("selected_indices", [])]
    selected_index_set = set(selected_indices)
    selected_records = [
        record
        for record in candidate_records
        if int(record.get("index", -1)) in selected_index_set
    ]
    selected_vectors = [list(map(float, record["vector"])) for record in selected_records]

    private_artifacts = private_dataset_artifacts(
        dataset_name=str(row["dataset_name"]),
        config_path=config_path,
        dataset_cache=dataset_cache,
        embedder_cache=embedder_cache,
    )
    coverage_stats = compute_selected_coverage_score(
        private_vectors=private_artifacts["private_vectors"],
        selected_vectors=selected_vectors,
    )
    support_mean = _mean([as_float(record["private_support"]) for record in selected_records])
    genericity_mean = _mean([as_float(record["genericity_penalty"]) for record in selected_records])
    return budget, {
        "selected_count": len(selected_records),
        "selected_indices": selected_indices,
        "support_score": support_mean,
        "support_mean": support_mean,
        "genericity_score": genericity_mean,
        "redundancy_score": compute_selected_redundancy_score(selected_vectors=selected_vectors),
        "coverage_mean": coverage_stats["coverage_mean"],
        "coverage_p25": coverage_stats["coverage_p25"],
        "coverage_min": coverage_stats["coverage_min"],
        "budget_cost": compute_budget_cost(
            seed_top_k=int(budget),
            candidate_seed_top_k=BUDGETS,
        ),
    }


def load_explicit_round19_replay(path: str | Path) -> ExplicitRound19ReplayPolicy:
    resolved = Path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Round19 replay mapping must be a JSON object {context_id: budget}")
    mapping = {str(key): int(value) for key, value in payload.items()}
    return ExplicitRound19ReplayPolicy(mapping, provenance=str(resolved.resolve()))


def build_round19_replay_policy_from_action_samples(
    *,
    action_samples_path: str | Path,
    context_ids: set[str] | None = None,
) -> ExplicitRound19ReplayPolicy:
    rows = read_jsonl(action_samples_path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        context_key = str(row["context_id"])
        if context_ids is not None and context_key not in context_ids:
            continue
        grouped.setdefault(context_key, []).append(row)

    rule_cfg = load_round19_rule_cfg()
    dataset_cache = build_private_dataset_cache()
    embedder_cache: dict[str, Any] = {}
    mapping: dict[str, int] = {}
    for context_key, context_rows in grouped.items():
        budgets = sorted(as_int(row["action_budget"]) for row in context_rows)
        if budgets != BUDGETS:
            raise Round19ReplayError(
                f"Context {context_key} does not provide a complete budget sweep {BUDGETS}: got {budgets}"
            )
        metrics_by_budget: dict[int, dict[str, Any]] = {}
        for row in context_rows:
            budget, metrics = _metrics_from_action_row(
                row,
                dataset_cache=dataset_cache,
                embedder_cache=embedder_cache,
            )
            metrics_by_budget[int(budget)] = metrics
        config_path = normalize_round22_path(str(context_rows[0]["config_path"]))
        private_artifacts = private_dataset_artifacts(
            dataset_name=str(context_rows[0]["dataset_name"]),
            config_path=config_path,
            dataset_cache=dataset_cache,
            embedder_cache=embedder_cache,
        )
        selected = resolve_hierarchical_budget(
            private_lengths=list(private_artifacts["private_lengths"]),
            metrics_by_budget=metrics_by_budget,
            rule_cfg=rule_cfg,
        )
        resolved_budget = selected.get("resolved_seed_top_k")
        if resolved_budget is None:
            raise Round19ReplayError(f"round19 replay returned no budget for context_id={context_key}")
        mapping[context_key] = int(resolved_budget)
    return ExplicitRound19ReplayPolicy(mapping, provenance=f"reconstructed_from:{Path(action_samples_path).resolve()}")


def validate_explicit_round19_replay(
    *,
    context_rows: list[dict[str, Any]],
    policy: ExplicitRound19ReplayPolicy,
) -> dict[str, Any]:
    missing: list[str] = []
    invalid: dict[str, int] = {}
    for row in context_rows:
        context_id = str(row["context_id"])
        if context_id not in policy.context_to_budget:
            missing.append(context_id)
            continue
        budget = int(policy.context_to_budget[context_id])
        if budget not in BUDGETS:
            invalid[context_id] = budget
    if missing or invalid:
        raise Round19ReplayError(
            f"Invalid explicit round19 replay mapping: missing={len(missing)} invalid={len(invalid)}"
        )
    return {
        "validated_context_count": len(context_rows),
        "mapping_count": len(policy.context_to_budget),
        "provenance": policy.provenance,
    }


def predict_round19_budget(state: dict[str, Any]) -> int:
    raise Round19ReplayError(
        "Direct stateless round19 replay from summary-only state is not supported. "
        "Use build_round19_replay_policy_from_action_samples() to reconstruct the resolver "
        "from round22 sweep outputs, then call policy.predict(context_row)."
    )
