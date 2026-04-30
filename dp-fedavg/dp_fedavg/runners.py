from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .aggregation import fixed_denominator_average
from .config import load_yaml_config
from .data import build_client_partitions, detect_partition_mode, load_eval_samples, load_private_samples, sample_client_ids
from .evaluation import run_downstream_eval
from .generation import build_generation_backend, build_generation_prompts, generate_synthetic_texts
from .paths import resolve_path_from_repo
from .privacy import add_gaussian_noise, clip_update, compute_noise_scale, estimate_privacy_epsilon
from .training import compute_local_update
from .types import ClientPartition, ExperimentRuntime, RunArtifacts


def build_experiment_runtime(config_path: Path, *, config: dict[str, Any] | None = None) -> ExperimentRuntime:
    cfg = config or load_yaml_config(config_path)
    runtime_cfg = dict(cfg.get("runtime", {}))
    output_root = resolve_path_from_repo(str(cfg["paths"]["output_root"]))
    return ExperimentRuntime(
        config_path=config_path,
        config=cfg,
        output_root=output_root,
        dataset_name=str(cfg["data"]["dataset_name"]),
        runner_mode=str(cfg["algorithm"]["runner_mode"]),
        seed=int(runtime_cfg.get("seed", 42)),
    )


def _ensure_output_dirs(output_root: Path) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    eval_dir = output_root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    return output_root, eval_dir


def _build_partitions(runtime: ExperimentRuntime, *, single_node: bool) -> list[ClientPartition]:
    cfg = runtime.config
    data_cfg = dict(cfg["data"])
    samples = load_private_samples(
        dataset_name=str(data_cfg["dataset_name"]),
        train_path=str(data_cfg["train_path"]),
        train_limit=int(data_cfg["train_limit"]) if data_cfg.get("train_limit") not in (None, "") else None,
    )
    natural_fields = list(data_cfg.get("natural_user_fields", []))
    if single_node:
        return [
            ClientPartition(
                client_id="client_0",
                samples=samples[: int(data_cfg.get("max_samples_per_client", len(samples) or 1))],
                metadata={"mode": "single_node"},
            )
        ]
    configured_mode = str(data_cfg.get("partition_mode", "auto")).strip().lower()
    if configured_mode == "auto":
        partition_mode = detect_partition_mode(samples, natural_fields)
    else:
        partition_mode = configured_mode
    num_clients = int(data_cfg.get("pseudo_num_clients", 8))
    max_samples_per_client = int(data_cfg.get("max_samples_per_client", 16))
    return build_client_partitions(
        samples,
        partition_mode=partition_mode,
        num_clients=num_clients,
        max_samples_per_client=max_samples_per_client,
        seed=runtime.seed,
        natural_user_fields=natural_fields,
    )


def _collect_selected_partitions(runtime: ExperimentRuntime, partitions: list[ClientPartition]) -> list[ClientPartition]:
    sampling_cfg = dict(runtime.config["algorithm"]["sampling"])
    if runtime.runner_mode == "single_node":
        return partitions[:1]
    selected_ids = set(
        sample_client_ids(
            [partition.client_id for partition in partitions],
            sample_rate=float(sampling_cfg.get("sample_rate", 1.0)),
            seed=runtime.seed,
        )
    )
    return [partition for partition in partitions if partition.client_id in selected_ids]


def _aggregate_updates(runtime: ExperimentRuntime, partitions: list[ClientPartition]) -> tuple[dict[str, list[float]], list[dict[str, Any]]]:
    dp_cfg = dict(runtime.config["algorithm"]["dp"])
    expected_clients = int(runtime.config["algorithm"]["sampling"].get("expected_clients", len(partitions) or 1))
    noise_scale = compute_noise_scale(
        clip_norm=float(dp_cfg.get("clip_norm", 1.0)),
        noise_multiplier=float(dp_cfg.get("noise_multiplier", 0.0)),
    )
    clipped_updates: list[dict[str, list[float]]] = []
    client_stats: list[dict[str, Any]] = []
    for index, partition in enumerate(partitions):
        raw_update = compute_local_update(partition)
        clipped = clip_update(raw_update, clip_norm=float(dp_cfg.get("clip_norm", 1.0)))
        clipped_updates.append(clipped)
        client_stats.append(
            {
                "client_id": partition.client_id,
                "sample_count": len(partition.samples),
                "raw_update": raw_update,
                "clipped_update": clipped,
                "round_seed": runtime.seed + index,
            }
        )
    averaged = fixed_denominator_average(clipped_updates, expected_clients=expected_clients)
    noised = add_gaussian_noise(averaged, noise_scale=noise_scale, seed=runtime.seed)
    return noised, client_stats


def _generate_or_fallback(runtime: ExperimentRuntime, partitions: list[ClientPartition]) -> list[str]:
    generation_cfg = dict(runtime.config.get("generation", {}))
    if not bool(generation_cfg.get("enabled", True)):
        return [
            sample.render_text().strip()
            for partition in partitions
            for sample in partition.samples[:1]
            if sample.render_text().strip()
        ]
    backend = build_generation_backend(dict(runtime.config["llm"]))
    prompts = build_generation_prompts(
        partitions,
        generated_per_round=int(generation_cfg.get("generated_per_round", 1)),
        exemplars_per_prompt=int(generation_cfg.get("exemplars_per_prompt", 1)),
        system_prompt=str(generation_cfg.get("initial_prompt", "Generate privacy-preserving synthetic text.")),
    )
    return generate_synthetic_texts(
        backend,
        prompts=prompts,
        max_new_tokens=int(generation_cfg.get("max_new_tokens", runtime.config["llm"].get("max_new_tokens", 96))),
        temperature=float(generation_cfg.get("temperature", runtime.config["llm"].get("temperature", 0.8))),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_stage_summary(runtime: ExperimentRuntime, selected: list[ClientPartition], aggregated_update: dict[str, list[float]], client_stats: list[dict[str, Any]]) -> dict[str, Any]:
    dp_cfg = dict(runtime.config["algorithm"]["dp"])
    sampling_cfg = dict(runtime.config["algorithm"]["sampling"])
    return {
        "dataset_name": runtime.dataset_name,
        "runner_mode": runtime.runner_mode,
        "selected_clients": [partition.client_id for partition in selected],
        "selected_client_count": len(selected),
        "client_stats": client_stats,
        "aggregated_update": aggregated_update,
        "clip_norm": float(dp_cfg.get("clip_norm", 1.0)),
        "noise_multiplier": float(dp_cfg.get("noise_multiplier", 0.0)),
        "noise_scale": compute_noise_scale(
            clip_norm=float(dp_cfg.get("clip_norm", 1.0)),
            noise_multiplier=float(dp_cfg.get("noise_multiplier", 0.0)),
        ),
        "privacy": {
            "delta": float(dp_cfg.get("delta", 1.0e-5)),
            "epsilon_estimate": estimate_privacy_epsilon(
                rounds=1,
                sample_rate=float(sampling_cfg.get("sample_rate", 1.0)),
                noise_multiplier=float(dp_cfg.get("noise_multiplier", 0.0)),
                delta=float(dp_cfg.get("delta", 1.0e-5)),
            ),
        },
    }


def _run(runtime: ExperimentRuntime, *, single_node: bool) -> RunArtifacts:
    output_root, eval_dir = _ensure_output_dirs(runtime.output_root)
    partitions = _build_partitions(runtime, single_node=single_node)
    selected = _collect_selected_partitions(runtime, partitions)
    aggregated_update, client_stats = _aggregate_updates(runtime, selected)
    synthetic_texts = _generate_or_fallback(runtime, selected)
    stage_summary = _build_stage_summary(runtime, selected, aggregated_update, client_stats)
    _write_json(output_root / "stage1_summary.json", stage_summary)
    eval_summary = None
    if bool(runtime.config.get("evaluation", {}).get("enabled", True)) and synthetic_texts:
        eval_summary = run_downstream_eval(
            synthetic_texts=synthetic_texts,
            dp_config=runtime.config,
            output_dir=eval_dir,
        )
    if eval_summary is not None:
        _write_json(eval_dir / "dp_fedavg_eval_summary.json", eval_summary)
    return RunArtifacts(
        synthetic_texts=synthetic_texts,
        stage_summary=stage_summary,
        eval_summary=eval_summary,
    )


def run_federated(runtime: ExperimentRuntime) -> RunArtifacts:
    return _run(runtime, single_node=False)


def run_single_node(runtime: ExperimentRuntime) -> RunArtifacts:
    return _run(runtime, single_node=True)
