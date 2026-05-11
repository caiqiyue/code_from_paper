#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from append_round22_bandit_summary import (
    SUMMARY_FIELDS,
    append_jsonl_row,
    append_tsv_row,
    build_summary_row,
    compute_coverage_metrics,
    initialize_tsv,
    read_json,
)


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    dataset_name: str
    meta_seed: int
    action_budget: int
    normalized_budget_cost: float
    config_path: Path
    output_root: str
    context_family: str
    source_env: str

    @property
    def context_key(self) -> tuple[str, int]:
        return (self.dataset_name, int(self.meta_seed))


def resolve_round22_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_round19_root(round22_root: Path) -> Path:
    candidate = round22_root.parent / "paper-new-round19"
    if not candidate.exists():
        raise FileNotFoundError(f"Could not locate sibling paper-new-round19 repo: {candidate}")
    return candidate.resolve()


ROUND22_ROOT = resolve_round22_root()
ROUND19_ROOT = resolve_round19_root(ROUND22_ROOT)

if str(ROUND19_ROOT) not in sys.path:
    sys.path.insert(0, str(ROUND19_ROOT))

from paper_new_selector.thesis_bridge import build_embedder_from_config, load_text_samples  # type: ignore  # noqa: E402


DATASET_ORDER = ["jobs", "congressional", "forums", "microblog"]
BUDGET_ORDER = [20, 18, 19, 21, 22]
REWARD_LAMBDA = 0.002


def _percentile_nearest_rank(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(int(value) for value in values)
    if percentile <= 0:
        return float(sorted_values[0])
    if percentile >= 100:
        return float(sorted_values[-1])
    rank = int(math.ceil((float(percentile) / 100.0) * len(sorted_values)))
    return float(sorted_values[max(0, rank - 1)])


def compute_shape_descriptor(
    private_lengths: list[int],
    *,
    tail_threshold: int,
    short_threshold: int,
) -> dict[str, float]:
    if not private_lengths:
        return {
            "median_len": 0.0,
            "p75_len": 0.0,
            "tail_ratio": 0.0,
            "short_ratio": 0.0,
            "iqr_len": 0.0,
        }
    q1 = _percentile_nearest_rank(private_lengths, 25)
    q3 = _percentile_nearest_rank(private_lengths, 75)
    total = float(len(private_lengths))
    return {
        "median_len": float(_percentile_nearest_rank(private_lengths, 50)),
        "p75_len": float(q3),
        "tail_ratio": float(sum(length >= int(tail_threshold) for length in private_lengths) / total),
        "short_ratio": float(sum(length <= int(short_threshold) for length in private_lengths) / total),
        "iqr_len": float(q3 - q1),
    }


def compute_shape_score(descriptor: dict[str, float], router_cfg: dict[str, Any]) -> float:
    reference = dict(router_cfg.get("screening_reference", {}))
    median_stats = dict(reference.get("median_len", {"mean": 0.0, "std": 1.0}))
    p75_stats = dict(reference.get("p75_len", {"mean": 0.0, "std": 1.0}))
    iqr_stats = dict(reference.get("iqr_len", {"mean": 0.0, "std": 1.0}))

    def _zscore(value: float, mean: float, std: float) -> float:
        if abs(float(std)) <= 1e-8:
            return 0.0
        return float((float(value) - float(mean)) / float(std))

    return (
        _zscore(descriptor["median_len"], median_stats["mean"], median_stats["std"])
        + _zscore(descriptor["p75_len"], p75_stats["mean"], p75_stats["std"])
        + _zscore(descriptor["iqr_len"], iqr_stats["mean"], iqr_stats["std"])
        + float(descriptor["tail_ratio"])
        - float(descriptor["short_ratio"])
    )


def load_manifest(manifest_path: Path) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            specs.append(
                ExperimentSpec(
                    experiment_id=str(row["experiment_id"]),
                    dataset_name=str(row["dataset"]),
                    meta_seed=int(row["seed"]),
                    action_budget=int(row["action_budget"]),
                    normalized_budget_cost=float(row["normalized_budget_cost"]),
                    config_path=normalize_manifest_config_path(str(row["config_path"])),
                    output_root=str(row["output_root"]),
                    context_family="natural",
                    source_env="round19_stable_pipeline",
                )
            )
    return specs


def normalize_manifest_config_path(raw_path: str) -> Path:
    normalized = raw_path.replace("\\", "/")
    marker = "/paper-new-round22/"
    if marker in normalized:
        relative = normalized.split(marker, 1)[1]
        candidate = (ROUND22_ROOT / relative).resolve()
        if candidate.exists():
            return candidate
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Could not normalize manifest config path: {raw_path}")


def build_seed_order(specs: list[ExperimentSpec]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for spec in specs:
        if spec.meta_seed not in seen:
            seen.add(spec.meta_seed)
            ordered.append(spec.meta_seed)
    return ordered


def order_specs_for_mode(specs: list[ExperimentSpec], mode: str) -> list[ExperimentSpec]:
    if mode not in {"smoke", "full"}:
        raise ValueError(f"Unsupported mode: {mode}")
    filtered = specs
    if mode == "smoke":
        filtered = [spec for spec in specs if spec.meta_seed == 42]

    by_key = {(spec.dataset_name, spec.meta_seed, spec.action_budget): spec for spec in filtered}
    seed_order = build_seed_order(filtered)
    ordered: list[ExperimentSpec] = []
    for seed in seed_order:
        for dataset_name in DATASET_ORDER:
            for budget in BUDGET_ORDER:
                spec = by_key.get((dataset_name, seed, budget))
                if spec is not None:
                    ordered.append(spec)
    return ordered


def successful_ids_from_tsv(summary_tsv: Path) -> set[str]:
    if not summary_tsv.exists():
        return set()
    with summary_tsv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {
            str(row["experiment_id"])
            for row in reader
            if str(row.get("status", "")).strip().lower() == "success"
        }


def successful_ids_from_jsonl(summary_jsonl: Path) -> set[str]:
    if not summary_jsonl.exists():
        return set()
    success: set[str] = set()
    with summary_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("status", "")).strip().lower() == "success":
                success.add(str(row["experiment_id"]))
    return success


def completed_ids(summary_tsv: Path, summary_jsonl: Path) -> set[str]:
    tsv_ids = successful_ids_from_tsv(summary_tsv)
    jsonl_ids = successful_ids_from_jsonl(summary_jsonl)
    if not tsv_ids:
        return jsonl_ids
    if not jsonl_ids:
        return tsv_ids
    return tsv_ids.intersection(jsonl_ids)


def resolve_output_root(spec: ExperimentSpec) -> Path:
    return (ROUND22_ROOT.parent / spec.output_root).resolve()


def config_router_cfg(config_path: Path) -> dict[str, Any]:
    import yaml

    merged = load_yaml_with_inherits(config_path)
    return dict(merged.get("selector", {}).get("seed_budget_rule", {}).get("router", {}))


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


def build_private_dataset_cache() -> dict[str, dict[str, Any]]:
    return {}


def private_dataset_features(
    spec: ExperimentSpec,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if spec.dataset_name in cache:
        return cache[spec.dataset_name]
    sample_bundle = load_text_samples(spec.config_path)
    private_texts = [sample.render_text() for sample in sample_bundle["train_samples"]]
    private_lengths = [len(text.split()) for text in private_texts]
    router_cfg = config_router_cfg(spec.config_path)
    descriptor = compute_shape_descriptor(
        private_lengths,
        tail_threshold=int(router_cfg.get("tail_threshold", 350)),
        short_threshold=int(router_cfg.get("short_threshold", 120)),
    )
    features = {
        "private_texts": private_texts,
        "private_lengths": private_lengths,
        "private_mean_length": float(sum(private_lengths) / len(private_lengths)) if private_lengths else 0.0,
        "private_p75_length": float(_percentile_nearest_rank(private_lengths, 75)),
        "private_length_iqr": float(descriptor["iqr_len"]),
        "shape_score": compute_shape_score(descriptor, router_cfg),
    }
    cache[spec.dataset_name] = features
    return features


def get_embedder(config_path: Path, cache: dict[str, Any]) -> Any:
    key = "singleton"
    if key not in cache:
        cache[key] = build_embedder_from_config(config_path)
    return cache[key]


def compute_reference_features_from_k20_output(
    k20_spec: ExperimentSpec,
    *,
    dataset_cache: dict[str, dict[str, Any]],
    embedder_cache: dict[str, Any],
) -> dict[str, Any]:
    output_root = resolve_output_root(k20_spec)
    stage1_summary = read_json(output_root / "stage1_summary.json")
    decision = dict(stage1_summary.get("decision", {}))
    candidate_records = list(decision.get("candidate_records", []))
    selected_indices = [int(index) for index in decision.get("selected_indices", [])]
    selected_records = [record for record in candidate_records if int(record["index"]) in set(selected_indices)]
    support_mean_at_k20 = (
        float(sum(float(record["private_support"]) for record in selected_records) / len(selected_records))
        if selected_records
        else 0.0
    )
    genericity_mean_at_k20 = (
        float(sum(float(record["genericity_penalty"]) for record in selected_records) / len(selected_records))
        if selected_records
        else 0.0
    )
    selected_vectors = [list(map(float, record["vector"])) for record in selected_records]
    dataset_features = private_dataset_features(k20_spec, dataset_cache)
    embedder = get_embedder(k20_spec.config_path, embedder_cache)
    private_vectors = [
        list(map(float, vector))
        for vector in embedder.embed_texts(dataset_features["private_texts"])
    ]
    coverage_mean_at_k20, coverage_p25_at_k20 = compute_coverage_metrics(
        private_vectors=private_vectors,
        selected_vectors=selected_vectors,
    )
    return {
        "shape_score": dataset_features["shape_score"],
        "private_mean_length": dataset_features["private_mean_length"],
        "private_p75_length": dataset_features["private_p75_length"],
        "private_length_iqr": dataset_features["private_length_iqr"],
        "support_mean_at_k20": support_mean_at_k20,
        "coverage_mean_at_k20": coverage_mean_at_k20,
        "coverage_p25_at_k20": coverage_p25_at_k20,
        "genericity_mean_at_k20": genericity_mean_at_k20,
    }


def build_k20_lookup(specs: list[ExperimentSpec]) -> dict[tuple[str, int], ExperimentSpec]:
    lookup: dict[tuple[str, int], ExperimentSpec] = {}
    for spec in specs:
        if spec.action_budget == 20:
            lookup[spec.context_key] = spec
    return lookup


def log(master_path: Path, message: str) -> None:
    line = f"{datetime.now().strftime('%F %T')} {message}"
    print(line, flush=True)
    with master_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_single_experiment(
    *,
    spec: ExperimentSpec,
    log_path: Path,
) -> int:
    command = [
        sys.executable,
        "-m",
        "paper_new_selector.run_selector_single_node",
        "--config",
        str(spec.config_path),
    ]
    child_env = dict(os.environ)
    if child_env.get("CUDA_VISIBLE_DEVICES") and not child_env.get("CUDA_DEVICE_ORDER"):
        child_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=ROUND19_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            env=child_env,
        )
    return int(completed.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run round22 contextual-bandit data collection.")
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--manifest",
        default=str(ROUND22_ROOT / "configs/experiments/bandit_data_collection/round22_bandit_collection_manifest.tsv"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    specs = load_manifest(manifest_path)
    ordered_specs = order_specs_for_mode(specs, args.mode)
    k20_lookup = build_k20_lookup(specs)

    if args.dry_run:
        print(json.dumps(
            {
                "mode": args.mode,
                "count": len(ordered_specs),
                "first_experiments": [spec.experiment_id for spec in ordered_specs[:10]],
                "summary_fields": SUMMARY_FIELDS,
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0

    logs_dir = ROUND22_ROOT / "logs"
    per_exp_logs = logs_dir / f"round22_bandit_{args.mode}_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    per_exp_logs.mkdir(parents=True, exist_ok=True)
    summary_tsv = logs_dir / f"round22_bandit_{args.mode}_summary.tsv"
    summary_jsonl = logs_dir / f"round22_bandit_{args.mode}_summary.jsonl"
    master_log = logs_dir / f"round22_bandit_{args.mode}_master.log"
    initialize_tsv(summary_tsv)
    if not summary_jsonl.exists():
        summary_jsonl.write_text("", encoding="utf-8")

    already_done = completed_ids(summary_tsv, summary_jsonl)
    pending_specs = [spec for spec in ordered_specs if spec.experiment_id not in already_done]
    dataset_cache: dict[str, dict[str, Any]] = {}
    embedder_cache: dict[str, Any] = {}
    context_feature_cache: dict[tuple[str, int], dict[str, Any]] = {}

    for spec in pending_specs:
        log(master_log, f"START {spec.experiment_id} dataset={spec.dataset_name} seed={spec.meta_seed} k={spec.action_budget}")
        output_root = resolve_output_root(spec)
        log_path = per_exp_logs / f"{spec.experiment_id}.log"
        attempt = 0
        success = False
        last_return_code = 0
        duration_seconds = 0.0
        while attempt < max(1, int(args.max_attempts)):
            attempt += 1
            shutil.rmtree(output_root, ignore_errors=True)
            start = time.monotonic()
            last_return_code = run_single_experiment(spec=spec, log_path=log_path)
            duration_seconds = time.monotonic() - start
            if last_return_code == 0:
                success = True
                break
            log(master_log, f"RETRY {spec.experiment_id} attempt={attempt} status={last_return_code}")
            time.sleep(5)

        if not success:
            log(master_log, f"STOP_ON_FAILURE {spec.experiment_id} status={last_return_code}")
            return int(last_return_code or 1)

        metrics = dict(read_json(output_root / "eval" / "downstream_eval_summary.json").get("metrics") or {})
        context_key = spec.context_key
        if context_key not in context_feature_cache:
            k20_spec = k20_lookup.get(context_key)
            if k20_spec is None:
                raise KeyError(f"Missing k20 reference spec for context: {context_key}")
            if spec.action_budget == 20:
                reference_spec = spec
            else:
                reference_spec = k20_spec
            context_feature_cache[context_key] = compute_reference_features_from_k20_output(
                reference_spec,
                dataset_cache=dataset_cache,
                embedder_cache=embedder_cache,
            )

        row = build_summary_row(
            experiment_id=spec.experiment_id,
            dataset_name=spec.dataset_name,
            meta_seed=spec.meta_seed,
            action_budget=spec.action_budget,
            normalized_budget_cost=spec.normalized_budget_cost,
            state_features=context_feature_cache[context_key],
            output_root=spec.output_root,
            config_path=str(spec.config_path),
            source_env=spec.source_env,
            context_family=spec.context_family,
            attempt=attempt,
            duration_seconds=duration_seconds,
            downstream_metrics=metrics,
        )
        append_tsv_row(summary_tsv, row)
        append_jsonl_row(summary_jsonl, row)
        log(master_log, f"END {spec.experiment_id} status=0 duration={round(duration_seconds, 3)}")
        time.sleep(2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
