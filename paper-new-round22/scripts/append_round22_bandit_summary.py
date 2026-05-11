#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


SUMMARY_FIELDS = [
    "experiment_id",
    "dataset_name",
    "meta_seed",
    "action_budget",
    "normalized_budget_cost",
    "best_top1",
    "reward",
    "shape_score",
    "private_mean_length",
    "private_p75_length",
    "private_length_iqr",
    "support_mean_at_k20",
    "coverage_mean_at_k20",
    "coverage_p25_at_k20",
    "genericity_mean_at_k20",
    "source_env",
    "context_family",
    "config_path",
    "output_root",
    "status",
    "attempt",
    "duration_seconds",
    "synthetic_train_count",
    "eval_count",
    "best_top3",
    "best_top5",
    "best_top10",
]


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _percentile_nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(float(value) for value in values)
    if percentile <= 0:
        return float(sorted_values[0])
    if percentile >= 100:
        return float(sorted_values[-1])
    rank = int(math.ceil((float(percentile) / 100.0) * len(sorted_values)))
    return float(sorted_values[max(0, rank - 1)])


def read_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Missing required artifact: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def initialize_tsv(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if not resolved.exists():
        resolved.write_text("\t".join(SUMMARY_FIELDS) + "\n", encoding="utf-8")
    return resolved


def append_tsv_row(path: str | Path, row: dict[str, Any]) -> None:
    initialize_tsv(path)
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write("\t".join(str(row.get(field, "")) for field in SUMMARY_FIELDS) + "\n")


def append_jsonl_row(path: str | Path, row: dict[str, Any]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def compute_reward(best_top1: float, normalized_budget_cost: float, reward_lambda: float = 0.002) -> float:
    return float(best_top1) - float(reward_lambda) * float(normalized_budget_cost)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) * float(value) for value in left))
    right_norm = math.sqrt(sum(float(value) * float(value) for value in right))
    denominator = left_norm * right_norm
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def compute_coverage_metrics(
    private_vectors: list[list[float]],
    selected_vectors: list[list[float]],
) -> tuple[float, float]:
    if not private_vectors or not selected_vectors:
        return 0.0, 0.0
    coverage_values = [
        max(cosine_similarity(private_vector, selected_vector) for selected_vector in selected_vectors)
        for private_vector in private_vectors
    ]
    return _mean(coverage_values), _percentile_nearest_rank(coverage_values, 25)


def build_summary_row(
    *,
    experiment_id: str,
    dataset_name: str,
    meta_seed: int,
    action_budget: int,
    normalized_budget_cost: float,
    state_features: dict[str, Any],
    output_root: str,
    config_path: str,
    source_env: str,
    context_family: str,
    attempt: int,
    duration_seconds: float,
    downstream_metrics: dict[str, Any],
) -> dict[str, Any]:
    best_top1 = float(downstream_metrics.get("best_top1", 0.0))
    row = {
        "experiment_id": experiment_id,
        "dataset_name": dataset_name,
        "meta_seed": int(meta_seed),
        "action_budget": int(action_budget),
        "normalized_budget_cost": float(normalized_budget_cost),
        "best_top1": best_top1,
        "reward": compute_reward(best_top1, normalized_budget_cost),
        "shape_score": state_features.get("shape_score", ""),
        "private_mean_length": state_features.get("private_mean_length", ""),
        "private_p75_length": state_features.get("private_p75_length", ""),
        "private_length_iqr": state_features.get("private_length_iqr", ""),
        "support_mean_at_k20": state_features.get("support_mean_at_k20", ""),
        "coverage_mean_at_k20": state_features.get("coverage_mean_at_k20", ""),
        "coverage_p25_at_k20": state_features.get("coverage_p25_at_k20", ""),
        "genericity_mean_at_k20": state_features.get("genericity_mean_at_k20", ""),
        "source_env": source_env,
        "context_family": context_family,
        "config_path": config_path,
        "output_root": output_root,
        "status": "success",
        "attempt": int(attempt),
        "duration_seconds": round(float(duration_seconds), 3),
        "synthetic_train_count": int(downstream_metrics.get("synthetic_train_count", 0)),
        "eval_count": int(downstream_metrics.get("eval_count", 0)),
        "best_top3": float(downstream_metrics.get("best_top3", 0.0)),
        "best_top5": float(downstream_metrics.get("best_top5", 0.0)),
        "best_top10": float(downstream_metrics.get("best_top10", 0.0)),
    }
    return row


def main() -> int:
    raise SystemExit(
        "append_round22_bandit_summary.py is intended to be imported by "
        "round22_bandit_collection_runner.py"
    )


if __name__ == "__main__":
    main()
