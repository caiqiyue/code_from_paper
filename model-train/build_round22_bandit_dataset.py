from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import (
    BUDGETS,
    DEFAULT_DATASET_DIR,
    DEFAULT_SCHEMA_YAML,
    DEFAULT_SUMMARY_JSONL,
    almost_equal,
    as_float,
    as_int,
    compute_reward,
    context_id,
    dump_json,
    load_yaml,
    maybe_write_parquet,
    normalize_record_key,
    read_jsonl,
    write_csv,
    write_jsonl,
)
from features import STATE_FEATURES


def _required_fields(schema: dict[str, Any]) -> list[str]:
    return (
        list(schema.get("provenance_fields", []))
        + list(schema.get("state_fields", []))
        + list(schema.get("reward_fields", []))
        + list(schema.get("derived_fields", []))
    )


def _normalize_action_row(
    row: dict[str, Any],
    *,
    reward_tolerance: float,
    reward_lambda: float,
) -> dict[str, Any]:
    normalized = dict(row)
    normalized["dataset_name"] = str(row["dataset_name"])
    normalized["meta_seed"] = as_int(row["meta_seed"])
    normalized["action_budget"] = as_int(row["action_budget"])
    normalized["normalized_budget_cost"] = as_float(row["normalized_budget_cost"])
    normalized["best_top1"] = as_float(row["best_top1"])
    for field in ("best_top3", "best_top5", "best_top10"):
        if field in row and row[field] != "":
            normalized[field] = as_float(row[field])
    for field in ("synthetic_train_count", "eval_count", "attempt"):
        if field in row and row[field] != "":
            normalized[field] = as_int(row[field])
    if "duration_seconds" in row and row["duration_seconds"] != "":
        normalized["duration_seconds"] = as_float(row["duration_seconds"])
    for field in STATE_FEATURES:
        normalized[field] = as_float(row[field])

    recomputed_reward = compute_reward(
        float(normalized["best_top1"]),
        float(normalized["normalized_budget_cost"]),
        reward_lambda=reward_lambda,
    )
    upstream_reward = as_float(row["reward"])
    if not almost_equal(recomputed_reward, upstream_reward, reward_tolerance):
        raise ValueError(
            "Reward mismatch for "
            f"{row.get('experiment_id')}: upstream={upstream_reward} recomputed={recomputed_reward}"
        )
    normalized["reward"] = recomputed_reward
    normalized["context_id"] = context_id(normalized["dataset_name"], normalized["meta_seed"])
    return normalized


def build_dataset(
    *,
    summary_jsonl: str | Path,
    schema_yaml: str | Path,
    output_dir: str | Path,
    reward_tolerance: float = 1e-9,
) -> dict[str, Any]:
    schema = load_yaml(schema_yaml)
    rows = read_jsonl(summary_jsonl)
    reward_lambda = as_float(schema.get("reward_lambda", 0.002))
    required = _required_fields(schema)
    if not rows:
        raise ValueError("No rows found in summary jsonl")

    for index, row in enumerate(rows, start=1):
        missing = [field for field in required if field not in row]
        if missing:
            raise KeyError(f"Row {index} missing required fields: {missing}")

    action_rows = [
        _normalize_action_row(
            row,
            reward_tolerance=reward_tolerance,
            reward_lambda=reward_lambda,
        )
        for row in rows
    ]
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in action_rows:
        grouped[normalize_record_key(row["dataset_name"], row["meta_seed"])].append(row)

    context_rows: list[dict[str, Any]] = []
    for (dataset_name, meta_seed), bucket_rows in sorted(grouped.items()):
        budgets = sorted(int(row["action_budget"]) for row in bucket_rows)
        if budgets != BUDGETS:
            raise ValueError(
                f"Context {dataset_name}_seed{meta_seed} does not cover budgets {BUDGETS}: got {budgets}"
            )

        anchor = bucket_rows[0]
        for other in bucket_rows[1:]:
            for field in STATE_FEATURES:
                if not almost_equal(float(anchor[field]), float(other[field]), 1e-9):
                    raise ValueError(
                        f"State feature drift within context {dataset_name}_seed{meta_seed}: {field}"
                    )

        by_budget = {int(row["action_budget"]): row for row in bucket_rows}
        oracle_budget = max(BUDGETS, key=lambda budget: float(by_budget[budget]["reward"]))
        context_row = {
            "context_id": context_id(dataset_name, meta_seed),
            "dataset_name": dataset_name,
            "meta_seed": meta_seed,
            **{field: float(anchor[field]) for field in STATE_FEATURES},
            **{f"reward_k{budget}": float(by_budget[budget]["reward"]) for budget in BUDGETS},
            **{f"best_top1_k{budget}": float(by_budget[budget]["best_top1"]) for budget in BUDGETS},
            "oracle_best_k": int(oracle_budget),
            "oracle_best_reward": float(by_budget[oracle_budget]["reward"]),
        }
        context_rows.append(context_row)

    output_root = Path(output_dir)
    action_fields = [
        "context_id",
        "experiment_id",
        "dataset_name",
        "meta_seed",
        "context_family",
        "source_env",
        "action_budget",
        "normalized_budget_cost",
        *STATE_FEATURES,
        "best_top1",
        "best_top3",
        "best_top5",
        "best_top10",
        "synthetic_train_count",
        "eval_count",
        "reward",
        "config_path",
        "output_root",
        "status",
        "attempt",
        "duration_seconds",
    ]
    context_fields = [
        "context_id",
        "dataset_name",
        "meta_seed",
        *STATE_FEATURES,
        *[f"reward_k{budget}" for budget in BUDGETS],
        *[f"best_top1_k{budget}" for budget in BUDGETS],
        "oracle_best_k",
        "oracle_best_reward",
    ]

    action_jsonl = output_root / "round22_action_samples.jsonl"
    action_csv = output_root / "round22_action_samples.csv"
    context_jsonl = output_root / "round22_context_table.jsonl"
    context_csv = output_root / "round22_context_table.csv"

    sorted_actions = sorted(
        action_rows,
        key=lambda row: (str(row["dataset_name"]), int(row["meta_seed"]), int(row["action_budget"])),
    )
    sorted_contexts = sorted(
        context_rows,
        key=lambda row: (str(row["dataset_name"]), int(row["meta_seed"])),
    )

    write_jsonl(action_jsonl, sorted_actions)
    write_csv(action_csv, sorted_actions, action_fields)
    write_jsonl(context_jsonl, sorted_contexts)
    write_csv(context_csv, sorted_contexts, context_fields)

    action_parquet_written = maybe_write_parquet(output_root / "round22_action_samples.parquet", sorted_actions)
    context_parquet_written = maybe_write_parquet(output_root / "round22_context_table.parquet", sorted_contexts)

    report = {
        "summary_jsonl": str(Path(summary_jsonl).resolve()),
        "schema_yaml": str(Path(schema_yaml).resolve()),
        "action_sample_count": len(sorted_actions),
        "context_count": len(sorted_contexts),
        "expected_action_count_per_context": len(BUDGETS),
        "reward_lambda": reward_lambda,
        "reward_tolerance": reward_tolerance,
        "datasets": {
            dataset: sum(1 for row in sorted_contexts if row["dataset_name"] == dataset)
            for dataset in sorted({row["dataset_name"] for row in sorted_contexts})
        },
        "outputs": {
            "action_jsonl": str(action_jsonl),
            "action_csv": str(action_csv),
            "context_jsonl": str(context_jsonl),
            "context_csv": str(context_csv),
            "action_parquet_written": action_parquet_written,
            "context_parquet_written": context_parquet_written,
        },
    }
    dump_json(output_root / "round22_dataset_build_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build round22 contextual-bandit dataset tables.")
    parser.add_argument("--summary-jsonl", default=str(DEFAULT_SUMMARY_JSONL))
    parser.add_argument("--schema-yaml", default=str(DEFAULT_SCHEMA_YAML))
    parser.add_argument("--output-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--reward-tolerance", type=float, default=1e-9)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_dataset(
        summary_jsonl=args.summary_jsonl,
        schema_yaml=args.schema_yaml,
        output_dir=args.output_dir,
        reward_tolerance=args.reward_tolerance,
    )
    print(
        f"BUILT action_samples={report['action_sample_count']} contexts={report['context_count']} "
        f"output_dir={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
