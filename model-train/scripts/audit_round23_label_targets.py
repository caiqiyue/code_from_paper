from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
MODEL_TRAIN_ROOT = THIS_DIR.parent

if str(MODEL_TRAIN_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(MODEL_TRAIN_ROOT))

from common import (
    BUDGETS,
    DELTA_ACTIONS,
    REFERENCE_BUDGET,
    compute_round23_controller_reward,
    compute_round23_training_target,
    context_id,
    controller_target_budget,
    dump_json,
    ensure_dir,
    read_jsonl,
    select_round23_oracle_delta,
    write_jsonl,
)
from round23_dataset_registry import get_dataset_partition


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return payload


def _budget_metric(row: dict[str, Any], *field_names: str, default: float | None = None) -> float:
    for field_name in field_names:
        if field_name in row and row[field_name] not in (None, ""):
            return float(row[field_name])
    if default is not None:
        return float(default)
    raise KeyError(f"Missing required budget metric. Tried: {field_names}")


def _discover_context_bundles(records_root: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(records_root)
    grouped: dict[str, dict[str, Any]] = {}
    for final_path in root.rglob("final_result_summary.json"):
        collection_dir = final_path.parent
        context_path = collection_dir / "context_summary.json"
        budget_path = collection_dir / "budget_table.jsonl"
        if not context_path.exists() or not budget_path.exists():
            continue
        context_summary = _read_json(context_path)
        final_result = _read_json(final_path)
        budget_rows = read_jsonl(budget_path)
        budget_row = budget_rows[0] if len(budget_rows) == 1 else None
        if budget_row is None:
            candidates = [row for row in budget_rows if int(row.get("budget_k", row.get("seed_top_k", -1))) in BUDGETS]
            if len(candidates) == 1:
                budget_row = candidates[0]
        if budget_row is None:
            continue
        budget_k = int(budget_row.get("budget_k", budget_row.get("seed_top_k", budget_row.get("selected_seed_count"))))
        dataset_name = str(context_summary["dataset_name"])
        meta_seed = int(context_summary["meta_seed"])
        key = str(context_summary.get("context_id") or context_id(dataset_name, meta_seed))
        bucket = grouped.setdefault(
            key,
            {
                "context_summary": dict(context_summary, context_id=key, dataset_name=dataset_name, meta_seed=meta_seed),
                "budgets": {},
            },
        )
        bucket["budgets"][budget_k] = {
            "budget_k": budget_k,
            "best_top1": float(final_result["best_top1"]),
            "support_mean": _budget_metric(budget_row, "support_mean_k", "support_mean", "support_score"),
            "coverage_p25": _budget_metric(budget_row, "coverage_p25_k", "coverage_p25"),
            "coverage_mean": _budget_metric(budget_row, "coverage_mean_k", "coverage_mean"),
        }
    return {key: bundle for key, bundle in grouped.items() if all(budget in bundle["budgets"] for budget in BUDGETS)}


def _candidate_specs() -> list[dict[str, Any]]:
    specs = [{"name": "old_reward", "mode": "old_reward", "tie_margin": 0.0}]
    for mode in ("top1_delta", "top1_value"):
        for margin in (0.0, 0.0005, 0.001):
            specs.append({"name": f"{mode}_m{margin:g}", "mode": mode, "tie_margin": margin})
    specs.append({"name": "calibrated_reward_m0.0005", "mode": "calibrated_reward", "tie_margin": 0.0005})
    return specs


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _summarize_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(int(row["oracle_best_delta_k"]) for row in rows)
    by_dataset: dict[str, Any] = {}
    by_partition: dict[str, Any] = {}
    for dataset_name in sorted({str(row["dataset_name"]) for row in rows}):
        ds_rows = [row for row in rows if str(row["dataset_name"]) == dataset_name]
        by_dataset[dataset_name] = dict(Counter(int(row["oracle_best_delta_k"]) for row in ds_rows))
    for partition in sorted({str(row["partition"]) for row in rows}):
        part_rows = [row for row in rows if str(row["partition"]) == partition]
        by_partition[partition] = dict(Counter(int(row["oracle_best_delta_k"]) for row in part_rows))
    target_by_delta: dict[str, Any] = {}
    for delta_k in DELTA_ACTIONS:
        values = [float(row[f"value_dk_{_delta_suffix(delta_k)}"]) for row in rows]
        target_by_delta[str(delta_k)] = {"mean": _mean(values), "median": _median(values)}
    return {
        "context_count": len(rows),
        "oracle_distribution": {str(delta): counts.get(delta, 0) for delta in DELTA_ACTIONS},
        "oracle_distribution_by_dataset": by_dataset,
        "oracle_distribution_by_partition": by_partition,
        "nonzero_action_ratio": _mean([1.0 if int(row["oracle_best_delta_k"]) != 0 else 0.0 for row in rows]),
        "tie_to_zero_ratio": _mean([float(row["tie_to_zero"]) for row in rows]),
        "mean_keep_k0_regret": _mean([float(row["oracle_value"]) - float(row["keep_k0_value"]) for row in rows]),
        "target_by_action": target_by_delta,
    }


def _delta_suffix(delta_k: int) -> str:
    if delta_k < 0:
        return f"neg{abs(delta_k)}"
    if delta_k > 0:
        return f"pos{delta_k}"
    return "0"


def audit_label_targets(*, records_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    bundles = _discover_context_bundles(records_root)
    per_context_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for spec in _candidate_specs():
        variant_rows: list[dict[str, Any]] = []
        for key, bundle in sorted(bundles.items()):
            context_summary = bundle["context_summary"]
            metrics_by_budget = bundle["budgets"]
            best_top1_k0 = float(metrics_by_budget[REFERENCE_BUDGET]["best_top1"])
            value_by_delta: dict[int, float] = {}
            top1_by_delta: dict[int, float] = {}
            raw_top1_best_delta = max(
                DELTA_ACTIONS,
                key=lambda delta_k: float(metrics_by_budget[controller_target_budget(delta_k)]["best_top1"]),
            )
            for delta_k in DELTA_ACTIONS:
                target_budget = controller_target_budget(delta_k)
                target_metrics = metrics_by_budget[target_budget]
                if spec["mode"] == "old_reward":
                    value = compute_round23_controller_reward(
                        best_top1_k0=best_top1_k0,
                        best_top1_k1=float(target_metrics["best_top1"]),
                        coverage_p25_k0=float(metrics_by_budget[REFERENCE_BUDGET]["coverage_p25"]),
                        coverage_p25_k1=float(target_metrics["coverage_p25"]),
                        support_mean_k0=float(metrics_by_budget[REFERENCE_BUDGET]["support_mean"]),
                        support_mean_k1=float(target_metrics["support_mean"]),
                        delta_k=delta_k,
                    )
                else:
                    value = compute_round23_training_target(
                        target_mode=str(spec["mode"]),
                        best_top1_k0=best_top1_k0,
                        best_top1_k1=float(target_metrics["best_top1"]),
                        coverage_p25_k0=float(metrics_by_budget[REFERENCE_BUDGET]["coverage_p25"]),
                        coverage_p25_k1=float(target_metrics["coverage_p25"]),
                        support_mean_k0=float(metrics_by_budget[REFERENCE_BUDGET]["support_mean"]),
                        support_mean_k1=float(target_metrics["support_mean"]),
                        delta_k=delta_k,
                    )
                value_by_delta[int(delta_k)] = float(value)
                top1_by_delta[int(delta_k)] = float(target_metrics["best_top1"])
            oracle_delta = select_round23_oracle_delta(
                action_values=value_by_delta,
                top1_by_delta=top1_by_delta,
                best_top1_k0=best_top1_k0,
                tie_margin=float(spec["tie_margin"]),
            )
            row = {
                "variant": spec["name"],
                "mode": spec["mode"],
                "tie_margin": float(spec["tie_margin"]),
                "context_id": key,
                "dataset_name": str(context_summary["dataset_name"]),
                "meta_seed": int(context_summary["meta_seed"]),
                "partition": get_dataset_partition(str(context_summary["dataset_name"])),
                "oracle_best_delta_k": int(oracle_delta),
                "oracle_best_target_budget": controller_target_budget(oracle_delta),
                "oracle_value": value_by_delta[int(oracle_delta)],
                "keep_k0_value": value_by_delta[0],
                "tie_to_zero": int(raw_top1_best_delta != 0 and oracle_delta == 0),
            }
            for delta_k in DELTA_ACTIONS:
                row[f"value_dk_{_delta_suffix(delta_k)}"] = value_by_delta[int(delta_k)]
                row[f"best_top1_dk_{_delta_suffix(delta_k)}"] = top1_by_delta[int(delta_k)]
            variant_rows.append(row)
            per_context_rows.append(row)
        summary = {"variant": spec["name"], "mode": spec["mode"], "tie_margin": spec["tie_margin"]}
        summary.update(_summarize_variant(variant_rows))
        summaries.append(summary)

    output_root = ensure_dir(output_dir)
    write_jsonl(output_root / "per_context_label_audit.jsonl", per_context_rows)
    dump_json(
        output_root / "label_audit_summary.json",
        {
            "records_root": str(Path(records_root).resolve()),
            "complete_context_count": len(bundles),
            "budget_count_per_context": len(BUDGETS),
            "candidate_count": len(summaries),
            "recommended_primary": "top1_delta_m0.0005",
            "recommended_backup": "top1_value_m0.0005",
            "summaries": summaries,
        },
    )
    _write_summary_csv(output_root / "label_audit_summary.csv", summaries)
    _write_summary_md(output_root / "label_audit_summary.md", summaries)
    return {"complete_context_count": len(bundles), "summaries": summaries}


def _write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fieldnames = [
        "variant",
        "mode",
        "tie_margin",
        "context_count",
        "nonzero_action_ratio",
        "tie_to_zero_ratio",
        "mean_keep_k0_regret",
        "oracle_distribution",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({field: json.dumps(row.get(field), ensure_ascii=False) if field == "oracle_distribution" else row.get(field, "") for field in fieldnames})


def _write_summary_md(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Round23 Label Target Audit",
        "",
        "| Variant | Mode | Margin | Non-zero Ratio | Tie-to-zero Ratio | Mean Keep-k0 Regret | Oracle Distribution |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['variant']} | {row['mode']} | {float(row['tie_margin']):.4f} | "
            f"{float(row['nonzero_action_ratio']):.4f} | {float(row['tie_to_zero_ratio']):.4f} | "
            f"{float(row['mean_keep_k0_regret']):.6f} | `{row['oracle_distribution']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit candidate label targets for round23 controller training.")
    parser.add_argument(
        "--records-root",
        default=str(MODEL_TRAIN_ROOT / "data" / "raw" / "round23_collection_repeat40_records" / "round23_collection_repeat40"),
    )
    parser.add_argument("--output-dir", default=str(MODEL_TRAIN_ROOT / "artifacts" / "round23_label_audit"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_label_targets(records_root=args.records_root, output_dir=args.output_dir)
    print(f"AUDITED round23 label targets contexts={report['complete_context_count']} variants={len(report['summaries'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
