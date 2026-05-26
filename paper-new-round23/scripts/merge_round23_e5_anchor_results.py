#!/usr/bin/env python3
"""Merge round23 E5 anchor-boundary summaries into table-ready outputs."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


ROUND23_ROOT = Path(__file__).resolve().parents[1]
DATASET_ORDER_SEEN4 = ("jobs", "congressional", "forums", "microblog")
DATASET_ORDER_ALL6 = ("jobs", "congressional", "forums", "microblog", "imdb", "openreview")
ANCHOR_ORDER = (19, 20, 21)
METHOD_ORDER = ("round23_keepk0", "round23")


def _read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required summary TSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _float_or_none(value: Any) -> float | None:
    if value in (None, "", "NA"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> str:
    if not values:
        return ""
    return f"{sum(values) / len(values):.6f}"


def _normalize_method(method: str) -> str:
    if method == "round23_keepk0":
        return "round23_keepk0"
    return "round23"


def _normalize_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                "anchor": int(row.get("reference_budget") or 0),
                "method": _normalize_method(str(row.get("method", ""))),
                "dataset": row.get("dataset_name", ""),
                "status": str(row.get("status", "")),
                "best_top1": _float_or_none(row.get("best_top1")),
                "best_top3": _float_or_none(row.get("best_top3")),
                "best_top5": _float_or_none(row.get("best_top5")),
                "best_top10": _float_or_none(row.get("best_top10")),
            }
        )
    return normalized


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["anchor"]), str(row["method"]), str(row["dataset"]))].append(row)

    output: list[dict[str, str]] = []
    for anchor in ANCHOR_ORDER:
        for method in METHOD_ORDER:
            for dataset in DATASET_ORDER_ALL6:
                group = grouped.get((anchor, method, dataset), [])
                success = [
                    row for row in group
                    if str(row.get("status", "")).lower() == "success" and row.get("best_top1") is not None
                ]
                output.append(
                    {
                        "anchor": str(anchor),
                        "method": method,
                        "dataset": dataset,
                        "best_top1": _mean([float(row["best_top1"]) for row in success if row.get("best_top1") is not None]),
                        "best_top3": _mean([float(row["best_top3"]) for row in success if row.get("best_top3") is not None]),
                        "best_top5": _mean([float(row["best_top5"]) for row in success if row.get("best_top5") is not None]),
                        "best_top10": _mean([float(row["best_top10"]) for row in success if row.get("best_top10") is not None]),
                        "n_success": str(len(success)),
                        "n_total": str(len(group)),
                    }
                )
    return output


def _build_table(method_dataset_rows: list[dict[str, str]], datasets: tuple[str, ...]) -> list[dict[str, str]]:
    by_key = {
        (int(row["anchor"]), row["method"], row["dataset"]): row
        for row in method_dataset_rows
    }
    output: list[dict[str, str]] = []
    for anchor in ANCHOR_ORDER:
        keep_avg_numeric: float | None = None
        cached_rows: dict[str, dict[str, str]] = {}
        for method in METHOD_ORDER:
            row: dict[str, str] = {
                "anchor": str(anchor),
                "method": method,
            }
            values: list[float] = []
            total_success = 0
            total_runs = 0
            for dataset in datasets:
                md = by_key.get((anchor, method, dataset), {})
                value = md.get("best_top1", "")
                row[f"{dataset} best_top1"] = value
                numeric = _float_or_none(value)
                if numeric is not None:
                    values.append(numeric)
                total_success += int(md.get("n_success", "0") or 0)
                total_runs += int(md.get("n_total", "0") or 0)
            avg_text = _mean(values)
            row["Avg."] = avg_text
            row["n_success"] = str(total_success)
            row["n_total"] = str(total_runs)
            row["Gain vs keep-k0"] = ""
            cached_rows[method] = row
            if method == "round23_keepk0":
                keep_avg_numeric = _float_or_none(avg_text)
            output.append(row)
        round23_row = cached_rows.get("round23")
        if round23_row is not None:
            round23_avg = _float_or_none(round23_row["Avg."])
            if round23_avg is not None and keep_avg_numeric is not None:
                round23_row["Gain vs keep-k0"] = f"{round23_avg - keep_avg_numeric:.6f}"
    return output


def _write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def merge_results(
    *,
    summary_paths: list[Path],
    output_dir: Path,
    output_prefix: str = "round23_e5_anchor",
) -> dict[str, Path]:
    rows: list[dict[str, Any]] = []
    for path in summary_paths:
        rows.extend(_normalize_rows(_read_tsv(path)))

    method_dataset_rows = _aggregate(rows)
    seen4_table = _build_table(method_dataset_rows, DATASET_ORDER_SEEN4)
    all6_table = _build_table(method_dataset_rows, DATASET_ORDER_ALL6)

    method_dataset_path = output_dir / f"{output_prefix}_method_dataset_summary.tsv"
    seen4_path = output_dir / f"{output_prefix}_seen4_table.tsv"
    all6_path = output_dir / f"{output_prefix}_all6_table.tsv"
    _write_tsv(method_dataset_path, method_dataset_rows)
    _write_tsv(seen4_path, seen4_table)
    _write_tsv(all6_path, all6_table)
    return {
        "method_dataset_summary": method_dataset_path,
        "seen4_table": seen4_path,
        "all6_table": all6_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge round23 E5 anchor-boundary summaries")
    parser.add_argument(
        "--summary",
        action="append",
        dest="summaries",
        default=[],
        help="Summary TSV path. May be repeated.",
    )
    parser.add_argument("--output-dir", default=str(ROUND23_ROOT / "logs"))
    parser.add_argument("--output-prefix", default="round23_e5_anchor")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.summaries:
        raise SystemExit("At least one --summary path is required.")
    outputs = merge_results(
        summary_paths=[Path(item) for item in args.summaries],
        output_dir=Path(args.output_dir),
        output_prefix=args.output_prefix,
    )
    for key, value in outputs.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
