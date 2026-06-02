#!/usr/bin/env python3
"""Merge round23 E4 one-shot and formal two-round summaries into table-ready outputs."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


ROUND23_ROOT = Path(__file__).resolve().parents[1]
DATASET_ORDER_ALL6 = ("jobs", "congressional", "forums", "microblog", "imdb", "openreview")
METHOD_ORDER = ("predict absolute k", "round23")
METRICS = ("best_top1", "best_top3", "best_top5", "best_top10", "duration_seconds")
DEFAULT_REFERENCE_BUDGET = 20
DEFAULT_FORMAL_ROUND23_BUNDLE = "round23_controller_1200_all6_top1_delta_m0005_extratrees_no_dataset"


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
    if method == "round23_absk_oneshot":
        return "predict absolute k"
    if method == "round23":
        return "round23"
    raise ValueError(f"Unsupported E4 method in summary TSV: {method}")


def _normalize_rows(
    rows: list[dict[str, str]],
    *,
    source_name: str,
    expected_method: str,
    expected_mode_prefixes: tuple[str, ...] | None,
    expected_reference_budget: int | None,
    expected_bundle: str | None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        raw_method = str(row.get("method", ""))
        if raw_method != expected_method:
            raise ValueError(f"{source_name}: expected method={expected_method}, got {raw_method}")
        method = _normalize_method(raw_method)
        dataset = str(row.get("dataset_name", row.get("dataset", "")))
        if dataset and dataset not in DATASET_ORDER_ALL6:
            raise ValueError(f"{source_name}: unsupported E4 dataset in summary TSV: {dataset}")
        mode = str(row.get("mode", "")).strip()
        if expected_mode_prefixes and mode and not any(mode.startswith(prefix) for prefix in expected_mode_prefixes):
            raise ValueError(f"{source_name}: expected mode prefix in {expected_mode_prefixes}, got {mode}")
        raw_reference_budget = str(row.get("reference_budget", "")).strip()
        if expected_reference_budget is not None and raw_reference_budget:
            if int(raw_reference_budget) != int(expected_reference_budget):
                raise ValueError(
                    f"{source_name}: expected reference_budget={expected_reference_budget}, got {raw_reference_budget}"
                )
        note_parts: list[str] = []
        if method == "predict absolute k":
            note_parts.append("one-shot absolute-k")
        bundle = str(row.get("bundle_version", "")).strip()
        if expected_bundle is not None and bundle and bundle != expected_bundle:
            raise ValueError(f"{source_name}: expected bundle_version={expected_bundle}, got {bundle}")
        if bundle:
            note_parts.append(bundle)
        normalized.append(
            {
                "method": method,
                "dataset": dataset,
                "meta_seed": str(row.get("meta_seed", "")),
                "status": str(row.get("status", "")),
                "best_top1": _float_or_none(row.get("best_top1")),
                "best_top3": _float_or_none(row.get("best_top3")),
                "best_top5": _float_or_none(row.get("best_top5")),
                "best_top10": _float_or_none(row.get("best_top10")),
                "duration_seconds": _float_or_none(row.get("duration_seconds")),
                "note": "; ".join(note_parts),
            }
        )
    return normalized


def _dedupe_latest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["method"]), str(row["dataset"]), str(row.get("meta_seed", "")))
        latest[key] = row
    return list(latest.values())


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["dataset"]))].append(row)

    output: list[dict[str, str]] = []
    for method in METHOD_ORDER:
        for dataset in DATASET_ORDER_ALL6:
            group = grouped.get((method, dataset), [])
            success = [
                row for row in group
                if str(row.get("status", "")).lower() == "success" and row.get("best_top1") is not None
            ]
            notes = sorted({str(row.get("note", "")) for row in group if row.get("note")})
            output.append(
                {
                    "Method": method,
                    "dataset": dataset,
                    "best_top1": _mean([float(row["best_top1"]) for row in success if row.get("best_top1") is not None]),
                    "best_top3": _mean([float(row["best_top3"]) for row in success if row.get("best_top3") is not None]),
                    "best_top5": _mean([float(row["best_top5"]) for row in success if row.get("best_top5") is not None]),
                    "best_top10": _mean([float(row["best_top10"]) for row in success if row.get("best_top10") is not None]),
                    "duration_seconds": _mean(
                        [float(row["duration_seconds"]) for row in success if row.get("duration_seconds") is not None]
                    ),
                    "n_success": str(len(success)),
                    "n_total": str(len(group)),
                    "Note": "; ".join(notes),
                }
            )
    return output


def _build_paper_table(method_dataset_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_method_dataset = {
        (row["Method"], row["dataset"]): row
        for row in method_dataset_rows
    }
    output: list[dict[str, str]] = []
    for method in METHOD_ORDER:
        row: dict[str, str] = {"Method": method}
        values: list[float] = []
        total_success = 0
        total_runs = 0
        notes: set[str] = set()
        for dataset in DATASET_ORDER_ALL6:
            md = by_method_dataset.get((method, dataset), {})
            value = md.get("best_top1", "")
            row[f"{dataset} best_top1"] = value
            numeric = _float_or_none(value)
            if numeric is not None:
                values.append(numeric)
            total_success += int(md.get("n_success", "0") or 0)
            total_runs += int(md.get("n_total", "0") or 0)
            if md.get("Note"):
                notes.add(str(md["Note"]))
        row["Avg."] = _mean(values)
        row["n_success"] = str(total_success)
        row["n_total"] = str(total_runs)
        row["Note"] = "; ".join(sorted(notes))
        output.append(row)
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
    oneshot_summary: Path,
    round23_summaries: list[Path],
    round23_mode_prefixes: tuple[str, ...],
    output_dir: Path,
    output_prefix: str = "round23_e4",
    round23_bundle_version: str = DEFAULT_FORMAL_ROUND23_BUNDLE,
) -> dict[str, Path]:
    rows = _normalize_rows(
        _read_tsv(oneshot_summary),
        source_name="oneshot_summary",
        expected_method="round23_absk_oneshot",
        expected_mode_prefixes=("e4_a_oneshot_",),
        expected_reference_budget=DEFAULT_REFERENCE_BUDGET,
        expected_bundle=None,
    )
    for round23_summary in round23_summaries:
        rows.extend(
            _normalize_rows(
                _read_tsv(round23_summary),
                source_name=f"round23_summary:{round23_summary.name}",
                expected_method="round23",
                expected_mode_prefixes=round23_mode_prefixes,
                expected_reference_budget=DEFAULT_REFERENCE_BUDGET,
                expected_bundle=round23_bundle_version,
            )
        )

    method_dataset_rows = _aggregate(_dedupe_latest(rows))
    paper_table_rows = _build_paper_table(method_dataset_rows)

    method_dataset_path = output_dir / f"{output_prefix}_method_dataset_summary.tsv"
    paper_table_path = output_dir / f"{output_prefix}_paper_table.tsv"
    _write_tsv(method_dataset_path, method_dataset_rows)
    _write_tsv(paper_table_path, paper_table_rows)
    return {
        "method_dataset_summary": method_dataset_path,
        "paper_table": paper_table_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge round23 E4 one-shot and two-round summaries")
    parser.add_argument("--oneshot-summary", required=True)
    parser.add_argument("--round23-summary", dest="round23_summaries", action="append", required=True)
    parser.add_argument("--round23-mode-prefix", dest="round23_mode_prefixes", action="append", required=True)
    parser.add_argument("--round23-bundle-version", default=DEFAULT_FORMAL_ROUND23_BUNDLE)
    parser.add_argument("--output-dir", default=str(ROUND23_ROOT / "logs"))
    parser.add_argument("--output-prefix", default="round23_e4")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = merge_results(
        oneshot_summary=Path(args.oneshot_summary),
        round23_summaries=[Path(path) for path in args.round23_summaries],
        round23_mode_prefixes=tuple(args.round23_mode_prefixes),
        output_dir=Path(args.output_dir),
        output_prefix=args.output_prefix,
        round23_bundle_version=str(args.round23_bundle_version),
    )
    for key, value in outputs.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
