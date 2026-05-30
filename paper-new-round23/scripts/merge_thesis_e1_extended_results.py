#!/usr/bin/env python3
"""Merge E1 extended results: original 5 methods + C4-only, Aug-PE, DP-Prompt.

Reads five summary TSV sources (round19, round23, c4only, augpe, dpprompt) and
produces two output files:
  - e1_extended_method_dataset_summary.tsv   per-(method, dataset) mean metrics
  - e1_extended_paper_table.tsv              paper-ready pivot (methods x datasets)
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


ROUND23_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROUND23_ROOT.parent
DATASET_ORDER = ("jobs", "congressional", "forums", "microblog")
METHOD_ORDER = (
    "PrE-Text",
    "round19",
    "WASP",
    "DPGA-TextSyn",
    "round23",
    "C4-only",
    "Aug-PE",
    "DP-Prompt",
)
METRICS = ("best_top1", "best_top3", "best_top5", "best_top10")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

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


def _write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# Row normalizers — one per source TSV format
# ---------------------------------------------------------------------------

def _normalize_round19(row: dict[str, str]) -> dict[str, Any]:
    method = row.get("method_display_name") or row.get("method") or ""
    note_parts = []
    if row.get("mapping_status"):
        note_parts.append(str(row["mapping_status"]))
    if row.get("implementation_key"):
        note_parts.append(f"implementation={row['implementation_key']}")
    return {
        "method": row.get("method", ""),
        "Method": method,
        "dataset": row.get("dataset", row.get("dataset_name", "")),
        "seed": row.get("seed", row.get("meta_seed", "")),
        "status": row.get("status", ""),
        "Note": "; ".join(note_parts),
        **{metric: _float_or_none(row.get(metric)) for metric in METRICS},
    }


def _normalize_round23(row: dict[str, str]) -> dict[str, Any]:
    bundle = row.get("bundle_version", "")
    note = "6-dataset controller"
    if bundle:
        note = f"{note}; {bundle}"
    return {
        "method": "round23",
        "Method": "round23",
        "dataset": row.get("dataset", row.get("dataset_name", "")),
        "seed": row.get("seed", row.get("meta_seed", "")),
        "status": row.get("status", ""),
        "Note": note,
        **{metric: _float_or_none(row.get(metric)) for metric in METRICS},
    }


def _normalize_c4only(row: dict[str, str]) -> dict[str, Any]:
    return {
        "method": "e1_c4only",
        "Method": "C4-only",
        "dataset": row.get("dataset", row.get("dataset_name", "")),
        "seed": row.get("seed", row.get("meta_seed", "")),
        "status": row.get("status", ""),
        "Note": "random C4 sample; eps=inf",
        **{metric: _float_or_none(row.get(metric)) for metric in METRICS},
    }


def _normalize_augpe(row: dict[str, str]) -> dict[str, Any]:
    return {
        "method": "e1_augpe",
        "Method": "Aug-PE",
        "dataset": row.get("dataset", row.get("dataset_name", "")),
        "seed": row.get("seed", row.get("meta_seed", "")),
        "status": row.get("status", ""),
        "Note": "Aug-PE centralized PE; eps=1.29",
        **{metric: _float_or_none(row.get(metric)) for metric in METRICS},
    }


def _normalize_dpprompt(row: dict[str, str]) -> dict[str, Any]:
    return {
        "method": "e1_dpprompt",
        "Method": "DP-Prompt",
        "dataset": row.get("dataset", row.get("dataset_name", "")),
        "seed": row.get("seed", row.get("meta_seed", "")),
        "status": row.get("status", ""),
        "Note": "DP-Prompt zero-shot paraphrase",
        **{metric: _float_or_none(row.get(metric)) for metric in METRICS},
    }


# ---------------------------------------------------------------------------
# Aggregation and table building (shared with merge_thesis_e1_main_results.py)
# ---------------------------------------------------------------------------

def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["Method"]), str(row["dataset"]))].append(row)

    output: list[dict[str, str]] = []
    for method in METHOD_ORDER:
        for dataset in DATASET_ORDER:
            group = grouped.get((method, dataset), [])
            success = [
                row for row in group
                if str(row.get("status", "")).lower() == "success"
                and row.get("best_top1") is not None
            ]
            notes = sorted({str(row.get("Note", "")) for row in group if row.get("Note")})
            output.append(
                {
                    "Method": method,
                    "dataset": dataset,
                    "best_top1": _mean([float(row["best_top1"]) for row in success if row.get("best_top1") is not None]),
                    "best_top3": _mean([float(row["best_top3"]) for row in success if row.get("best_top3") is not None]),
                    "best_top5": _mean([float(row["best_top5"]) for row in success if row.get("best_top5") is not None]),
                    "best_top10": _mean([float(row["best_top10"]) for row in success if row.get("best_top10") is not None]),
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
    table_rows: list[dict[str, str]] = []
    for method in METHOD_ORDER:
        values: list[float] = []
        row: dict[str, str] = {"Method": method}
        total_success = 0
        total_runs = 0
        notes: set[str] = set()
        for dataset in DATASET_ORDER:
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
        table_rows.append(row)
    return table_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def merge_extended_results(
    *,
    round19_summary: Path,
    round23_summary: Path,
    c4only_summary: Path,
    augpe_summary: Path,
    dpprompt_summary: Path,
    output_dir: Path,
    scale: str = "repeat30",
) -> dict[str, Path]:
    rows: list[dict[str, Any]] = []

    rows.extend(_normalize_round19(row) for row in _read_tsv(round19_summary))
    rows.extend(_normalize_round23(row) for row in _read_tsv(round23_summary))

    for path, normalizer, label in [
        (c4only_summary, _normalize_c4only, "c4only"),
        (augpe_summary, _normalize_augpe, "augpe"),
        (dpprompt_summary, _normalize_dpprompt, "dpprompt"),
    ]:
        if path.exists():
            rows.extend(normalizer(row) for row in _read_tsv(path))
        else:
            print(f"[warn] {label} summary not found, skipping: {path}")

    method_dataset_rows = _aggregate(rows)
    paper_table_rows = _build_paper_table(method_dataset_rows)

    method_dataset_path = output_dir / f"e1_extended_{scale}_method_dataset_summary.tsv"
    paper_table_path = output_dir / f"e1_extended_{scale}_paper_table.tsv"
    _write_tsv(method_dataset_path, method_dataset_rows)
    _write_tsv(paper_table_path, paper_table_rows)
    return {
        "method_dataset_summary": method_dataset_path,
        "paper_table": paper_table_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge E1 extended results (8 methods)")
    parser.add_argument(
        "--scale",
        default="repeat30",
        help="Scale label used in output filenames (default: repeat30)",
    )
    parser.add_argument("--round19-summary", default=None)
    parser.add_argument("--round23-summary", default=None)
    parser.add_argument("--c4only-summary", default=None)
    parser.add_argument("--augpe-summary", default=None)
    parser.add_argument("--dpprompt-summary", default=None)
    parser.add_argument("--output-dir", default=str(ROUND23_ROOT / "logs"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logs = ROUND23_ROOT / "logs"
    round19_logs = REPO_ROOT / "paper-new-round19" / "logs"

    scale = args.scale

    round19_summary = (
        Path(args.round19_summary)
        if args.round19_summary
        else round19_logs / f"thesis_e1_main_seen_{scale}_summary.tsv"
    )
    round23_summary = (
        Path(args.round23_summary)
        if args.round23_summary
        else logs / f"round23_thesis_main_seen_{scale}_summary.tsv"
    )
    c4only_summary = (
        Path(args.c4only_summary)
        if args.c4only_summary
        else logs / f"round23_e1_c4only_seen_{scale}_summary.tsv"
    )
    augpe_summary = (
        Path(args.augpe_summary)
        if args.augpe_summary
        else logs / f"round23_e1_augpe_seen_{scale}_summary.tsv"
    )
    dpprompt_summary = (
        Path(args.dpprompt_summary)
        if args.dpprompt_summary
        else logs / f"round23_e1_dpprompt_seen_{scale}_summary.tsv"
    )

    outputs = merge_extended_results(
        round19_summary=round19_summary,
        round23_summary=round23_summary,
        c4only_summary=c4only_summary,
        augpe_summary=augpe_summary,
        dpprompt_summary=dpprompt_summary,
        output_dir=Path(args.output_dir),
        scale=scale,
    )
    print(f"method_dataset_summary={outputs['method_dataset_summary']}")
    print(f"paper_table={outputs['paper_table']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
