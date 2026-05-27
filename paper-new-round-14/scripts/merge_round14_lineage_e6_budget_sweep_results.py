#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle, delimiter="\t"))
    return [row for row in rows if row.get("status") == "success" and row.get("best_top1") not in ("", None)]


def merge_results(*, summary_paths: list[Path], output_dir: Path, output_prefix: str) -> dict[str, Path]:
    rows = _read_rows(summary_paths)
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset_name"], int(row["budget"]))].append(float(row["best_top1"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / f"{output_prefix}_seen4_budget_sweep.tsv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "budget", "mean_best_top1", "runs"], delimiter="\t")
        writer.writeheader()
        for (dataset, budget), values in sorted(grouped.items()):
            writer.writerow(
                {
                    "dataset": dataset,
                    "budget": budget,
                    "mean_best_top1": f"{sum(values) / len(values):.6f}",
                    "runs": len(values),
                }
            )
    return {"seen4_table": table_path}


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge round14-lineage E6 budget sweep summaries.")
    parser.add_argument("--summary-path", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default="round14_lineage_e6")
    args = parser.parse_args()
    outputs = merge_results(
        summary_paths=[Path(item) for item in args.summary_path],
        output_dir=Path(args.output_dir),
        output_prefix=args.output_prefix,
    )
    for name, path in outputs.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
