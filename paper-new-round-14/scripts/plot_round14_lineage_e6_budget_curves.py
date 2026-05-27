#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def plot_table(table_path: Path, output_path: Path) -> None:
    series: dict[str, list[tuple[int, float]]] = {}
    with table_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            series.setdefault(row["dataset"], []).append((int(row["budget"]), float(row["mean_best_top1"])))
    plt.figure(figsize=(8, 5))
    for dataset, values in sorted(series.items()):
        values = sorted(values)
        plt.plot([item[0] for item in values], [item[1] for item in values], marker="o", label=dataset)
    plt.xlabel("seed_top_k")
    plt.ylabel("best_top1")
    plt.title("Round14 lineage E6 budget sensitivity")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot round14-lineage E6 budget sweep curves.")
    parser.add_argument("--input-table", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    plot_table(Path(args.input_table), Path(args.output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
