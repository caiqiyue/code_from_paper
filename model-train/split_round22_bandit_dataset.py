from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import DEFAULT_DATASET_DIR, DEFAULT_SPLIT_DIR, dump_json, read_csv, read_jsonl


DEFAULT_FINAL_TEST_COUNTS = {
    "jobs": 4,
    "forums": 4,
    "congressional": 6,
    "microblog": 6,
}


def _read_context_rows(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path)
    if resolved.suffix.lower() == ".jsonl":
        return read_jsonl(resolved)
    if resolved.suffix.lower() == ".csv":
        return read_csv(resolved)
    raise ValueError(f"Unsupported context table format: {resolved}")


def build_splits(
    *,
    context_table_path: str | Path,
    output_dir: str | Path,
    random_seed: int = 42,
    fold_count: int = 5,
    final_test_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    final_test_counts = dict(final_test_counts or DEFAULT_FINAL_TEST_COUNTS)
    rows = _read_context_rows(context_table_path)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset_name"])].append(row)

    rng = random.Random(random_seed)
    final_test_ids: list[str] = []
    cv_contexts: list[dict[str, Any]] = []
    per_dataset_report: dict[str, Any] = {}

    for dataset_name, dataset_rows in sorted(by_dataset.items()):
        shuffled = list(dataset_rows)
        rng.shuffle(shuffled)
        needed = int(final_test_counts.get(dataset_name, 0))
        if len(shuffled) < needed:
            raise ValueError(
                f"Dataset {dataset_name} has only {len(shuffled)} contexts, cannot hold out {needed}"
            )
        final_part = shuffled[:needed]
        dev_part = shuffled[needed:]
        final_test_ids.extend(str(row["context_id"]) for row in final_part)
        cv_contexts.extend(dev_part)
        per_dataset_report[dataset_name] = {
            "total_contexts": len(shuffled),
            "final_test_contexts": len(final_part),
            "dev_contexts": len(dev_part),
        }

    folds: list[dict[str, Any]] = [
        {"fold_index": index, "validation_context_ids": [], "training_context_ids": []}
        for index in range(fold_count)
    ]
    dev_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cv_contexts:
        dev_by_dataset[str(row["dataset_name"])].append(row)

    fold_offset = 0
    for dataset_name, dataset_rows in sorted(dev_by_dataset.items()):
        shuffled = list(dataset_rows)
        rng.shuffle(shuffled)
        for index, row in enumerate(shuffled):
            target_fold = (fold_offset + index) % fold_count
            folds[target_fold]["validation_context_ids"].append(str(row["context_id"]))
        fold_offset = (fold_offset + len(shuffled)) % fold_count

    all_dev_ids = {str(row["context_id"]) for row in cv_contexts}
    for fold in folds:
        val_ids = set(fold["validation_context_ids"])
        fold["validation_context_ids"] = sorted(val_ids)
        fold["training_context_ids"] = sorted(all_dev_ids - val_ids)

    output_root = Path(output_dir)
    final_payload = {
        "random_seed": random_seed,
        "context_table_path": str(Path(context_table_path).resolve()),
        "final_test_context_ids": sorted(final_test_ids),
        "counts_by_dataset": final_test_counts,
    }
    cv_payload = {
        "random_seed": random_seed,
        "fold_count": fold_count,
        "folds": folds,
    }
    report = {
        "context_count": len(rows),
        "final_test_count": len(final_test_ids),
        "dev_context_count": len(cv_contexts),
        "fold_count": fold_count,
        "per_dataset": per_dataset_report,
    }

    dump_json(output_root / "round22_final_test_contexts.json", final_payload)
    dump_json(output_root / "round22_cv_folds.json", cv_payload)
    dump_json(output_root / "round22_split_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create round22 context-level train/val/test splits.")
    parser.add_argument(
        "--context-table",
        default=str(DEFAULT_DATASET_DIR / "round22_context_table.jsonl"),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_SPLIT_DIR))
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument(
        "--final-test-counts-json",
        default=None,
        help="Optional JSON object or JSON file path overriding per-dataset final test counts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    final_test_counts = None
    if args.final_test_counts_json:
        override = str(args.final_test_counts_json)
        candidate = Path(override)
        if candidate.exists():
            final_test_counts = json.loads(candidate.read_text(encoding="utf-8-sig"))
        else:
            final_test_counts = json.loads(override)
    report = build_splits(
        context_table_path=args.context_table,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
        fold_count=args.fold_count,
        final_test_counts=final_test_counts,
    )
    print(
        f"SPLIT contexts={report['context_count']} final_test={report['final_test_count']} "
        f"dev={report['dev_context_count']} folds={report['fold_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
