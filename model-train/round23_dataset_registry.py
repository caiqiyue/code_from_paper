from __future__ import annotations

from collections import defaultdict
from typing import Any


FORMAL_ROUND23_DATASET_ORDER = [
    "jobs",
    "congressional",
    "forums",
    "microblog",
    "imdb",
    "openreview",
]
FORMAL_ROUND23_TRAIN_DATASETS = [
    "jobs",
    "congressional",
    "forums",
    "microblog",
]
FORMAL_ROUND23_UNSEEN_TEST_DATASETS = [
    "imdb",
    "openreview",
]


def require_formal_dataset_name(dataset_name: str) -> str:
    normalized = str(dataset_name).strip()
    if normalized not in FORMAL_ROUND23_DATASET_ORDER:
        raise ValueError(f"Unsupported round23 formal dataset: {dataset_name}")
    return normalized


def get_dataset_partition(dataset_name: str) -> str:
    normalized = require_formal_dataset_name(dataset_name)
    if normalized in FORMAL_ROUND23_TRAIN_DATASETS:
        return "train"
    return "unseen_test"


def get_formal_onehot_order() -> list[str]:
    return list(FORMAL_ROUND23_DATASET_ORDER)


def summarize_dataset_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "train": {"datasets": {}, "context_count": 0},
        "unseen_test": {"datasets": {}, "context_count": 0},
    }
    seen_context_ids: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        dataset_name = require_formal_dataset_name(str(row["dataset_name"]))
        partition = get_dataset_partition(dataset_name)
        context_key = str(row["context_id"])
        seen_context_ids[partition].add(context_key)
        summary[partition]["datasets"].setdefault(dataset_name, 0)
        summary[partition]["datasets"][dataset_name] += 1
    for partition in ("train", "unseen_test"):
        summary[partition]["context_count"] = len(seen_context_ids[partition])
    return summary
