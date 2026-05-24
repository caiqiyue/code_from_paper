#!/usr/bin/env python3
"""Convert round22 bandit 500-experiment data into tree-model RL training format.

Output structure (per split: train/val/test):
  - {split}_contexts.jsonl    # one per (dataset, seed), features + all k rewards + oracle
  - {split}_action_samples.jsonl  # one per experiment, context_id + action + reward

For tree model training, action_samples is the primary dataset:
  Feature vector (8 context features + one-hot) + (action, reward) -> predict best action
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_DIR = DATA_DIR / "raw/full-500"
OUTPUT_DIR = DATA_DIR / "ready/full-500"
SPLITS_DIR = DATA_DIR / "splits/full-500"

CONTEXT_FEATURES = [
    "shape_score",
    "private_mean_length",
    "private_p75_length",
    "private_length_iqr",
    "support_mean_at_k20",
    "coverage_mean_at_k20",
    "coverage_p25_at_k20",
    "genericity_mean_at_k20",
]

DATASET_ORDER = ["jobs", "congressional", "forums", "microblog"]
BUDGETS = [18, 19, 20, 21, 22]


def load_raw_data() -> list[dict]:
    """Load raw experiment records."""
    records = []
    with open(RAW_DIR / "round22_bandit_full_summary.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_contexts(records: list[dict]) -> dict[str, dict]:
    """Group records by context_id = f'{dataset_name}_seed{meta_seed}'.

    Each context has:
      - context_id
      - dataset_name, meta_seed
      - 8 context features (from k=20 reference)
      - reward_k{k} for each k in BUDGETS
      - best_top1_k{k} for each k
      - oracle_best_k (k with highest reward)
      - oracle_best_reward
    """
    # Group by context
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        cid = f"{r['dataset_name']}_seed{r['meta_seed']}"
        grouped[cid].append(r)

    contexts = {}
    for cid, recs in grouped.items():
        if len(recs) != 5:
            print(f"WARNING: {cid} has {len(recs)} records, expected 5")
            continue

        # All records in a context share the same context features (from k=20)
        # Use the k=20 record for features
        k20_rec = next((r for r in recs if r["action_budget"] == 20), recs[0])

        ctx = {
            "context_id": cid,
            "dataset_name": k20_rec["dataset_name"],
            "meta_seed": k20_rec["meta_seed"],
        }

        # Add context features
        for f in CONTEXT_FEATURES:
            ctx[f] = k20_rec[f]

        # Add reward for each budget
        for k in BUDGETS:
            rec = next((r for r in recs if r["action_budget"] == k), None)
            if rec:
                ctx[f"reward_k{k}"] = rec["reward"]
                ctx[f"best_top1_k{k}"] = rec["best_top1"]
            else:
                ctx[f"reward_k{k}"] = None
                ctx[f"best_top1_k{k}"] = None

        # Oracle: best k by reward
        rewards_by_k = {k: ctx[f"reward_k{k}"] for k in BUDGETS if ctx[f"reward_k{k}"] is not None}
        if rewards_by_k:
            best_k = max(rewards_by_k, key=rewards_by_k.get)
            ctx["oracle_best_k"] = best_k
            ctx["oracle_best_reward"] = rewards_by_k[best_k]
        else:
            ctx["oracle_best_k"] = None
            ctx["oracle_best_reward"] = None

        contexts[cid] = ctx

    return contexts


def create_context_action_samples(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Create action_samples (context features + action + reward) for tree model.

    Returns (contexts_list, samples_list) where:
      - contexts_list: unique context feature vectors (one per context)
      - samples_list: each experiment as (features, action=k, reward)
    """
    contexts_dict = build_contexts(records)

    # Action samples: one per experiment record
    samples = []
    for r in records:
        cid = f"{r['dataset_name']}_seed{r['meta_seed']}"
        ctx = contexts_dict.get(cid)
        if ctx is None:
            continue

        sample = {
            "experiment_id": r["experiment_id"],
            "context_id": cid,
            "dataset_name": r["dataset_name"],
            "meta_seed": r["meta_seed"],
            "action_budget": r["action_budget"],
            "normalized_budget_cost": r["normalized_budget_cost"],
            "reward": r["reward"],
            "best_top1": r["best_top1"],
            # Context features
            **{f: ctx[f] for f in CONTEXT_FEATURES},
        }
        samples.append(sample)

    contexts_list = list(contexts_dict.values())
    return contexts_list, samples


def create_train_val_test_splits(
    contexts_list: list[dict],
    samples_list: list[dict],
    test_ratio: float = 0.2,
    val_ratio: float = 0.2,
    random_seed: int = 42,
) -> dict[str, dict]:
    """Split data into train/val/test by context_id seeds.

    Returns dict with keys: train, val, test
    Each has: contexts_jsonl, action_samples_jsonl
    """
    random.seed(random_seed)
    context_ids = list(set(c["context_id"] for c in contexts_list))
    random.shuffle(context_ids)

    n = len(context_ids)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_context_ids = set(context_ids[:n_test])
    val_context_ids = set(context_ids[n_test:n_test + n_val])
    train_context_ids = set(context_ids[n_test + n_val:])

    print(f"Split: {len(train_context_ids)} train, {len(val_context_ids)} val, {len(test_context_ids)} test contexts")
    print(f"  train: {sorted(train_context_ids)[:5]}...")
    print(f"  val:   {sorted(val_context_ids)[:5]}...")
    print(f"  test:  {sorted(test_context_ids)[:5]}...")

    splits = {
        "train": train_context_ids,
        "val": val_context_ids,
        "test": test_context_ids,
    }

    result = {}
    for split_name, context_ids_set in splits.items():
        split_contexts = [c for c in contexts_list if c["context_id"] in context_ids_set]
        split_samples = [s for s in samples_list if s["context_id"] in context_ids_set]

        result[split_name] = {
            "contexts": split_contexts,
            "samples": split_samples,
            "context_count": len(split_contexts),
            "sample_count": len(split_samples),
        }

    return result


def append_dataset_onehot(features: list[float], dataset_name: str) -> list[float]:
    """Append 4-d one-hot encoding for dataset."""
    onehot = [1.0 if d == dataset_name else 0.0 for d in DATASET_ORDER]
    return features + onehot


def write_jsonl(items: list[dict], path: Path) -> None:
    """Write list of dicts as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def create_split_files(split_name: str, data: dict, output_dir: Path) -> None:
    """Write contexts and action_samples JSONL files for a split."""
    contexts_path = output_dir / f"{split_name}_contexts.jsonl"
    samples_path = output_dir / f"{split_name}_action_samples.jsonl"

    write_jsonl(data["contexts"], contexts_path)
    write_jsonl(data["samples"], samples_path)

    print(f"  {split_name}: {data['context_count']} contexts, {data['sample_count']} samples")


def create_summary(contexts_list: list[dict], splits: dict) -> dict:
    """Create dataset summary statistics."""
    total_contexts = len(contexts_list)
    total_samples = sum(d["sample_count"] for d in splits.values())

    # Per-dataset stats
    dataset_stats = {}
    for ds in DATASET_ORDER:
        ds_contexts = [c for c in contexts_list if c["dataset_name"] == ds]
        ds_samples = [s for s in contexts_list if s["dataset_name"] == ds]  # Note: this is wrong, fix
        ds_samples = [s for c in ds_contexts for s in c["samples"]] if "samples" in c else []

        dataset_stats[ds] = {
            "context_count": len(ds_contexts),
        }

    return {
        "total_contexts": total_contexts,
        "total_samples": total_samples,
        "dataset_stats": dataset_stats,
        "splits": {name: {"contexts": d["context_count"], "samples": d["sample_count"]}
                   for name, d in splits.items()},
        "budgets": BUDGETS,
        "context_features": CONTEXT_FEATURES,
    }


def main() -> None:
    print("=" * 60)
    print("CONVERT ROUND22 500 EXPERIMENTS TO RL TRAINING FORMAT")
    print("=" * 60)

    # Load raw data
    print("\n[1] Loading raw data...")
    records = load_raw_data()
    print(f"    Loaded {len(records)} experiment records")

    # Build contexts and samples
    print("\n[2] Building contexts and action samples...")
    contexts_list, samples_list = create_context_action_samples(records)
    print(f"    {len(contexts_list)} unique contexts")
    print(f"    {len(samples_list)} action samples")

    # Create splits
    print("\n[3] Creating train/val/test splits...")
    splits = create_train_val_test_splits(contexts_list, samples_list, test_ratio=0.2, val_ratio=0.2)

    # Write output
    print("\n[4] Writing output files...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, data in splits.items():
        create_split_files(split_name, data, OUTPUT_DIR)

    # Write contexts combined
    write_jsonl(contexts_list, OUTPUT_DIR / "all_contexts.jsonl")
    write_jsonl(samples_list, OUTPUT_DIR / "all_action_samples.jsonl")

    # Write summary
    summary = {
        "source": "round22_bandit_full_summary.jsonl",
        "total_records": len(records),
        "total_contexts": len(contexts_list),
        "total_samples": len(samples_list),
        "splits": {name: {"contexts": d["context_count"], "samples": d["sample_count"]}
                   for name, d in splits.items()},
        "budgets": BUDGETS,
        "context_features": CONTEXT_FEATURES,
        "dataset_order": DATASET_ORDER,
    }

    with open(OUTPUT_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Write CV folds (for consistency with existing pipeline)
    cv_folds = {
        "random_seed": 42,
        "fold_count": 5,
        "folds": [
            {
                "fold_index": i,
                "validation_context_ids": sorted([c["context_id"] for c in contexts_list if hash(c["context_id"] + str(i)) % 5 == 0]),
                "training_context_ids": sorted([c["context_id"] for c in contexts_list if hash(c["context_id"] + str(i)) % 5 != 0]),
            }
            for i in range(5)
        ],
    }

    with open(SPLITS_DIR / "round22_cv_folds.json", "w", encoding="utf-8") as f:
        json.dump(cv_folds, f, indent=2, ensure_ascii=False)

    print("\n[5] Summary...")
    print(f"    Output directory: {OUTPUT_DIR}")
    print(f"    Total contexts: {len(contexts_list)}")
    print(f"    Total samples: {len(samples_list)}")
    for split_name, data in splits.items():
        print(f"    {split_name}: {data['context_count']} contexts, {data['sample_count']} samples")

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()