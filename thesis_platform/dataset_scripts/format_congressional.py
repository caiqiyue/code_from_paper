#!/usr/bin/env python3
"""Format manually downloaded congressional raw data into PrE-Text compatible JSON files.

Run from thesis_platform root:
    python dataset_scripts/format_congressional.py

Input:  datasets/congressional/raw/congressional_data_YYYY-MM.json
Output: datasets/congressional/formatted/congressional_train.json
        datasets/congressional/formatted/congressional_eval.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).parent.parent.resolve()
    raw_dir = repo_root / "datasets" / "congressional" / "raw"
    formatted_dir = repo_root / "datasets" / "congressional" / "formatted"

    if not raw_dir.exists():
        print(f"Error: Raw data directory not found: {raw_dir}", file=sys.stderr)
        sys.exit(1)

    # Load all monthly files
    monthly_files = sorted(raw_dir.glob("congressional_data_*.json"))
    if not monthly_files:
        print(f"Error: No congressional_data_*.json files found in {raw_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {len(monthly_files)} monthly JSON files...")
    records = []
    for mf in monthly_files:
        try:
            data = json.loads(mf.read_text(encoding="utf-8"))
            if isinstance(data, list):
                records.extend(data)
                print(f"  {mf.name}: {len(data)} records")
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Warning: Failed to read {mf.name}: {e}", file=sys.stderr)

    if not records:
        print("Error: No valid records found.", file=sys.stderr)
        sys.exit(1)

    print(f"\nTotal records loaded: {len(records)}")

    # Extract speech texts from the 'data' field
    texts = []
    for record in records:
        text = record.get("data", "").strip()
        if text and len(text.split()) >= 20:
            texts.append(text)

    print(f"Valid texts (>= 20 words): {len(texts)}")

    # Shuffle and split: 90% train, 10% eval
    random.seed(42)
    random.shuffle(texts)

    split_idx = int(len(texts) * 0.9)
    train_texts = texts[:split_idx]
    eval_texts = texts[split_idx:]

    print(f"Train: {len(train_texts)}, Eval: {len(eval_texts)}")

    # Write output files
    formatted_dir.mkdir(parents=True, exist_ok=True)

    import json as json_mod
    with open(formatted_dir / "congressional_train.json", "w", encoding="utf-8") as f:
        json_mod.dump(train_texts, f, ensure_ascii=False)

    with open(formatted_dir / "congressional_eval.json", "w", encoding="utf-8") as f:
        json_mod.dump({"1": eval_texts}, f, ensure_ascii=False)

    # Write metadata
    metadata = {
        "name": "congressional",
        "description": "Canadian Congressional records formatted for PrE-Text experiments",
        "downloaded_at": "2026-04-01",
        "dataset_root": "datasets/congressional",
        "raw_path": "datasets/congressional/raw",
        "formatted_path": "datasets/congressional/formatted",
        "formatter_name": "congressional",
        "required_paths": [
            "datasets/congressional/raw",
            "datasets/congressional/formatted/congressional_train.json",
            "datasets/congressional/formatted/congressional_eval.json",
        ],
        "raw_format": "json",
        "formatted_format": "json",
        "split_sizes": {
            "train": len(train_texts),
            "eval": len(eval_texts),
        },
        "total_records": len(records),
        "valid_texts": len(texts),
        "provenance_note": "Canadian parliamentary records (ourcommons.ca). Split: 90% train, 10% eval.",
        "artifact_sample_counts": {
            "raw": {"total": len(records)},
            "formatted": {"train": len(train_texts), "eval": len(eval_texts), "total": len(texts)},
        },
    }

    with open(formatted_dir / "metadata.json", "w", encoding="utf-8") as f:
        json_mod.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nFormatted data written to {formatted_dir}/")
    print(f"  - congressional_train.json ({len(train_texts)} texts)")
    print(f"  - congressional_eval.json ({len(eval_texts)} texts)")
    print(f"  - metadata.json")


if __name__ == "__main__":
    main()
