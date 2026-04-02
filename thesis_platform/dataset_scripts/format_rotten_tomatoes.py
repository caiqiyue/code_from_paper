#!/usr/bin/env python3
"""Format rotten_tomatoes raw arrow data into PrE-Text compatible JSON files.

Run from thesis_platform root:
    python dataset_scripts/format_rotten_tomatoes.py

Input:  datasets/rotten_tomatoes/raw/
Output: datasets/rotten_tomatoes/formatted/rotten_tomatoes_train.json
        datasets/rotten_tomatoes/formatted/rotten_tomatoes_eval.json

Requires: pyarrow (pip install pyarrow) or datasets (pip install datasets)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_arrow_auto(path: Path) -> list[str]:
    """Try multiple methods to load arrow/parquet file."""
    import pyarrow as pa
    # Try pyarrow IPC stream (HuggingFace datasets uses this format)
    try:
        import pyarrow.ipc as ipc
        with pa.memory_map(str(path), "r") as mmap:
            reader = ipc.open_stream(mmap)
            table = reader.read_all()
        df = table.to_pandas()
        return df["text"].tolist()
    except Exception:
        pass

    # Try pyarrow IPC file format
    try:
        import pyarrow.ipc as ipc
        with path.open("rb") as f:
            reader = ipc.open_file(f)
            table = reader.read_all()
        df = table.to_pandas()
        return df["text"].tolist()
    except Exception:
        pass

    # Try parquet
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(str(path))
        df = table.to_pandas()
        return df["text"].tolist()
    except Exception:
        pass

    raise ImportError(
        "Could not load data. Install pyarrow: pip install pyarrow"
    )


def main() -> None:
    repo_root = Path(__file__).parent.parent.resolve()
    raw_dir = repo_root / "datasets" / "rotten_tomatoes" / "raw"
    formatted_dir = repo_root / "datasets" / "rotten_tomatoes" / "formatted"

    if not raw_dir.exists():
        print(f"Error: Raw data directory not found: {raw_dir}", file=sys.stderr)
        sys.exit(1)

    # Load train and validation splits
    train_texts = []
    eval_texts = []

    train_arrow = raw_dir / "train" / "data-00000-of-00001.arrow"
    val_arrow = raw_dir / "validation" / "data-00000-of-00001.arrow"

    if train_arrow.exists():
        try:
            train_texts = load_arrow_auto(train_arrow)
            print(f"Train: {len(train_texts)} records")
        except Exception as e:
            print(f"Error loading train data: {e}", file=sys.stderr)
            raise
    else:
        print(f"Warning: Train data not found at {train_arrow}", file=sys.stderr)

    if val_arrow.exists():
        try:
            eval_texts = load_arrow_auto(val_arrow)
            print(f"Validation: {len(eval_texts)} records")
        except Exception as e:
            print(f"Error loading validation data: {e}", file=sys.stderr)
            raise
    else:
        print(f"Warning: Validation data not found at {val_arrow}", file=sys.stderr)

    if not train_texts and not eval_texts:
        print("Error: No data loaded from arrow files.", file=sys.stderr)
        sys.exit(1)

    print(f"\nTotal: {len(train_texts)} train, {len(eval_texts)} eval")

    # Write output files
    formatted_dir.mkdir(parents=True, exist_ok=True)

    with open(formatted_dir / "rotten_tomatoes_train.json", "w", encoding="utf-8") as f:
        json.dump(train_texts, f, ensure_ascii=False)

    with open(formatted_dir / "rotten_tomatoes_eval.json", "w", encoding="utf-8") as f:
        json.dump(eval_texts, f, ensure_ascii=False)

    # Write metadata
    metadata = {
        "name": "rotten_tomatoes",
        "description": "Rotten Tomatoes movie review dataset formatted for PrE-Text experiments",
        "downloaded_at": "2026-04-01",
        "dataset_root": "datasets/rotten_tomatoes",
        "raw_path": "datasets/rotten_tomatoes/raw",
        "formatted_path": "datasets/rotten_tomatoes/formatted",
        "formatter_name": "rotten_tomatoes",
        "required_paths": [
            "datasets/rotten_tomatoes/raw",
            "datasets/rotten_tomatoes/formatted/rotten_tomatoes_train.json",
            "datasets/rotten_tomatoes/formatted/rotten_tomatoes_eval.json",
        ],
        "raw_format": "arrow_stream",
        "formatted_format": "json",
        "split_sizes": {
            "train": len(train_texts),
            "eval": len(eval_texts),
        },
        "total_records": len(train_texts) + len(eval_texts),
        "provenance_note": "Rotten Tomatoes movie reviews from HuggingFace. Split: original train/validation.",
        "task_type": "sentiment_analysis",
    }

    with open(formatted_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nFormatted data written to {formatted_dir}/")
    print(f"  - rotten_tomatoes_train.json ({len(train_texts)} texts)")
    print(f"  - rotten_tomatoes_eval.json ({len(eval_texts)} texts)")
    print(f"  - metadata.json")


if __name__ == "__main__":
    main()
