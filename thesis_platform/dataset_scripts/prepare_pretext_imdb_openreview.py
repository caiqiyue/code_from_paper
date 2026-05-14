#!/usr/bin/env python3
"""Import IMDb and OpenReview into thesis_platform/datasets with a pretext-style layout.

Run from thesis_platform root:
    python dataset_scripts/prepare_pretext_imdb_openreview.py
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_formatted_train(path: Path, rows: list[dict]) -> None:
    texts = [row["text"] for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(texts, ensure_ascii=False, indent=2), encoding="utf-8")


def write_formatted_eval(path: Path, rows: list[dict]) -> None:
    texts = [row["text"] for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"1": texts}, ensure_ascii=False, indent=2), encoding="utf-8")


def load_imdb_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            record = json.loads(line)
            text = str(record.get("text", "")).strip()
            label = record.get("label")
            if not text:
                continue
            rows.append(
                {
                    "idx": idx,
                    "text": text,
                    "label": label,
                }
            )
    return rows


def load_openreview_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for idx, record in enumerate(reader):
            text = str(record.get("text", "")).strip()
            label = str(record.get("label2", "")).strip()
            if not text or not label:
                continue
            rows.append(
                {
                    "idx": idx,
                    "text": text,
                    "label": label,
                    "label1": record.get("label1"),
                    "label2": record.get("label2"),
                }
            )
    return rows


def write_metadata(
    dataset_dir: Path,
    name: str,
    description: str,
    source_type: str,
    source_root: str,
    source_files: dict[str, str],
    raw_rows: dict[str, int],
    formatted_files: list[str],
    extra: dict | None = None,
) -> None:
    metadata = {
        "name": name,
        "description": description,
        "optional": True,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": f"datasets/{name}",
        "raw_path": f"datasets/{name}/raw",
        "formatted_path": f"datasets/{name}/formatted",
        "formatter_name": "pretext_json",
        "required_paths": [
            f"datasets/{name}/raw/train.jsonl",
            f"datasets/{name}/raw/eval.jsonl",
            *[f"datasets/{name}/{rel}" for rel in formatted_files],
        ],
        "raw_format": "jsonl",
        "formatted_format": "json",
        "source_type": source_type,
        "source_root": source_root,
        "source_files": source_files,
        "formatted_files": formatted_files,
        "split_sizes": raw_rows,
        "artifact_sample_counts": {
            "raw": {
                "splits": raw_rows,
                "total": sum(raw_rows.values()),
            },
            "formatted": {
                "splits": {
                    Path(formatted_files[0]).stem: raw_rows["train"],
                    Path(formatted_files[1]).stem: raw_rows["eval"],
                },
                "total": sum(raw_rows.values()),
            },
        },
    }
    if extra:
        metadata.update(extra)
    dataset_dir.joinpath("metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_pretext_imdb(root: Path) -> None:
    source_dir = root.parent / "WASP" / "src" / "data" / "imdb" / "std"
    train_rows = load_imdb_jsonl(source_dir / "train.jsonl")
    eval_rows = load_imdb_jsonl(source_dir / "test.jsonl")

    dataset_dir = root / "datasets" / "pretext_imdb"
    raw_dir = dataset_dir / "raw"
    formatted_dir = dataset_dir / "formatted"
    train_rel = "formatted/imdb_train.json"
    eval_rel = "formatted/imdb_eval.json"

    write_jsonl(raw_dir / "train.jsonl", train_rows)
    write_jsonl(raw_dir / "eval.jsonl", eval_rows)
    write_formatted_train(formatted_dir / "imdb_train.json", train_rows)
    write_formatted_eval(formatted_dir / "imdb_eval.json", eval_rows)
    write_metadata(
        dataset_dir=dataset_dir,
        name="pretext_imdb",
        description="IMDb text dataset normalized into the pretext-style dataset layout.",
        source_type="vendored_local_files",
        source_root="..\\WASP\\src\\data\\imdb\\std",
        source_files={"train": "train.jsonl", "eval": "test.jsonl"},
        raw_rows={"train": len(train_rows), "eval": len(eval_rows)},
        formatted_files=[train_rel, eval_rel],
        extra={
            "paper_alignment": {
                "dataset": "IMDb",
            },
            "selected_label_field": "label",
            "provenance_note": "Raw splits are copied from WASP imdb/std; test.jsonl is mapped to eval.jsonl for the pretext-style contract.",
        },
    )


def build_pretext_openreview(root: Path) -> None:
    source_dir = root.parent / "DPGA-TextSyn" / "data" / "openreview"
    train_rows = load_openreview_csv(source_dir / "train.csv")
    eval_rows = load_openreview_csv(source_dir / "test.csv")

    dataset_dir = root / "datasets" / "pretext_openreview"
    raw_dir = dataset_dir / "raw"
    formatted_dir = dataset_dir / "formatted"
    train_rel = "formatted/openreview_train.json"
    eval_rel = "formatted/openreview_eval.json"

    write_jsonl(raw_dir / "train.jsonl", train_rows)
    write_jsonl(raw_dir / "eval.jsonl", eval_rows)
    write_formatted_train(formatted_dir / "openreview_train.json", train_rows)
    write_formatted_eval(formatted_dir / "openreview_eval.json", eval_rows)
    write_metadata(
        dataset_dir=dataset_dir,
        name="pretext_openreview",
        description="OpenReview text dataset normalized into the pretext-style dataset layout using label2 as the supervised target.",
        source_type="vendored_local_files",
        source_root="..\\DPGA-TextSyn\\data\\openreview",
        source_files={"train": "train.csv", "eval": "test.csv", "unused_validation": "val.csv"},
        raw_rows={"train": len(train_rows), "eval": len(eval_rows)},
        formatted_files=[train_rel, eval_rel],
        extra={
            "paper_alignment": {
                "dataset": "OpenReview",
            },
            "selected_label_field": "label2",
            "available_source_labels": ["label1", "label2"],
            "provenance_note": "OpenReview rows are normalized from CSV; label2 is used as the canonical supervised label and test.csv is mapped to eval.jsonl for the pretext-style contract.",
        },
    )


def main() -> None:
    root = repo_root()
    build_pretext_imdb(root)
    build_pretext_openreview(root)
    print("Prepared datasets:")
    print("  - datasets/pretext_imdb")
    print("  - datasets/pretext_openreview")


if __name__ == "__main__":
    main()
