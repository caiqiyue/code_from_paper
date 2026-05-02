from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from dp_prompt.data.schema import DatasetBundle


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset split file does not exist: {path}")
    if path.suffix == ".jsonl":
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return pd.DataFrame(records)
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            return pd.DataFrame(payload.get("records", []))
        raise TypeError(f"Unsupported JSON payload in {path}")
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset file type: {path}")


def _standardize_split(
    frame: pd.DataFrame,
    *,
    split_name: str,
    text_field: str,
    label_field: str,
    author_field: str,
) -> pd.DataFrame:
    missing = [
        field
        for field in (text_field, label_field, author_field)
        if field not in frame.columns
    ]
    if missing:
        raise KeyError(f"Missing required dataset fields for {split_name}: {missing}")

    standardized = pd.DataFrame(
        {
            "sample_id": [
                str(value)
                for value in frame.get(
                    "sample_id",
                    [f"{split_name}:{index}" for index in range(len(frame))],
                )
            ],
            "text": frame[text_field].astype(str),
            "label": frame[label_field],
            "author_id": frame[author_field].astype(str),
            "split": split_name,
        }
    )
    standardized["source_index"] = frame.index.astype(int)
    return standardized


def load_document_dataset(config: dict[str, Any]) -> DatasetBundle:
    dataset_cfg = config["dataset"]
    text_field = dataset_cfg["text_field"]
    label_field = dataset_cfg["label_field"]
    author_field = dataset_cfg["author_field"]

    frames: list[pd.DataFrame] = []
    for split_name, raw_path in dataset_cfg["splits"].items():
        split_frame = _read_frame(Path(raw_path).expanduser())
        standardized = _standardize_split(
            split_frame,
            split_name=split_name,
            text_field=text_field,
            label_field=label_field,
            author_field=author_field,
        )
        frames.append(standardized)

    combined = pd.concat(frames, ignore_index=True)
    return DatasetBundle(
        dataframe=combined,
        text_field=text_field,
        label_field=label_field,
        author_field=author_field,
    )
