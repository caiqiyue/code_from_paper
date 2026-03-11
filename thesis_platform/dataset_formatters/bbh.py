from __future__ import annotations

import json

from .base import BaseDatasetFormatter
from .registry import register_dataset_formatter


@register_dataset_formatter
class BBHCSVFormatter(BaseDatasetFormatter):
    """Materialize BBH JSON into FedTextGrad-compatible CSV splits."""

    name = "bbh"

    def required_paths(self, downloader):
        target = downloader.formatted_path()
        if target is None:
            return []
        return [target / "train.csv", target / "val.csv", target / "test.csv"]

    def perform_format(self, downloader, force: bool, raw_metadata: dict[str, object]):
        import pandas as pd

        raw_path = downloader.raw_path()
        if raw_path is None:
            raise ValueError(f"{downloader.name} requires raw BBH artifacts.")
        raw_json_path = raw_path / f"{downloader.task_name}.json"
        if not raw_json_path.exists():
            raise FileNotFoundError(f"Missing BBH raw JSON: {raw_json_path}")
        self.prepare_target(downloader)
        target = downloader.formatted_path()
        if target is None:
            raise ValueError("BBH formatter requires a formatted path.")
        target.mkdir(parents=True, exist_ok=True)

        payload = json.loads(raw_json_path.read_text(encoding="utf-8"))
        examples = payload["examples"]
        train_rows = [{"x": item["input"], "y": item["target"]} for item in examples[:50]]
        val_rows = [{"x": item["input"], "y": item["target"]} for item in examples[50:150]]
        test_rows = [{"x": item["input"], "y": item["target"]} for item in examples[150:]]

        pd.DataFrame(train_rows).to_csv(target / "train.csv")
        pd.DataFrame(val_rows).to_csv(target / "val.csv")
        pd.DataFrame(test_rows).to_csv(target / "test.csv")
        return {
            "message": "Created FedTextGrad-compatible train/val/test CSV splits for BBH.",
            "metadata": {
                "formatted_format": "csv",
                "split_sizes": {
                    "train": len(train_rows),
                    "val": len(val_rows),
                    "test": len(test_rows),
                },
                "provenance_note": "Formatted output mirrors FedTextGrad's local BBH split construction.",
            },
        }
