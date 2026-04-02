from __future__ import annotations

import json
import random
from pathlib import Path

from .base import BaseDatasetFormatter
from .registry import register_dataset_formatter


def _load_congressional_raw(raw_path: Path) -> list[dict]:
    """Load all monthly congressional JSON files and return a flat list of records."""
    records = []
    if raw_path is None or not raw_path.exists():
        return records

    monthly_files = sorted(raw_path.glob("congressional_data_*.json"))
    for monthly_file in monthly_files:
        try:
            data = json.loads(monthly_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                records.extend(data)
        except (json.JSONDecodeError, IOError):
            continue
    return records


@register_dataset_formatter
class CongressionalFormatter(BaseDatasetFormatter):
    """Format Canadian Congressional records into PrE-Text compatible JSON files.

    Raw format: monthly JSON files with objects containing:
        - url, date_str, title, speaker, data (speech text), chamber, country

    Target format:
        - congressional_train.json: JSON array of strings (speech texts)
        - congressional_eval.json: JSON object with key "1" containing eval texts
    """

    name = "congressional"

    def required_paths(self, downloader) -> list[Path]:
        target = self.formatted_path(downloader)
        if target is None:
            return []
        prefix = getattr(downloader, "congressional_output_prefix", "congressional")
        return [
            target / f"{prefix}_train.json",
            target / f"{prefix}_eval.json",
        ]

    def formatted_path(self, downloader) -> Path | None:
        return getattr(downloader, "formatted_path", None) or (downloader.dataset_root() / "formatted")

    def perform_format(self, downloader, force: bool, raw_metadata: dict) -> dict:
        self.prepare_target(downloader)
        target = self.formatted_path(downloader)
        raw = getattr(downloader, "raw_path", lambda: None)()
        if target is None or raw is None:
            raise ValueError("Congressional formatter requires both formatted and raw paths.")

        target.mkdir(parents=True, exist_ok=True)

        # Load all records from monthly JSON files
        records = _load_congressional_raw(raw)
        if not records:
            raise ValueError(f"No congressional records found in {raw}")

        # Extract speech texts from the 'data' field
        texts = [str(record["data"]) for record in records if record.get("data")]

        if not texts:
            raise ValueError("No valid speech texts found in congressional records.")

        # Shuffle and split: 90% train, 10% eval (matching PrE-Text convention)
        random.seed(42)
        random.shuffle(texts)

        split_idx = int(len(texts) * 0.9)
        train_texts = texts[:split_idx]
        eval_texts = texts[split_idx:]

        prefix = getattr(downloader, "congressional_output_prefix", "congressional")

        from thesis_platform.core.io_utils import write_json

        write_json(target / f"{prefix}_train.json", train_texts)
        write_json(target / f"{prefix}_eval.json", {"1": eval_texts})

        return {
            "message": f"Created congressional train/eval JSON splits. Train: {len(train_texts)}, Eval: {len(eval_texts)}",
            "metadata": {
                "formatted_format": "json",
                "formatted_files": [
                    f"formatted/{prefix}_train.json",
                    f"formatted/{prefix}_eval.json",
                ],
                "split_sizes": {"train": len(train_texts), "eval": len(eval_texts)},
                "provenance_note": (
                    "Congressional data from Canadian parliamentary records (ourcommons.ca). "
                    "Split: 90% train, 10% eval."
                ),
            },
        }
