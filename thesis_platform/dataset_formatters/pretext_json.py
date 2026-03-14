from __future__ import annotations

import json

from thesis_platform.core.io_utils import write_json

from .base import BaseDatasetFormatter
from .registry import register_dataset_formatter


def _read_jsonl_rows(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@register_dataset_formatter
class PretextJSONFormatter(BaseDatasetFormatter):
    """Materialize PrE-Text-ready JSON files from staged JSONL artifacts."""

    name = "pretext_json"

    def required_paths(self, downloader):
        target = downloader.formatted_path()
        if target is None:
            return []
        if getattr(downloader, "pretext_dataset_kind", "") == "initialization":
            return [target / "initialization.json"]
        prefix = getattr(downloader, "pretext_output_prefix", downloader.name.removeprefix("pretext_"))
        return [target / f"{prefix}_train.json", target / f"{prefix}_eval.json"]

    def perform_format(self, downloader, force: bool, raw_metadata: dict[str, object]):
        self.prepare_target(downloader)
        target = downloader.formatted_path()
        raw = downloader.raw_path()
        if target is None or raw is None:
            raise ValueError("pretext_json formatter requires both formatted and raw paths.")
        target.mkdir(parents=True, exist_ok=True)

        if getattr(downloader, "pretext_dataset_kind", "") == "initialization":
            rows = _read_jsonl_rows(raw / "initialization.jsonl")
            texts = [str(row["text"]) for row in rows]
            write_json(target / "initialization.json", texts)
            return {
                "message": "Created PrE-Text initialization.json from the staged C4-derived JSONL rows.",
                "metadata": {
                    "formatted_format": "json",
                    "formatted_files": ["formatted/initialization.json"],
                    "split_sizes": {"initialization": len(texts)},
                    "paper_alignment_note": (
                        "PrE-Text's initialization.json is a public seed pool, stored here as a JSON list of strings."
                    ),
                },
            }

        prefix = getattr(downloader, "pretext_output_prefix", downloader.name.removeprefix("pretext_"))
        train_rows = _read_jsonl_rows(raw / "train.jsonl")
        eval_rows = _read_jsonl_rows(raw / "eval.jsonl")
        train_texts = [str(row["text"]) for row in train_rows]
        eval_payload = {"1": [str(row["text"]) for row in eval_rows]}
        write_json(target / f"{prefix}_train.json", train_texts)
        write_json(target / f"{prefix}_eval.json", eval_payload)
        return {
            "message": "Created PrE-Text train/eval JSON files from the staged JSONL rows.",
            "metadata": {
                "formatted_format": "json",
                "formatted_files": [
                    f"formatted/{prefix}_train.json",
                    f"formatted/{prefix}_eval.json",
                ],
                "split_sizes": {"train": len(train_texts), "eval": len(eval_payload["1"])},
                "paper_alignment_note": (
                    "PrE-Text expects one JSON list for private training text and one JSON object keyed by '1' for eval text."
                ),
            },
        }
