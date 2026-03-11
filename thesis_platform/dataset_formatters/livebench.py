from __future__ import annotations

from thesis_platform.core.io_utils import write_jsonl

from .base import BaseDatasetFormatter
from .registry import register_dataset_formatter


@register_dataset_formatter
class LiveBenchFormatter(BaseDatasetFormatter):
    """Filter one LiveBench task and split it deterministically into JSONL files."""

    name = "livebench"

    def required_paths(self, downloader):
        target = downloader.formatted_path()
        if target is None:
            return []
        return [target / "train.jsonl", target / "valid.jsonl", target / "test.jsonl"]

    def perform_format(self, downloader, force: bool, raw_metadata: dict[str, object]):
        from datasets import load_from_disk

        raw_path = downloader.raw_path()
        if raw_path is None:
            raise ValueError(f"{downloader.name} requires raw LiveBench artifacts.")
        self.prepare_target(downloader)
        target = downloader.formatted_path()
        if target is None:
            raise ValueError("LiveBench formatter requires a formatted path.")
        target.mkdir(parents=True, exist_ok=True)

        dataset = load_from_disk(str(raw_path))
        filtered = dataset.filter(lambda row: row["task"] == downloader.livebench_task)
        shuffled = filtered.shuffle(seed=0)
        total = len(shuffled)
        train_end = int(total * 0.64)
        valid_end = train_end + int(total * 0.16)
        train_rows = [dict(row) for row in shuffled.select(range(0, train_end))]
        valid_rows = [dict(row) for row in shuffled.select(range(train_end, valid_end))]
        test_rows = [dict(row) for row in shuffled.select(range(valid_end, total))]

        write_jsonl(target / "train.jsonl", train_rows)
        write_jsonl(target / "valid.jsonl", valid_rows)
        write_jsonl(target / "test.jsonl", test_rows)
        return {
            "message": f"Filtered LiveBench task '{downloader.livebench_task}' and wrote deterministic JSONL splits.",
            "metadata": {
                "formatted_format": "jsonl",
                "filtered_task": downloader.livebench_task,
                "split_seed": 0,
                "split_ratio": {"train": 0.64, "valid": 0.16, "test": 0.20},
                "split_sizes": {
                    "train": len(train_rows),
                    "valid": len(valid_rows),
                    "test": len(test_rows),
                },
                "provenance_note": "Formatted output filters first, then applies deterministic LiveBench train/valid/test slicing.",
            },
        }
