from __future__ import annotations

import random

from thesis_platform.core.io_utils import write_jsonl

from .base import BaseDatasetFormatter
from .registry import register_dataset_formatter


def _to_dspy_rows(split_dataset) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for example in split_dataset:
        answer = example["answer"].strip().split()
        if len(answer) < 2 or answer[-2] != "####":
            raise ValueError("Unexpected GSM8K answer format; expected the final delimiter token '####'.")
        rows.append(
            {
                "question": example["question"],
                "gold_reasoning": " ".join(answer[:-2]),
                "answer": str(int(answer[-1].replace(",", ""))),
            }
        )
    return rows


@register_dataset_formatter
class GSM8KDSPyFormatter(BaseDatasetFormatter):
    """Create deterministic DSPy-style JSONL splits for GSM8K."""

    name = "gsm8k"

    def required_paths(self, downloader):
        target = downloader.formatted_path()
        if target is None:
            return []
        return [target / "train.jsonl", target / "val.jsonl", target / "test.jsonl"]

    def perform_format(self, downloader, force: bool, raw_metadata: dict[str, object]):
        from datasets import load_from_disk

        raw_path = downloader.raw_path()
        if raw_path is None:
            raise ValueError("gsm8k requires raw Hugging Face artifacts before formatting.")
        self.prepare_target(downloader)
        target = downloader.formatted_path()
        if target is None:
            raise ValueError("gsm8k formatter requires a formatted path.")
        target.mkdir(parents=True, exist_ok=True)

        dataset = load_from_disk(str(raw_path))
        official_train = _to_dspy_rows(dataset["train"])
        official_test = _to_dspy_rows(dataset["test"])

        random.Random(0).shuffle(official_train)
        random.Random(0).shuffle(official_test)

        train_rows = official_train[:50]
        val_rows = official_train[200:300]
        test_rows = official_test[300:400]

        write_jsonl(target / "train.jsonl", train_rows)
        write_jsonl(target / "val.jsonl", val_rows)
        write_jsonl(target / "test.jsonl", test_rows)
        return {
            "message": "Created deterministic DSPy-style GSM8K JSONL splits.",
            "metadata": {
                "formatted_format": "jsonl",
                "dspy_split": {
                    "shuffle_seed": 0,
                    "train": "official_train[:50]",
                    "valid": "official_train[200:300]",
                    "test": "official_test[300:400]",
                },
                "split_sizes": {
                    "train": len(train_rows),
                    "val": len(val_rows),
                    "test": len(test_rows),
                },
                "provenance_note": "Formatted output mirrors FedTextGrad's GSM8K_DSPy slicing logic.",
            },
        }
