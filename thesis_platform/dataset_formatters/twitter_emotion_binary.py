from __future__ import annotations

from .base import BaseDatasetFormatter
from .registry import register_dataset_formatter


@register_dataset_formatter
class TwitterEmotionBinaryFormatter(BaseDatasetFormatter):
    """Keep only the binary sadness/joy subset used by GRADMM."""

    name = "twitter_emotion_binary"

    def perform_format(self, downloader, force: bool, raw_metadata: dict[str, object]):
        from datasets import DatasetDict, load_from_disk

        raw_path = downloader.raw_path()
        if raw_path is None:
            raise ValueError("twitter_emotion_binary requires raw Hugging Face artifacts before formatting.")
        self.prepare_target(downloader)
        raw_dataset = load_from_disk(str(raw_path))
        filtered = DatasetDict(
            {split_name: split.filter(lambda row: row["label"] in [0, 1]) for split_name, split in raw_dataset.items()}
        )
        formatted_path = downloader.formatted_path()
        if formatted_path is None:
            raise ValueError("twitter_emotion_binary formatter requires a formatted path.")
        filtered.save_to_disk(str(formatted_path))
        return {
            "message": "Filtered dair-ai/emotion down to the sadness/joy binary subset.",
            "metadata": {
                "formatted_format": "huggingface_save_to_disk",
                "filter_description": "Keep only rows where label is 0 or 1 (sadness or joy).",
                "split_sizes": {split_name: len(split) for split_name, split in filtered.items()},
                "provenance_note": "Formatted output is the binary subset consumed by GRADMM.",
            },
        }
