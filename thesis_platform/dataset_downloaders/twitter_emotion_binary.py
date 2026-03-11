from __future__ import annotations

from .hf import HuggingFaceDatasetDownloader
from .registry import register_dataset_downloader


@register_dataset_downloader
class TwitterEmotionBinaryDownloader(HuggingFaceDatasetDownloader):
    """Download the binary Twitter Emotion subset used in GRADMM."""

    name = "twitter_emotion_binary"
    description = "Download the binary sadness/joy subset of dair-ai/emotion used in GRADMM."
    formatter_name = "twitter_emotion_binary"

    def build_raw_dataset(self):
        from datasets import DatasetDict, load_dataset

        train_dataset, validation_dataset = load_dataset("dair-ai/emotion", "split", split=["train", "validation"])
        return DatasetDict({"train": train_dataset, "validation": validation_dataset}), {
            "source_dataset": "dair-ai/emotion",
            "subset": "split",
            "source_splits": ["train", "validation"],
            "provenance_note": "Raw artifacts keep only the train and validation splits used by GRADMM before binary filtering.",
        }
