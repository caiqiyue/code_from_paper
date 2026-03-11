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
        from datasets import load_dataset

        return load_dataset("dair-ai/emotion", "split"), {
            "source_dataset": "dair-ai/emotion",
            "subset": "split",
        }
