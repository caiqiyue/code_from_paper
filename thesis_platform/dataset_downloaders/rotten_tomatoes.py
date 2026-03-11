from __future__ import annotations

from .hf import HuggingFaceDatasetDownloader
from .registry import register_dataset_downloader


@register_dataset_downloader
class RottenTomatoesDownloader(HuggingFaceDatasetDownloader):
    """Download the Rotten Tomatoes dataset used in GRADMM."""

    name = "rotten_tomatoes"
    description = "Download the Rotten Tomatoes sentiment dataset used in GRADMM."

    def build_raw_dataset(self):
        from datasets import load_dataset

        return load_dataset("rotten_tomatoes"), {"source_dataset": "rotten_tomatoes"}
