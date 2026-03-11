from __future__ import annotations

from .hf import HuggingFaceDatasetDownloader
from .registry import register_dataset_downloader


@register_dataset_downloader
class IMDBDownloader(HuggingFaceDatasetDownloader):
    """Download the official IMDB dataset for GRADMM."""

    name = "imdb"
    description = "Download the official Hugging Face IMDB dataset referenced by GRADMM."
    formatter_name = "imdb"

    def build_raw_dataset(self):
        from datasets import load_dataset

        return load_dataset("imdb"), {
            "source_dataset": "imdb",
            "provenance_note": "The raw download stores the official Hugging Face IMDB dataset before formatting to GRADMM's vendored len256 subset.",
        }
