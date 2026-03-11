from __future__ import annotations

from .hf import HuggingFaceDatasetDownloader
from .registry import register_dataset_downloader


@register_dataset_downloader
class RottenTomatoesDownloader(HuggingFaceDatasetDownloader):
    """Download the Rotten Tomatoes dataset used in GRADMM."""

    name = "rotten_tomatoes"
    description = "Download the Rotten Tomatoes sentiment dataset used in GRADMM."

    def build_raw_dataset(self):
        from datasets import DatasetDict, load_dataset

        train_dataset, validation_dataset = load_dataset("rotten_tomatoes", split=["train", "validation"])
        return DatasetDict({"train": train_dataset, "validation": validation_dataset}), {
            "source_dataset": "rotten_tomatoes",
            "source_splits": ["train", "validation"],
            "provenance_note": "Raw artifacts keep only the train and validation splits used by GRADMM.",
        }
