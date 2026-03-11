from __future__ import annotations

from .hf import HuggingFaceDatasetDownloader
from .registry import register_dataset_downloader


@register_dataset_downloader
class LiveBenchReasoningSpatialDownloader(HuggingFaceDatasetDownloader):
    """Download the LiveBench spatial reasoning subset used in FedTextGrad."""

    name = "livebench_reasoning_spatial"
    description = "Download the LiveBench reasoning spatial subset used in FedTextGrad."
    formatter_name = "livebench"
    livebench_task = "spatial"

    def build_raw_dataset(self):
        from datasets import load_dataset

        dataset = load_dataset("livebench/reasoning", split="test").filter(lambda row: row["task"] == self.livebench_task)
        return dataset, {
            "source_dataset": "livebench/reasoning",
            "raw_split": "test",
            "filtered_task": self.livebench_task,
            "provenance_note": "Raw artifacts already keep only the LiveBench task used in the FedTextGrad experiment.",
        }
