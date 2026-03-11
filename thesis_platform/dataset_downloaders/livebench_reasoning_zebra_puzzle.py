from __future__ import annotations

from .hf import HuggingFaceDatasetDownloader
from .registry import register_dataset_downloader


@register_dataset_downloader
class LiveBenchReasoningZebraPuzzleDownloader(HuggingFaceDatasetDownloader):
    """Download the LiveBench zebra puzzle subset used in FedTextGrad."""

    name = "livebench_reasoning_zebra_puzzle"
    description = "Download the LiveBench reasoning zebra puzzle subset used in FedTextGrad."
    formatter_name = "livebench"
    livebench_task = "zebra_puzzle"

    def build_raw_dataset(self):
        from datasets import load_dataset

        return load_dataset("livebench/reasoning")["test"], {
            "source_dataset": "livebench/reasoning",
            "raw_split": "test",
        }
