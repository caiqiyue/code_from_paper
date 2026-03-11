from __future__ import annotations

from .hf import HuggingFaceDatasetDownloader
from .registry import register_dataset_downloader


@register_dataset_downloader
class LiveBenchReasoningWebOfLiesDownloader(HuggingFaceDatasetDownloader):
    """Download the LiveBench Web of Lies V2 subset used in FedTextGrad."""

    name = "livebench_reasoning_web_of_lies_v2"
    description = "Download the LiveBench reasoning Web of Lies V2 subset used in FedTextGrad."
    formatter_name = "livebench"
    livebench_task = "web_of_lies_v2"

    def build_raw_dataset(self):
        from datasets import load_dataset

        return load_dataset("livebench/reasoning")["test"], {
            "source_dataset": "livebench/reasoning",
            "raw_split": "test",
        }
