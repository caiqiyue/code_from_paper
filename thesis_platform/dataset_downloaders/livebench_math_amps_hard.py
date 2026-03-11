from __future__ import annotations

from .hf import HuggingFaceDatasetDownloader
from .registry import register_dataset_downloader


@register_dataset_downloader
class LiveBenchMathAMPSHardDownloader(HuggingFaceDatasetDownloader):
    """Download the LiveBench AMPS-Hard subset used in FedTextGrad."""

    name = "livebench_math_amps_hard"
    description = "Download the LiveBench math AMPS-Hard subset used in FedTextGrad."
    formatter_name = "livebench"
    livebench_task = "AMPS_Hard"

    def build_raw_dataset(self):
        from datasets import load_dataset

        return load_dataset("livebench/math")["test"], {
            "source_dataset": "livebench/math",
            "raw_split": "test",
        }
