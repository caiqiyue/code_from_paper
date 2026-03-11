from __future__ import annotations

from .hf import HuggingFaceDatasetDownloader
from .registry import register_dataset_downloader


@register_dataset_downloader
class GSM8KDownloader(HuggingFaceDatasetDownloader):
    """Download the GSM8K dataset and document the DSPy split used by FedTextGrad."""

    name = "gsm8k"
    description = "Download GSM8K main and record the DSPy split rules used by FedTextGrad."
    formatter_name = "gsm8k"

    def build_raw_dataset(self):
        from datasets import load_dataset

        dataset = load_dataset("gsm8k", "main")
        return dataset, {
            "source_dataset": "gsm8k",
            "subset": "main",
        }
