from __future__ import annotations

from .hf import HuggingFaceDatasetDownloader
from .registry import register_dataset_downloader


@register_dataset_downloader
class ThreeStylesPromptedDownloader(HuggingFaceDatasetDownloader):
    """Download the DataInf style-transfer dataset from Hugging Face."""

    name = "three_styles_prompted_250_512x512"
    description = "Download the DataInf style-transfer dataset from Hugging Face."

    def build_raw_dataset(self):
        from datasets import load_dataset

        return load_dataset("kewu93/three_styles_prompted_250_512x512"), {
            "source_dataset": "kewu93/three_styles_prompted_250_512x512",
        }
