from __future__ import annotations

from .hf import HuggingFaceDatasetDownloader
from .registry import register_dataset_downloader


@register_dataset_downloader
class GlueQQPDownloader(HuggingFaceDatasetDownloader):
    """Download the GLUE QQP benchmark used in DataInf."""

    name = "glue_qqp"
    description = "Download the GLUE QQP splits used by DataInf."

    def build_raw_dataset(self):
        from datasets import load_dataset

        return load_dataset("glue", "qqp"), {"source_dataset": "glue", "subset": "qqp"}
