from __future__ import annotations

from .hf import HuggingFaceDatasetDownloader
from .registry import register_dataset_downloader


@register_dataset_downloader
class GlueQNLIDownloader(HuggingFaceDatasetDownloader):
    """Download the GLUE QNLI benchmark used in DataInf."""

    name = "glue_qnli"
    description = "Download the GLUE QNLI splits used by DataInf."

    def build_raw_dataset(self):
        from datasets import load_dataset

        return load_dataset("glue", "qnli"), {"source_dataset": "glue", "subset": "qnli"}
