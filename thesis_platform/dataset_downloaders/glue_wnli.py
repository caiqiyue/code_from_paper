from __future__ import annotations

from .hf import HuggingFaceDatasetDownloader
from .registry import register_dataset_downloader


@register_dataset_downloader
class GlueWNLIDownloader(HuggingFaceDatasetDownloader):
    """Download the GLUE WNLI benchmark used in DataInf."""

    name = "glue_wnli"
    description = "Download the GLUE WNLI splits used by DataInf."

    def build_raw_dataset(self):
        from datasets import load_dataset

        return load_dataset("glue", "wnli"), {"source_dataset": "glue", "subset": "wnli"}
