from __future__ import annotations

from .hf import HuggingFaceDatasetDownloader
from .registry import register_dataset_downloader


@register_dataset_downloader
class GlueMRPCDownloader(HuggingFaceDatasetDownloader):
    """Download the GLUE MRPC benchmark used in DataInf."""

    name = "glue_mrpc"
    description = "Download the GLUE MRPC splits used by DataInf."

    def build_raw_dataset(self):
        from datasets import load_dataset

        return load_dataset("glue", "mrpc"), {"source_dataset": "glue", "subset": "mrpc"}
