from __future__ import annotations

from .hf import HuggingFaceDatasetDownloader
from .registry import register_dataset_downloader


@register_dataset_downloader
class GlueSST2Downloader(HuggingFaceDatasetDownloader):
    """Download the GLUE SST-2 benchmark used by DataInf and GRADMM."""

    name = "glue_sst2"
    description = "Download the GLUE SST-2 splits used by DataInf and GRADMM."

    def build_raw_dataset(self):
        from datasets import load_dataset

        return load_dataset("glue", "sst2"), {"source_dataset": "glue", "subset": "sst2"}
