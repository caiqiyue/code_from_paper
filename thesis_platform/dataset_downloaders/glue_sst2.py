from __future__ import annotations

from .glue_utils import build_glue_train_validation_dataset
from .hf import HuggingFaceDatasetDownloader
from .registry import register_dataset_downloader


@register_dataset_downloader
class GlueSST2Downloader(HuggingFaceDatasetDownloader):
    """Download the GLUE SST-2 benchmark used by DataInf and GRADMM."""

    name = "glue_sst2"
    description = "Download the GLUE SST-2 splits used by DataInf and GRADMM."
    formatter_name = "glue_datainf"

    def build_raw_dataset(self):
        dataset, metadata = build_glue_train_validation_dataset("sst2")
        metadata["provenance_note"] = (
            "Raw artifacts keep the official train/validation splits used by GRADMM, while the formatted artifacts "
            "store the capped DataInf subset."
        )
        return dataset, metadata
