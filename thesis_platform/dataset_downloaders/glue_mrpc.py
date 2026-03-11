from __future__ import annotations

from .glue_utils import build_glue_train_validation_dataset
from .hf import HuggingFaceDatasetDownloader
from .registry import register_dataset_downloader


@register_dataset_downloader
class GlueMRPCDownloader(HuggingFaceDatasetDownloader):
    """Download the GLUE MRPC benchmark used in DataInf."""

    name = "glue_mrpc"
    description = "Download the GLUE MRPC splits used by DataInf."
    formatter_name = "glue_datainf"

    def build_raw_dataset(self):
        return build_glue_train_validation_dataset("mrpc")
