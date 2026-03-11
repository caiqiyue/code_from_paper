from __future__ import annotations

from .glue_utils import build_glue_train_validation_dataset
from .hf import HuggingFaceDatasetDownloader
from .registry import register_dataset_downloader


@register_dataset_downloader
class GlueWNLIDownloader(HuggingFaceDatasetDownloader):
    """Download the GLUE WNLI benchmark used in DataInf."""

    name = "glue_wnli"
    description = "Download the GLUE WNLI splits used by DataInf."
    formatter_name = "glue_datainf"

    def build_raw_dataset(self):
        return build_glue_train_validation_dataset("wnli")
