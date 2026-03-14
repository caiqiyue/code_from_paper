from __future__ import annotations

from .base import BaseDatasetDownloader
from .pretext_utils import PRETEXT_INITIALIZATION_POOL_SIZE, stage_pretext_initialization_raw
from .registry import register_dataset_downloader


@register_dataset_downloader
class PretextInitializationC4ENDownloader(BaseDatasetDownloader):
    """Stage the public C4 initialization pool used by PrE-Text."""

    name = "pretext_initialization_c4_en"
    description = "Approximate the PrE-Text 87k public c4-en initialization pool."
    formatter_name = "pretext_json"
    optional = True
    pretext_c4_category = "initialization"
    pretext_dataset_kind = "initialization"
    pretext_output_prefix = "initialization"

    def perform_download_raw(self, force: bool):
        payload = stage_pretext_initialization_raw(self, force=force)
        payload["metadata"]["paper_target_counts"] = {"initialization": PRETEXT_INITIALIZATION_POOL_SIZE}
        return payload
