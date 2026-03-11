from __future__ import annotations

from .base import BaseDatasetDownloader
from .registry import register_dataset_downloader


@register_dataset_downloader
class RTPolarityDownloader(BaseDatasetDownloader):
    """Copy the vendored RT-Polarity JSONL files from the GRADMM repo."""

    name = "rt_polarity"
    description = "Copy the vendored RT-Polarity JSONL files bundled in GRADMM."
    formatter_name = "rt_polarity"

    def raw_path(self):
        return None

    def perform_download_raw(self, force: bool):
        return {
            "metadata": {
                "raw_format": None,
                "provenance_note": "RT-Polarity does not define a separate raw artifact in this repository; formatting copies the vendored GRADMM JSONL files directly.",
            }
        }
