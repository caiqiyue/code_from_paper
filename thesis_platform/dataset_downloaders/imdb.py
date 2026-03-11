from __future__ import annotations

from pathlib import Path

from .base import BaseDatasetDownloader
from .registry import register_dataset_downloader


@register_dataset_downloader
class IMDBDownloader(BaseDatasetDownloader):
    """Stage the vendored IMDB subset used by GRADMM."""

    name = "imdb"
    description = "Stage the vendored IMDB len256 subset used by GRADMM."
    formatter_name = "imdb"

    def raw_path(self) -> Path | None:
        """GRADMM uses the vendored formatted subset directly."""

        return None

    def perform_download_raw(self, force: bool):
        return {
            "message": "IMDB uses the vendored GRADMM len256 subset and does not download a separate raw artifact.",
            "metadata": {
                "source_type": "vendored_local_files",
                "paper_alignment": {
                    "paper": "GRADMM",
                    "experiment": "IMDB len256 subset",
                },
                "provenance_note": "GRADMM ships the authoritative IMDB len256 JSONL files locally, so the downloader stages only those formatted artifacts.",
            },
        }
