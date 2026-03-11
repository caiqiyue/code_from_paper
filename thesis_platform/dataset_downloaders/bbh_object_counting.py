from __future__ import annotations

from .base import BaseDatasetDownloader
from .bbh_utils import download_bbh_task_raw
from .registry import register_dataset_downloader


@register_dataset_downloader
class BBHObjectCountingDownloader(BaseDatasetDownloader):
    """Download the BBH object counting task used in FedTextGrad."""

    name = "bbh_object_counting"
    description = "Download the BBH object counting task and recreate train/val/test CSV splits."
    formatter_name = "bbh"
    task_name = "object_counting"

    def perform_download_raw(self, force: bool):
        return {
            "message": "Downloaded raw BBH task JSON.",
            "metadata": download_bbh_task_raw(self.task_name, self.raw_path()),
        }
