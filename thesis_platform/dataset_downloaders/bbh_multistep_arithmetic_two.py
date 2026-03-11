from __future__ import annotations

from .base import BaseDatasetDownloader
from .bbh_utils import download_bbh_task_raw
from .registry import register_dataset_downloader


@register_dataset_downloader
class BBHMultistepArithmeticTwoDownloader(BaseDatasetDownloader):
    """Download the BBH multistep arithmetic task used in FedTextGrad."""

    name = "bbh_multistep_arithmetic_two"
    description = "Download the BBH multistep arithmetic two task and recreate train/val/test CSV splits."
    formatter_name = "bbh"
    task_name = "multistep_arithmetic_two"

    def perform_download_raw(self, force: bool):
        return {
            "message": "Downloaded raw BBH task JSON.",
            "metadata": download_bbh_task_raw(self.task_name, self.raw_path()),
        }
