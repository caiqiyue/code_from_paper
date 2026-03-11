from __future__ import annotations

from typing import Any

from .base import BaseDatasetDownloader
from .common import remove_path


class HuggingFaceDatasetDownloader(BaseDatasetDownloader):
    """Base class for datasets saved with Hugging Face `save_to_disk`."""

    raw_format = "huggingface_save_to_disk"

    def build_raw_dataset(self) -> tuple[Any, dict[str, Any]]:
        """Return the raw dataset object plus module-specific metadata."""

        raise NotImplementedError

    def perform_download_raw(self, force: bool) -> dict[str, Any]:
        """Materialize the raw dataset into the shared dataset directory."""

        target = self.raw_path()
        if target is None:
            raise ValueError(f"{self.name} does not define a raw_path().")
        if target.exists():
            remove_path(target)
        dataset, metadata = self.build_raw_dataset()
        dataset.save_to_disk(str(target))
        return {
            "message": "Raw dataset saved with Hugging Face save_to_disk().",
            "metadata": {
                "raw_format": self.raw_format,
                "source_type": "huggingface_dataset",
                **metadata,
            },
        }
