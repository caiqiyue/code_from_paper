from __future__ import annotations

from .base import BaseDatasetFormatter
from .registry import register_dataset_formatter


@register_dataset_formatter
class IdentityDatasetFormatter(BaseDatasetFormatter):
    """Reuse raw artifacts when they already match the experiment format."""

    name = "identity"

    def formatted_path(self, downloader):
        return downloader.raw_path()

    def perform_format(self, downloader, force: bool, raw_metadata: dict[str, object]):
        raw_path = downloader.raw_path()
        if raw_path is None:
            raise ValueError(f"{downloader.name} cannot use the identity formatter without raw artifacts.")
        return {
            "message": "Raw dataset artifacts already match the experiment format.",
            "metadata": {
                "formatted_format": raw_metadata.get("raw_format", "raw"),
                "provenance_note": "Formatted artifacts reuse the raw download without an extra conversion step.",
            },
        }
