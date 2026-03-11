from __future__ import annotations

from pathlib import Path
from typing import Any

from thesis_platform.dataset_downloaders.common import remove_path


class BaseDatasetFormatter:
    """Shared workflow for dataset formatters."""

    name = ""

    def formatted_path(self, downloader) -> Path | None:
        """Return the formatted artifact root for one dataset."""

        return downloader.default_formatted_path()

    def required_paths(self, downloader) -> list[Path]:
        """Return the formatted artifact paths needed for readiness."""

        path = self.formatted_path(downloader)
        return [] if path is None else [path]

    def prepare_target(self, downloader) -> None:
        """Clear any stale formatted artifacts before writing new ones."""

        target = self.formatted_path(downloader)
        raw = downloader.raw_path()
        if target is None or target == raw:
            return
        if target.exists():
            remove_path(target)

    def perform_format(self, downloader, force: bool, raw_metadata: dict[str, Any]) -> dict[str, Any]:
        """Transform downloaded raw artifacts into experiment-ready outputs."""

        raise NotImplementedError

    def format(self, downloader, force: bool = False, raw_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run one formatter and return metadata about the formatted artifacts."""

        payload = self.perform_format(downloader, force=force, raw_metadata=raw_metadata or {})
        return {
            "message": payload.get("message", "Formatted dataset artifacts are ready."),
            "metadata": {
                "formatter_name": self.name,
                **payload.get("metadata", {}),
            },
        }
