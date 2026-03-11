from __future__ import annotations

from pathlib import Path

from thesis_platform.dataset_downloaders.common import copy_file, repo_root, to_package_relative

from .base import BaseDatasetFormatter
from .registry import register_dataset_formatter


@register_dataset_formatter
class VendoredRTPolarityFormatter(BaseDatasetFormatter):
    """Copy the GRADMM vendored RT-Polarity JSONL files into formatted artifacts."""

    name = "rt_polarity"

    def required_paths(self, downloader):
        target = downloader.formatted_path()
        if target is None:
            return []
        return [target / "train.jsonl", target / "validation.jsonl"]

    def perform_format(self, downloader, force: bool, raw_metadata: dict[str, object]):
        self.prepare_target(downloader)
        target = downloader.formatted_path()
        if target is None:
            raise ValueError("rt_polarity formatter requires a formatted path.")
        target.mkdir(parents=True, exist_ok=True)
        source_root = repo_root() / "GRADMM" / "data" / "rtpolarity"
        copied_files: list[str] = []
        for file_name in ("train.jsonl", "validation.jsonl"):
            source = source_root / file_name
            if not source.exists():
                raise FileNotFoundError(f"Missing vendored RT-Polarity formatted file: {source}")
            destination = target / file_name
            copy_file(source, destination)
            copied_files.append(to_package_relative(destination))
        return {
            "message": "Copied vendored GRADMM RT-Polarity JSONL files into the formatted dataset directory.",
            "metadata": {
                "formatted_format": "jsonl",
                "source_type": "vendored_local_files",
                "source_root": str(Path("..") / "GRADMM" / "data" / "rtpolarity"),
                "copied_files": copied_files,
                "provenance_note": "GRADMM's vendored RT-Polarity JSONL files are treated as the authoritative experiment-ready format.",
            },
        }
