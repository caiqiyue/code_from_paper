from __future__ import annotations

from pathlib import Path

from thesis_platform.dataset_downloaders.common import copy_file, repo_root, to_package_relative

from .base import BaseDatasetFormatter
from .registry import register_dataset_formatter


@register_dataset_formatter
class VendoredIMDBFormatter(BaseDatasetFormatter):
    """Copy the GRADMM vendored IMDB JSONL subset into the formatted dataset directory."""

    name = "imdb"

    def required_paths(self, downloader):
        target = downloader.formatted_path()
        if target is None:
            return []
        return [target / "train_len256.jsonl", target / "validation_len256.jsonl"]

    def perform_format(self, downloader, force: bool, raw_metadata: dict[str, object]):
        self.prepare_target(downloader)
        target = downloader.formatted_path()
        if target is None:
            raise ValueError("imdb formatter requires a formatted path.")
        target.mkdir(parents=True, exist_ok=True)
        source_root = repo_root() / "GRADMM" / "data" / "imdb"
        copied_files: list[str] = []
        for file_name in ("train_len256.jsonl", "validation_len256.jsonl"):
            source = source_root / file_name
            if not source.exists():
                raise FileNotFoundError(f"Missing vendored IMDB formatted file: {source}")
            destination = target / file_name
            copy_file(source, destination)
            copied_files.append(to_package_relative(destination))
        return {
            "message": "Copied vendored GRADMM IMDB len256 JSONL files into the formatted dataset directory.",
            "metadata": {
                "formatted_format": "jsonl",
                "source_type": "vendored_local_files",
                "source_root": str(Path("..") / "GRADMM" / "data" / "imdb"),
                "copied_files": copied_files,
                "provenance_note": "GRADMM's vendored IMDB len256 subset is treated as the authoritative experiment-ready format.",
            },
        }
