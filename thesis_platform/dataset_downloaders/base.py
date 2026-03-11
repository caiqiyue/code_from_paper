from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from thesis_platform.core.io_utils import write_json

from .common import inspect_sample_counts, optional_package_relative, to_package_relative, utc_timestamp


@dataclass
class DatasetDownloadResult:
    """Structured outcome for one dataset download attempt."""

    name: str
    status: str
    dataset_root: str
    raw_path: str | None
    formatted_path: str | None
    metadata_path: str | None
    description: str
    formatter_name: str
    message: str = ""
    error: str | None = None
    sample_counts: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


class BaseDatasetDownloader:
    """Shared workflow for dataset download modules."""

    name = ""
    description = ""
    formatter_name = "identity"

    def dataset_root(self) -> Path:
        """Return the primary root directory for this dataset."""

        from .common import datasets_root

        return datasets_root() / self.name

    def raw_path(self) -> Path | None:
        """Return the raw artifact directory for this dataset."""

        return self.dataset_root() / "raw"

    def default_formatted_path(self) -> Path:
        """Return the default formatted artifact directory."""

        return self.dataset_root() / "formatted"

    def formatter(self):
        """Instantiate the configured dataset formatter."""

        from thesis_platform.dataset_formatters import create_dataset_formatter

        return create_dataset_formatter(self.formatter_name)

    def formatted_path(self) -> Path | None:
        """Return the primary formatted artifact directory for this dataset."""

        return self.formatter().formatted_path(self)

    def metadata_path(self) -> Path:
        """Return the metadata file path written after a successful download."""

        return self.dataset_root() / "metadata.json"

    def raw_required_paths(self) -> list[Path]:
        """Return the raw paths that must exist before formatting."""

        raw = self.raw_path()
        return [] if raw is None else [raw]

    def formatted_required_paths(self) -> list[Path]:
        """Return the formatted paths that must exist for readiness."""

        return self.formatter().required_paths(self)

    def required_paths(self) -> list[Path]:
        """Return every path that must exist for the dataset to be considered ready."""

        unique_paths: list[Path] = []
        seen: set[Path] = set()
        for path in [*self.raw_required_paths(), *self.formatted_required_paths()]:
            if path in seen:
                continue
            unique_paths.append(path)
            seen.add(path)
        return unique_paths

    def is_ready(self) -> bool:
        """Return whether the dataset artifacts and metadata already exist."""

        return all(path.exists() for path in self.required_paths()) and self.metadata_path().exists()

    def perform_download_raw(self, force: bool) -> dict[str, Any]:
        """Download or generate the raw dataset artifacts and return metadata."""

        raise NotImplementedError

    def base_metadata(self) -> dict[str, Any]:
        """Return metadata shared by every dataset downloader."""

        return {
            "name": self.name,
            "description": self.description,
            "downloaded_at": utc_timestamp(),
            "dataset_root": to_package_relative(self.dataset_root()),
            "raw_path": optional_package_relative(self.raw_path()),
            "formatted_path": optional_package_relative(self.formatted_path()),
            "formatter_name": self.formatter_name,
            "required_paths": [to_package_relative(path) for path in self.required_paths()],
        }

    def collect_sample_counts(self) -> dict[str, Any] | None:
        """Inspect raw and formatted artifacts and summarize their sample counts."""

        sample_counts: dict[str, Any] = {}
        raw_counts = inspect_sample_counts(self.raw_path())
        formatted_counts = inspect_sample_counts(self.formatted_path())
        if raw_counts is not None:
            sample_counts["raw"] = raw_counts
        if formatted_counts is not None:
            sample_counts["formatted"] = formatted_counts
        return sample_counts or None

    def load_cached_sample_counts(self) -> dict[str, Any] | None:
        """Reuse sample counts from metadata when available."""

        metadata_path = self.metadata_path()
        if not metadata_path.exists():
            return None
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        sample_counts = payload.get("artifact_sample_counts")
        return sample_counts if isinstance(sample_counts, dict) else None

    def download(self, force: bool = False) -> DatasetDownloadResult:
        """Run the download workflow or skip when the dataset is already ready."""

        if self.is_ready() and not force:
            return DatasetDownloadResult(
                name=self.name,
                status="skipped",
                dataset_root=to_package_relative(self.dataset_root()),
                raw_path=optional_package_relative(self.raw_path()),
                formatted_path=optional_package_relative(self.formatted_path()),
                metadata_path=to_package_relative(self.metadata_path()),
                description=self.description,
                formatter_name=self.formatter_name,
                message="Dataset artifacts already exist.",
                sample_counts=self.load_cached_sample_counts() or self.collect_sample_counts(),
            )

        raw_payload = self.perform_download_raw(force=force)
        formatted_payload = self.formatter().format(self, force=force, raw_metadata=raw_payload.get("metadata", {}))
        sample_counts = self.collect_sample_counts()
        metadata = {
            **self.base_metadata(),
            **raw_payload.get("metadata", {}),
            **formatted_payload.get("metadata", {}),
        }
        if sample_counts is not None:
            metadata["artifact_sample_counts"] = sample_counts
        write_json(self.metadata_path(), metadata)
        return DatasetDownloadResult(
            name=self.name,
            status="downloaded",
            dataset_root=to_package_relative(self.dataset_root()),
            raw_path=optional_package_relative(self.raw_path()),
            formatted_path=optional_package_relative(self.formatted_path()),
            metadata_path=to_package_relative(self.metadata_path()),
            description=self.description,
            formatter_name=self.formatter_name,
            message=str(formatted_payload.get("message") or raw_payload.get("message") or "Dataset download completed."),
            sample_counts=sample_counts,
        )
