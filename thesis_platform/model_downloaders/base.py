from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from thesis_platform.core.io_utils import write_json

from .common import compute_path_size_bytes, format_bytes, models_root, to_package_relative, utc_timestamp


@dataclass
class ModelDownloadResult:
    """Structured outcome for one model download attempt."""

    name: str
    status: str
    repo_id: str
    default_repo_id: str
    resolved_repo_id: str
    target_path: str
    metadata_path: str | None
    description: str
    optional: bool
    parameter_count_billions: float | None
    model_size_label: str
    large_model: bool
    repo_overridden: bool
    source_policy: str
    message: str = ""
    error: str | None = None
    disk_usage_bytes: int = 0
    disk_usage_human: str = "0 B"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


class BaseModelDownloader:
    """Shared workflow for model download modules."""

    name = ""
    description = ""
    repo_id = ""
    optional = False
    community_mirror_only = False
    parameter_count_billions: float | None = None
    model_size_label = ""
    large_model_threshold_billions = 15.0

    def __init__(self, repo_override: str | None = None) -> None:
        self.default_repo_id = self.repo_id
        self.resolved_repo_id = repo_override or self.default_repo_id

    @property
    def source_policy(self) -> str:
        """Describe how this downloader chooses its upstream model source."""

        if self.community_mirror_only:
            return "community_mirror_only"
        return "default_repo"

    @property
    def repo_overridden(self) -> bool:
        """Return whether a CLI override changed the default repo id."""

        return self.resolved_repo_id != self.default_repo_id

    @property
    def large_model(self) -> bool:
        """Return whether this model exceeds the explicit opt-in threshold."""

        return (
            self.parameter_count_billions is not None
            and self.parameter_count_billions > self.large_model_threshold_billions
        )

    @property
    def resolved_model_size_label(self) -> str:
        """Return one human-readable model size label for reports."""

        if self.model_size_label:
            return self.model_size_label
        if self.parameter_count_billions is None:
            return "unknown"
        if self.parameter_count_billions >= 1:
            return f"{self.parameter_count_billions:g}B"
        millions = self.parameter_count_billions * 1000
        if millions >= 1:
            return f"{millions:g}M"
        return f"{self.parameter_count_billions:g}B"

    def target_path(self) -> Path:
        """Return the primary target directory for this model."""

        return models_root() / self.name

    def metadata_path(self) -> Path:
        """Return the metadata file path written after a successful download."""

        return self.target_path() / "metadata.json"

    def required_paths(self) -> list[Path]:
        """Return every path that must exist for the model to be considered ready."""

        return [self.target_path()]

    def is_ready(self) -> bool:
        """Return whether the model artifacts and metadata already exist."""

        return all(path.exists() for path in self.required_paths()) and self.metadata_path().exists()

    def perform_download(self, force: bool) -> dict[str, Any]:
        """Download or materialize the model artifacts and return metadata."""

        raise NotImplementedError

    def base_metadata(self) -> dict[str, Any]:
        """Return metadata shared by every model downloader."""

        return {
            "name": self.name,
            "description": self.description,
            "repo_id": self.resolved_repo_id,
            "default_repo_id": self.default_repo_id,
            "resolved_repo_id": self.resolved_repo_id,
            "optional": self.optional,
            "parameter_count_billions": self.parameter_count_billions,
            "model_size_label": self.resolved_model_size_label,
            "large_model": self.large_model,
            "repo_overridden": self.repo_overridden,
            "source_policy": self.source_policy,
            "downloaded_at": utc_timestamp(),
            "target_path": to_package_relative(self.target_path()),
            "required_paths": [to_package_relative(path) for path in self.required_paths()],
        }

    def collect_disk_usage(self) -> tuple[int, str]:
        """Return the current target disk usage in bytes and human-readable form."""

        disk_usage_bytes = compute_path_size_bytes(self.target_path())
        return disk_usage_bytes, format_bytes(disk_usage_bytes)

    def load_cached_disk_usage(self) -> tuple[int, str]:
        """Reuse cached disk usage from metadata when available."""

        metadata_path = self.metadata_path()
        if not metadata_path.exists():
            return self.collect_disk_usage()
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return self.collect_disk_usage()
        disk_usage_bytes = payload.get("disk_usage_bytes")
        disk_usage_human = payload.get("disk_usage_human")
        if isinstance(disk_usage_bytes, int) and isinstance(disk_usage_human, str):
            return disk_usage_bytes, disk_usage_human
        return self.collect_disk_usage()

    def download(self, force: bool = False) -> ModelDownloadResult:
        """Run the download workflow or skip when the model is already ready."""

        if self.is_ready() and not force:
            disk_usage_bytes, disk_usage_human = self.load_cached_disk_usage()
            return ModelDownloadResult(
                name=self.name,
                status="skipped",
                repo_id=self.resolved_repo_id,
                default_repo_id=self.default_repo_id,
                resolved_repo_id=self.resolved_repo_id,
                target_path=to_package_relative(self.target_path()),
                metadata_path=to_package_relative(self.metadata_path()),
                description=self.description,
                optional=self.optional,
                parameter_count_billions=self.parameter_count_billions,
                model_size_label=self.resolved_model_size_label,
                large_model=self.large_model,
                repo_overridden=self.repo_overridden,
                source_policy=self.source_policy,
                message="Model artifacts already exist.",
                disk_usage_bytes=disk_usage_bytes,
                disk_usage_human=disk_usage_human,
            )

        payload = self.perform_download(force=force)
        metadata = {**self.base_metadata(), **payload.get("metadata", {})}
        write_json(self.metadata_path(), metadata)
        disk_usage_bytes, disk_usage_human = self.collect_disk_usage()
        metadata["disk_usage_bytes"] = disk_usage_bytes
        metadata["disk_usage_human"] = disk_usage_human
        write_json(self.metadata_path(), metadata)
        return ModelDownloadResult(
            name=self.name,
            status="downloaded",
            repo_id=self.resolved_repo_id,
            default_repo_id=self.default_repo_id,
            resolved_repo_id=self.resolved_repo_id,
            target_path=to_package_relative(self.target_path()),
            metadata_path=to_package_relative(self.metadata_path()),
            description=self.description,
            optional=self.optional,
            parameter_count_billions=self.parameter_count_billions,
            model_size_label=self.resolved_model_size_label,
            large_model=self.large_model,
            repo_overridden=self.repo_overridden,
            source_policy=self.source_policy,
            message=str(payload.get("message", "Model download completed.")),
            disk_usage_bytes=disk_usage_bytes,
            disk_usage_human=disk_usage_human,
        )
