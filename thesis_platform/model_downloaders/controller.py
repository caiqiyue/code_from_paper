from __future__ import annotations

from collections import Counter
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only in minimal environments.
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        """Fallback progress wrapper used when tqdm is unavailable."""

        return iterable

from thesis_platform.core.io_utils import write_json

from .base import ModelDownloadResult
from .common import models_root, to_package_relative, utc_timestamp
from .registry import create_model_downloader, get_registered_model_names, resolve_model_names


def list_model_downloaders() -> list[dict[str, Any]]:
    """Return metadata for every registered model downloader."""

    entries: list[dict[str, Any]] = []
    for name in get_registered_model_names(include_optional=True):
        downloader = create_model_downloader(name)
        entries.append(
            {
                "name": downloader.name,
                "default_repo_id": downloader.default_repo_id,
                "optional": downloader.optional,
                "community_mirror_only": downloader.community_mirror_only,
                "description": downloader.description,
            }
        )
    return entries


def resolve_model_downloaders(
    names: list[str] | None = None,
    include_optional: bool = False,
    repo_overrides: dict[str, str] | None = None,
) -> list[Any]:
    """Instantiate the selected model downloaders."""

    overrides = repo_overrides or {}
    unknown_override_names = [name for name in overrides if name not in get_registered_model_names(include_optional=True)]
    if unknown_override_names:
        raise ValueError(f"Unknown model downloaders in repo overrides: {', '.join(sorted(unknown_override_names))}")
    return [
        create_model_downloader(name, repo_override=overrides.get(name))
        for name in resolve_model_names(names, include_optional=include_optional)
    ]


def build_model_report(results: list[ModelDownloadResult], requested_names: list[str], include_optional: bool) -> dict[str, Any]:
    """Assemble the final model download summary."""

    counts = Counter(result.status for result in results)
    return {
        "kind": "model_download",
        "generated_at": utc_timestamp(),
        "download_root": to_package_relative(models_root()),
        "requested_names": requested_names,
        "include_optional": include_optional,
        "counts": {
            "downloaded": counts.get("downloaded", 0),
            "skipped": counts.get("skipped", 0),
            "failed": counts.get("failed", 0),
            "total": len(results),
        },
        "results": [result.to_dict() for result in results],
    }


def download_models(
    names: list[str] | None = None,
    force: bool = False,
    include_optional: bool = False,
    repo_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run the selected model downloads and write one summary report."""

    downloaders = resolve_model_downloaders(
        names=names,
        include_optional=include_optional,
        repo_overrides=repo_overrides,
    )
    requested_names = [downloader.name for downloader in downloaders]
    results: list[ModelDownloadResult] = []

    for downloader in tqdm(downloaders, desc="Models", unit="model"):
        try:
            results.append(downloader.download(force=force))
        except Exception as exc:
            results.append(
                ModelDownloadResult(
                    name=downloader.name,
                    status="failed",
                    repo_id=downloader.resolved_repo_id,
                    default_repo_id=downloader.default_repo_id,
                    resolved_repo_id=downloader.resolved_repo_id,
                    target_path=to_package_relative(downloader.target_path()),
                    metadata_path=to_package_relative(downloader.metadata_path()),
                    description=downloader.description,
                    optional=downloader.optional,
                    repo_overridden=downloader.repo_overridden,
                    source_policy=downloader.source_policy,
                    error=str(exc),
                    message="Model download failed and was skipped.",
                )
            )

    report = build_model_report(results, requested_names, include_optional)
    write_json(models_root() / "download_report.json", report)
    return report
