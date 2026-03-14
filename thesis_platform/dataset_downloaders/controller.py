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

from .base import DatasetDownloadResult
from .common import datasets_root, to_package_relative, utc_timestamp
from .registry import create_dataset_downloader, get_registered_dataset_names, resolve_dataset_names


def list_dataset_downloaders() -> list[dict[str, str]]:
    """Return metadata for every registered dataset downloader."""

    entries: list[dict[str, Any]] = []
    for name in get_registered_dataset_names(include_optional=True):
        downloader = create_dataset_downloader(name)
        entries.append(
            {
                "name": downloader.name,
                "description": downloader.description,
                "optional": downloader.optional,
            }
        )
    return entries


def resolve_dataset_downloaders(names: list[str] | None = None, include_optional: bool = False) -> list[Any]:
    """Instantiate the selected dataset downloaders."""

    return [
        create_dataset_downloader(name)
        for name in resolve_dataset_names(names, include_optional=include_optional)
    ]


def build_dataset_report(
    results: list[DatasetDownloadResult],
    requested_names: list[str],
    include_optional: bool,
) -> dict[str, Any]:
    """Assemble the final dataset download summary."""

    counts = Counter(result.status for result in results)
    return {
        "kind": "dataset_download",
        "generated_at": utc_timestamp(),
        "download_root": to_package_relative(datasets_root()),
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


def download_datasets(
    names: list[str] | None = None,
    force: bool = False,
    include_optional: bool = False,
) -> dict[str, Any]:
    """Run the selected dataset downloads and write one summary report."""

    downloaders = resolve_dataset_downloaders(names, include_optional=include_optional)
    requested_names = [downloader.name for downloader in downloaders]
    results: list[DatasetDownloadResult] = []
    pretext_cache_error: Exception | None = None
    pretext_categories = sorted(
        {
            category
            for downloader in downloaders
            for category in [getattr(downloader, "pretext_c4_category", None)]
            if category is not None
        }
    )

    if pretext_categories:
        from .pretext_utils import ensure_pretext_c4_cache

        try:
            ensure_pretext_c4_cache(required_categories=pretext_categories, force=force)
        except Exception as exc:
            pretext_cache_error = exc

    for downloader in tqdm(downloaders, desc="Datasets", unit="dataset"):
        if pretext_cache_error is not None and getattr(downloader, "pretext_c4_category", None) is not None:
            results.append(
                DatasetDownloadResult(
                    name=downloader.name,
                    status="failed",
                    dataset_root=to_package_relative(downloader.dataset_root()),
                    raw_path=(
                        to_package_relative(downloader.raw_path())
                        if downloader.raw_path() is not None
                        else None
                    ),
                    formatted_path=(
                        to_package_relative(downloader.formatted_path())
                        if downloader.formatted_path() is not None
                        else None
                    ),
                    metadata_path=to_package_relative(downloader.metadata_path()),
                    description=downloader.description,
                    formatter_name=downloader.formatter_name,
                    optional=downloader.optional,
                    error=str(pretext_cache_error),
                    message="Shared PrE-Text C4 cache build failed and dependent datasets were skipped.",
                    sample_counts=None,
                )
            )
            continue
        try:
            results.append(downloader.download(force=force))
        except Exception as exc:
            try:
                sample_counts = downloader.collect_sample_counts()
            except Exception:
                sample_counts = None
            results.append(
                DatasetDownloadResult(
                    name=downloader.name,
                    status="failed",
                    dataset_root=to_package_relative(downloader.dataset_root()),
                    raw_path=(
                        to_package_relative(downloader.raw_path())
                        if downloader.raw_path() is not None
                        else None
                    ),
                    formatted_path=(
                        to_package_relative(downloader.formatted_path())
                        if downloader.formatted_path() is not None
                        else None
                    ),
                    metadata_path=to_package_relative(downloader.metadata_path()),
                    description=downloader.description,
                    formatter_name=downloader.formatter_name,
                    optional=downloader.optional,
                    error=str(exc),
                    message="Dataset download failed and was skipped.",
                    sample_counts=sample_counts,
                )
            )

    report = build_dataset_report(results, requested_names, include_optional)
    write_json(datasets_root() / "download_report.json", report)
    return report
