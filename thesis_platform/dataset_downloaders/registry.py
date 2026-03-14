from __future__ import annotations

from collections.abc import Iterable

from .base import BaseDatasetDownloader

_DATASET_DOWNLOADERS: dict[str, type[BaseDatasetDownloader]] = {}


def register_dataset_downloader(cls: type[BaseDatasetDownloader]) -> type[BaseDatasetDownloader]:
    """Register a dataset downloader class by its public name."""

    if not cls.name:
        raise ValueError("Dataset downloader classes must define a non-empty name.")
    if cls.name in _DATASET_DOWNLOADERS:
        raise ValueError(f"Duplicate dataset downloader registration: {cls.name}")
    _DATASET_DOWNLOADERS[cls.name] = cls
    return cls


def get_registered_dataset_names(include_optional: bool = True) -> list[str]:
    """Return every registered dataset downloader name."""

    if include_optional:
        return sorted(_DATASET_DOWNLOADERS)
    return sorted(name for name, downloader_cls in _DATASET_DOWNLOADERS.items() if not downloader_cls.optional)


def create_dataset_downloader(name: str) -> BaseDatasetDownloader:
    """Instantiate one registered dataset downloader."""

    try:
        downloader_cls = _DATASET_DOWNLOADERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset downloader: {name}") from exc
    return downloader_cls()


def resolve_dataset_names(names: Iterable[str] | None = None, include_optional: bool = False) -> list[str]:
    """Resolve an optional subset of dataset names into a stable ordered list."""

    selected = list(names) if names is not None else get_registered_dataset_names(include_optional=include_optional)
    deduplicated = list(dict.fromkeys(selected))
    unknown = [name for name in deduplicated if name not in _DATASET_DOWNLOADERS]
    if unknown:
        raise ValueError(f"Unknown dataset downloaders: {', '.join(unknown)}")
    return deduplicated
