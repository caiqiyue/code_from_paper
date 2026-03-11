from __future__ import annotations

from collections.abc import Iterable

from .base import BaseModelDownloader

_MODEL_DOWNLOADERS: dict[str, type[BaseModelDownloader]] = {}


def register_model_downloader(cls: type[BaseModelDownloader]) -> type[BaseModelDownloader]:
    """Register a model downloader class by its public name."""

    if not cls.name:
        raise ValueError("Model downloader classes must define a non-empty name.")
    if cls.name in _MODEL_DOWNLOADERS:
        raise ValueError(f"Duplicate model downloader registration: {cls.name}")
    _MODEL_DOWNLOADERS[cls.name] = cls
    return cls


def get_registered_model_names(include_optional: bool = True) -> list[str]:
    """Return every registered model downloader name."""

    names = []
    for name in sorted(_MODEL_DOWNLOADERS):
        if include_optional or not _MODEL_DOWNLOADERS[name].optional:
            names.append(name)
    return names


def create_model_downloader(name: str, repo_override: str | None = None) -> BaseModelDownloader:
    """Instantiate one registered model downloader."""

    try:
        downloader_cls = _MODEL_DOWNLOADERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model downloader: {name}") from exc
    return downloader_cls(repo_override=repo_override)


def resolve_model_names(names: Iterable[str] | None = None, include_optional: bool = False) -> list[str]:
    """Resolve an optional subset of model names into a stable ordered list."""

    if names is None:
        selected = get_registered_model_names(include_optional=include_optional)
    else:
        selected = list(names)
    deduplicated = list(dict.fromkeys(selected))
    unknown = [name for name in deduplicated if name not in _MODEL_DOWNLOADERS]
    if unknown:
        raise ValueError(f"Unknown model downloaders: {', '.join(unknown)}")
    return deduplicated
