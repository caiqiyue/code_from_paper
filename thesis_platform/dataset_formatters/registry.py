from __future__ import annotations

from .base import BaseDatasetFormatter

_DATASET_FORMATTERS: dict[str, type[BaseDatasetFormatter]] = {}


def register_dataset_formatter(cls: type[BaseDatasetFormatter]) -> type[BaseDatasetFormatter]:
    """Register one dataset formatter class."""

    if not cls.name:
        raise ValueError("Dataset formatter classes must define a non-empty name.")
    if cls.name in _DATASET_FORMATTERS:
        raise ValueError(f"Duplicate dataset formatter registration: {cls.name}")
    _DATASET_FORMATTERS[cls.name] = cls
    return cls


def create_dataset_formatter(name: str) -> BaseDatasetFormatter:
    """Instantiate one registered dataset formatter."""

    try:
        formatter_cls = _DATASET_FORMATTERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset formatter: {name}") from exc
    return formatter_cls()


def get_registered_dataset_formatter_names() -> list[str]:
    """Return every registered dataset formatter name."""

    return sorted(_DATASET_FORMATTERS)
