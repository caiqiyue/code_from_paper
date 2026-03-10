from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable


Factory = Callable[..., Any]
_REGISTRY: dict[str, dict[str, Factory]] = defaultdict(dict)


def register(kind: str, name: str, factory: Factory) -> None:
    """Register an adapter factory under a kind/name pair."""

    _REGISTRY[kind][name] = factory


def create(kind: str, name: str, *args: Any, **kwargs: Any) -> Any:
    """Instantiate a registered adapter."""

    if name not in _REGISTRY.get(kind, {}):
        available = ", ".join(sorted(_REGISTRY.get(kind, {}).keys()))
        raise KeyError(f"Unknown {kind} adapter '{name}'. Available: {available}")
    return _REGISTRY[kind][name](*args, **kwargs)


def registered_names(kind: str) -> list[str]:
    """Return all registered adapter names for one adapter kind."""

    return sorted(_REGISTRY.get(kind, {}).keys())
