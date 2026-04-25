"""Core platform primitives."""

from __future__ import annotations

from typing import Any

__all__ = ["SingleNodeRunner"]


def __getattr__(name: str) -> Any:
    if name == "SingleNodeRunner":
        from thesis_platform.core.single_node_runner import SingleNodeRunner

        return SingleNodeRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
