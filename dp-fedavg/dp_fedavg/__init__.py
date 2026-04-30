"""Standalone DP-FedAvg baseline package."""

from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

__all__ = [
    "aggregation",
    "config",
    "data",
    "evaluation",
    "generation",
    "paths",
    "privacy",
    "runners",
    "training",
    "types",
]
