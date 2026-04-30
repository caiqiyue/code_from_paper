from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_with_inherits(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config at {path} must decode to a mapping.")
    inherits = payload.pop("inherits", []) or []
    merged: dict[str, Any] = {}
    for inherit in inherits:
        merged = _deep_merge(merged, _load_with_inherits((path.parent / str(inherit)).resolve()))
    return _deep_merge(merged, payload)


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    return _load_with_inherits(Path(config_path).resolve())
