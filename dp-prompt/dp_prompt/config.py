from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "inherits":
            continue
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    raw = os.path.expandvars(path.read_text(encoding="utf-8"))
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config file must contain a mapping: {path}")
    return data


def load_experiment_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    data = _read_yaml(path)
    merged: dict[str, Any] = {}
    config_chain: list[str] = []

    for inherited in data.get("inherits", []):
        inherited_path = Path(inherited)
        if not inherited_path.is_absolute():
            inherited_path = (path.parent / inherited_path).resolve()
        inherited_cfg = load_experiment_config(inherited_path)
        merged = deep_merge_dicts(merged, inherited_cfg)
        config_chain.extend(inherited_cfg.get("_meta", {}).get("config_chain", []))

    merged = deep_merge_dicts(merged, data)
    merged["_meta"] = {
        "config_path": str(path),
        "config_chain": [*config_chain, str(path)],
    }
    return merged
