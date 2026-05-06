from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only in constrained envs
    yaml = None


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
    if yaml is not None:
        data = yaml.safe_load(raw) or {}
    else:
        data = _load_yaml_without_dependency(raw)
    if not isinstance(data, dict):
        raise TypeError(f"Config file must contain a mapping: {path}")
    return data


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if value == "":
        return ""
    if value.startswith(('"', "'")) and value.endswith(('"', "'")):
        return value[1:-1]
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None"}:
        return None
    if value.startswith("[") or value.startswith("{"):
        return json.loads(value.replace("'", '"'))
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value


def _load_yaml_without_dependency(text: str) -> dict[str, Any]:
    lines = []
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        lines.append((indent, raw_line.strip()))

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        mapping: dict[str, Any] = {}
        sequence: list[Any] | None = None
        while index < len(lines):
            current_indent, content = lines[index]
            if current_indent < indent:
                break
            if content.startswith("- "):
                if sequence is None:
                    sequence = []
                item = content[2:].strip()
                if item:
                    sequence.append(_parse_scalar(item))
                    index += 1
                else:
                    child, index = parse_block(index + 1, current_indent + 2)
                    sequence.append(child)
                continue

            key, _, raw_value = content.partition(":")
            if not _:
                raise ValueError(f"Invalid YAML line: {content}")
            if raw_value.strip():
                mapping[key.strip()] = _parse_scalar(raw_value)
                index += 1
            else:
                child, index = parse_block(index + 1, current_indent + 2)
                mapping[key.strip()] = child
        return (sequence if sequence is not None else mapping), index

    parsed, _ = parse_block(0, 0)
    if not isinstance(parsed, dict):
        raise TypeError("Top-level YAML object must be a mapping.")
    return parsed


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
