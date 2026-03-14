from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
import json
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    """Create a directory tree if needed and return the same path."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def to_jsonable(value: Any) -> Any:
    """Convert platform objects into JSON-serializable structures."""

    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def write_json(path: Path, payload: Any) -> None:
    """Write a JSON document with UTF-8 encoding."""

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[Any]) -> None:
    """Write newline-delimited JSON rows."""

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def write_text(path: Path, text: str) -> None:
    """Write plain UTF-8 text to disk."""

    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
