from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from thesis_platform.core.schemas import Sample


def _flatten_item(value: Any) -> list[str]:
    """Flatten nested JSON payload fragments into a plain text list."""

    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()]
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            parts.extend(_flatten_item(item))
        return parts
    if isinstance(value, dict):
        parts: list[str] = []
        for item in value.values():
            parts.extend(_flatten_item(item))
        return parts
    return [str(value).strip()]


def _sorted_json_keys(payload: dict[Any, Any]) -> list[Any]:
    """Return stable numeric-first ordering for JSON object keys."""

    return sorted(payload.keys(), key=lambda item: int(item) if str(item).isdigit() else str(item))


def _looks_like_dataset_container(payload: dict[Any, Any]) -> bool:
    """Heuristically distinguish dataset containers from single-sample records."""

    if not payload:
        return True
    keys = [str(key) for key in payload.keys()]
    if all(key.isdigit() for key in keys):
        return True
    if all(isinstance(value, list) for value in payload.values()):
        return True
    split_like_keys = {
        "train",
        "eval",
        "validation",
        "val",
        "test",
        "initialization",
        "seed",
        "public_seed",
    }
    return all(key.lower() in split_like_keys for key in keys)


def _normalize_json_payload(payload: Any) -> list[str]:
    """Normalize supported JSON dataset shapes into a list of raw texts."""

    if isinstance(payload, dict):
        if _looks_like_dataset_container(payload):
            normalized: list[str] = []
            for key in _sorted_json_keys(payload):
                normalized.extend(text for text in _flatten_item(payload[key]) if text)
            return normalized
        text = " ".join(_flatten_item(payload)).strip()
        return [text] if text else []
    if isinstance(payload, list):
        normalized: list[str] = []
        for item in payload:
            text = " ".join(_flatten_item(item)).strip()
            if text:
                normalized.append(text)
        return normalized
    raise ValueError("Unsupported payload type for dataset normalization.")


def load_texts(path: Path) -> list[str]:
    """Load text samples from a JSON file or a directory of JSON files."""

    if path.is_dir():
        texts: list[str] = []
        for child in sorted(path.glob("*.json")):
            texts.extend(load_texts(child))
        return texts
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return _normalize_json_payload(payload)


def build_samples(
    texts: list[str],
    *,
    dataset_name: str,
    source: str,
    task_type: str,
    round_id: int,
    client_id: str,
    prefix: str,
) -> list[Sample]:
    """Wrap raw texts into the platform's unified Sample schema."""

    return [
        Sample(
            sample_id=f"{prefix}_{idx}",
            client_id=client_id,
            round_id=round_id,
            source=source,
            dataset_name=dataset_name,
            task_type=task_type,
            text=text,
        )
        for idx, text in enumerate(texts)
    ]


def load_samples(
    path: Path,
    *,
    dataset_name: str,
    source: str,
    task_type: str,
    round_id: int,
    client_id: str,
    prefix: str,
) -> list[Sample]:
    """Load texts from disk and convert them into Sample objects."""

    return build_samples(
        load_texts(path),
        dataset_name=dataset_name,
        source=source,
        task_type=task_type,
        round_id=round_id,
        client_id=client_id,
        prefix=prefix,
    )
