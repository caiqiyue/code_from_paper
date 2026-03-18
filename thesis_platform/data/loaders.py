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


def _expand_items(value: Any) -> list[Any]:
    """Expand nested list containers while preserving dict records."""

    if value is None:
        return []
    if isinstance(value, list):
        expanded: list[Any] = []
        for item in value:
            if isinstance(item, list):
                expanded.extend(_expand_items(item))
            else:
                expanded.append(item)
        return expanded
    return [value]


def _record_from_item(item: Any, *, sample_format: str) -> dict[str, Any] | None:
    """Normalize one raw JSON item into a structured sample record."""

    sample_format = sample_format.lower()
    if sample_format == "raw_text":
        if isinstance(item, str):
            text = item.strip()
            return {"text": text} if text else None
        if isinstance(item, dict):
            instruction = item.get("instruction")
            response = item.get("response")
            label = item.get("label")
            if instruction is not None or response is not None:
                instruction_text = " ".join(_flatten_item(instruction)).strip() if instruction is not None else None
                response_text = " ".join(_flatten_item(response)).strip() if response is not None else None
                text = "\n".join(
                    part
                    for part in [
                        f"Instruction: {instruction_text}" if instruction_text else "",
                        f"Response: {response_text}" if response_text else "",
                    ]
                    if part
                ).strip()
                return {
                    "text": text,
                    "instruction": instruction_text,
                    "response": response_text,
                    "label": label,
                }
            text = " ".join(_flatten_item(item)).strip()
            return {"text": text, "label": label} if text else None
        text = " ".join(_flatten_item(item)).strip()
        return {"text": text} if text else None

    if sample_format == "instruction_response":
        if not isinstance(item, dict):
            raise ValueError("instruction_response format expects JSON objects with instruction/response fields.")
        instruction = " ".join(_flatten_item(item.get("instruction"))).strip()
        response = " ".join(_flatten_item(item.get("response"))).strip()
        if not instruction or not response:
            raise ValueError("instruction_response format requires non-empty instruction and response values.")
        text = f"Instruction: {instruction}\nResponse: {response}"
        return {
            "text": text,
            "instruction": instruction,
            "response": response,
            "label": item.get("label"),
        }

    if sample_format == "classification":
        if not isinstance(item, dict):
            raise ValueError("classification format expects JSON objects with text and label fields.")
        text = " ".join(_flatten_item(item.get("text", item.get("input", "")))).strip()
        if not text:
            raise ValueError("classification format requires a non-empty text field.")
        if "label" not in item:
            raise ValueError("classification format requires a label field.")
        return {"text": text, "label": item["label"]}

    raise ValueError(f"Unsupported sample_format '{sample_format}'.")


def _load_sample_records(path: Path, *, sample_format: str) -> list[dict[str, Any]]:
    """Load JSON samples while preserving bucket metadata for Non-IID partitioning."""

    if path.is_dir():
        records: list[dict[str, Any]] = []
        for child in sorted(path.glob("*.json")):
            child_records = _load_sample_records(child, sample_format=sample_format)
            for record in child_records:
                record.setdefault("meta", {})
                record["meta"].setdefault("split", child.stem)
            records.extend(child_records)
        return records

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    records: list[dict[str, Any]] = []
    if isinstance(payload, dict) and _looks_like_dataset_container(payload):
        for bucket_key in _sorted_json_keys(payload):
            for item in _expand_items(payload[bucket_key]):
                record = _record_from_item(item, sample_format=sample_format)
                if record is None:
                    continue
                record.setdefault("meta", {})
                record["meta"].setdefault("bucket_id", str(bucket_key))
                records.append(record)
        return records

    if isinstance(payload, list):
        for item in _expand_items(payload):
            record = _record_from_item(item, sample_format=sample_format)
            if record is not None:
                records.append(record)
        return records

    record = _record_from_item(payload, sample_format=sample_format)
    return [record] if record is not None else []


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
    sample_format: str = "raw_text",
    limit: int | None = None,
) -> list[Sample]:
    """Load supported JSON samples from disk and convert them into Sample objects."""

    records = _load_sample_records(path, sample_format=sample_format)
    if limit is not None:
        records = records[: max(0, limit)]
    samples: list[Sample] = []
    for idx, record in enumerate(records):
        samples.append(
            Sample(
                sample_id=f"{prefix}_{idx}",
                client_id=client_id,
                round_id=round_id,
                source=source,
                dataset_name=dataset_name,
                task_type=task_type,
                text=str(record.get("text", "")).strip(),
                instruction=record.get("instruction"),
                response=record.get("response"),
                label=record.get("label"),
                meta=dict(record.get("meta", {})),
            )
        )
    return samples
