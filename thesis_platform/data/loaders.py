from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

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


def _normalize_source_domain(url: str) -> str:
    """Normalize one source URL into a stable bucket-friendly domain label."""

    parsed = urlparse(str(url).strip())
    domain = parsed.netloc.lower().strip()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _merge_restored_meta(record_meta: dict[str, Any], restored_meta: dict[str, Any]) -> None:
    """Merge recovered metadata, replacing placeholder bucket IDs when needed."""

    existing_bucket = str(record_meta.get("bucket_id", "")).strip()
    restored_bucket = str(restored_meta.get("bucket_id", "")).strip()
    if restored_bucket and (not existing_bucket or existing_bucket.isdigit()):
        record_meta["bucket_id"] = restored_bucket

    for key, value in restored_meta.items():
        if key == "bucket_id":
            continue
        if value not in (None, ""):
            record_meta.setdefault(key, value)


def _pretext_sidecar_path(path: Path) -> Path | None:
    """Return the raw JSONL sidecar that preserves source URLs for formatted PrE-Text files."""

    if path.name == "initialization.json":
        return None
    stem = path.stem.lower()
    if stem.endswith("_train"):
        return path.parent.parent / "raw" / "train.jsonl"
    if stem.endswith("_eval"):
        return path.parent.parent / "raw" / "eval.jsonl"
    return None


def _attach_pretext_sidecar_metadata(path: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Restore source-domain metadata for formatted PrE-Text datasets when raw sidecars exist."""

    sidecar_path = _pretext_sidecar_path(path)
    if sidecar_path is None or not sidecar_path.exists() or not records:
        return records

    sidecar_rows = _read_jsonl_rows(sidecar_path)
    if len(sidecar_rows) < len(records):
        return records

    for record, row in zip(records, sidecar_rows):
        url = str(row.get("url") or "").strip()
        if not url:
            continue
        domain = _normalize_source_domain(url)
        restored_meta: dict[str, Any] = {"source_url": url}
        if domain:
            restored_meta["source_domain"] = domain
            restored_meta["bucket_id"] = domain
        _merge_restored_meta(record.setdefault("meta", {}), restored_meta)
    return records


def _normalize_bucket_label(value: Any) -> str:
    """Normalize one bucket label while preserving readable identity text."""

    normalized = re.sub(r"\s+", " ", str(value or "").strip()).lower()
    return normalized


@lru_cache(maxsize=8)
def _load_congressional_text_metadata(raw_dir: str) -> dict[str, tuple[dict[str, Any], ...]]:
    """Index congressional raw monthly JSON files by speech text for metadata recovery."""

    raw_path = Path(raw_dir)
    text_to_meta: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for monthly_file in sorted(raw_path.glob("congressional_data_*.json")):
        month_token = monthly_file.stem.rsplit("_", 1)[-1]
        try:
            payload = json.loads(monthly_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict):
                continue
            text = str(row.get("data") or "").strip()
            if not text or len(text.split()) < 20:
                continue
            date_str = str(row.get("date_str") or month_token).strip()
            month_bucket = _normalize_bucket_label(date_str[:7] or month_token)
            speaker = str(row.get("speaker") or "").strip()
            title = str(row.get("title") or "").strip()
            chamber = str(row.get("chamber") or "").strip()
            country = str(row.get("country") or "").strip()
            url = str(row.get("url") or "").strip()
            text_to_meta[text].append(
                {
                    "bucket_id": month_bucket or "congressional",
                    "source_domain": month_bucket or "congressional",
                    "source_url": url,
                    "speaker": speaker,
                    "title": title,
                    "date_str": date_str,
                    "source_month": month_bucket,
                    "chamber": chamber,
                    "country": country,
                }
            )
    return {text: tuple(items) for text, items in text_to_meta.items()}


def _attach_congressional_metadata(path: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recover monthly bucket metadata for congressional formatted datasets."""

    if not records:
        return records
    if path.parent.name != "formatted":
        return records
    if path.parent.parent.name.lower() != "congressional":
        return records

    raw_dir = path.parent.parent / "raw"
    if not raw_dir.exists():
        return records

    indexed_meta = {
        text: list(entries)
        for text, entries in _load_congressional_text_metadata(str(raw_dir.resolve())).items()
    }
    if not indexed_meta:
        return records

    for record in records:
        text = str(record.get("text") or "").strip()
        if not text:
            continue
        candidates = indexed_meta.get(text)
        if not candidates:
            continue
        restored_meta = candidates.pop()
        _merge_restored_meta(record.setdefault("meta", {}), restored_meta)
    return records


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Read newline-delimited JSON rows from one sidecar file."""

    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
    records = _attach_pretext_sidecar_metadata(path, records)
    records = _attach_congressional_metadata(path, records)
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
