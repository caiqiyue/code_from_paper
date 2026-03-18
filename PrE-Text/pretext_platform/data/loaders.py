from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

from pretext_platform.core.config import ExperimentConfig
from pretext_platform.core.types import DatasetBundle


def _sorted_keys(payload: dict[Any, Any]) -> list[Any]:
    """Return stable numeric-first ordering for JSON object keys."""

    return sorted(payload.keys(), key=lambda item: int(item) if str(item).isdigit() else str(item))


def _flatten_item(value: Any) -> list[str]:
    """Flatten nested JSON payload fragments into a plain text list."""

    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            parts.extend(_flatten_item(item))
        return parts
    if isinstance(value, dict):
        parts: list[str] = []
        for key in _sorted_keys(value):
            parts.extend(_flatten_item(value[key]))
        return parts
    text = str(value).strip()
    return [text] if text else []


def load_json(path: Path) -> Any:
    """Load one JSON file from disk."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sample_client_texts(texts: list[str], *, max_samples_per_client: int, rng: random.Random) -> list[str]:
    """Select a deterministic subset of one client's texts while preserving their order."""

    if len(texts) <= max_samples_per_client:
        return list(texts)
    chosen_indices = sorted(rng.sample(range(len(texts)), k=max_samples_per_client))
    return [texts[index] for index in chosen_indices]


def load_train_texts(
    path: Path,
    *,
    max_samples_per_client: int,
    seed: int,
) -> tuple[list[str], dict[str, Any]]:
    """Load training texts from either a flat list or a client-bucketed JSON object."""

    payload = load_json(path)
    if isinstance(payload, list):
        texts = [text for text in _flatten_item(payload) if text]
        metadata = {
            "shape": "flat_list",
            "train_client_count": 1,
            "raw_train_sample_count": len(texts),
            "sampled_train_sample_count": len(texts),
        }
        return texts, metadata

    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported train dataset shape in {path}.")

    rng = random.Random(seed)
    flattened: list[str] = []
    raw_count = 0
    client_count = 0
    for key in _sorted_keys(payload):
        client_texts = [text for text in _flatten_item(payload[key]) if text]
        if not client_texts:
            continue
        client_count += 1
        raw_count += len(client_texts)
        flattened.extend(
            _sample_client_texts(
                client_texts,
                max_samples_per_client=max_samples_per_client,
                rng=rng,
            )
        )
    metadata = {
        "shape": "client_buckets",
        "train_client_count": client_count,
        "raw_train_sample_count": raw_count,
        "sampled_train_sample_count": len(flattened),
    }
    return flattened, metadata


def load_eval_texts(path: Path) -> list[str]:
    """Load evaluation texts from either a flat list or a dataset container object."""

    payload = load_json(path)
    if isinstance(payload, list):
        return [text for text in _flatten_item(payload) if text]
    if isinstance(payload, dict):
        texts: list[str] = []
        for key in _sorted_keys(payload):
            texts.extend(_flatten_item(payload[key]))
        return [text for text in texts if text]
    raise ValueError(f"Unsupported eval dataset shape in {path}.")


def load_initialization_texts(path: Path, *, min_words: int) -> list[str]:
    """Load and filter the initialization pool used by Private Evolution."""

    payload = load_json(path)
    texts = [text for text in _flatten_item(payload) if text]
    return [text for text in texts if len(text.split()) > min_words]


def _apply_limit(texts: list[str], limit: Any, *, field_name: str) -> list[str]:
    """Optionally cap one text list to a configured maximum size."""

    if limit in (None, ""):
        return texts
    resolved_limit = int(limit)
    if resolved_limit <= 0:
        raise ValueError(f"data.{field_name} must be a positive integer when provided.")
    return texts[:resolved_limit]


def load_dataset_bundle(config: ExperimentConfig) -> DatasetBundle:
    """Load all dataset resources required by the configured experiment."""

    data_cfg = config.data
    dataset_name = str(data_cfg.get("dataset_name", "dataset"))
    dataset_root = config.dataset_root()

    train_path = config.resolve_path(data_cfg.get("train_path")) or (dataset_root / f"{dataset_name}_train.json").resolve()
    eval_path = config.resolve_path(data_cfg.get("eval_path")) or (dataset_root / f"{dataset_name}_eval.json").resolve()
    initialization_path = config.resolve_path(data_cfg.get("initialization_path")) or (dataset_root / "initial_set.json").resolve()

    max_samples_per_client = int(data_cfg.get("max_samples_per_client", 8))
    seed = int(config.meta.get("seed", 42))
    train_texts, train_meta = load_train_texts(
        train_path,
        max_samples_per_client=max_samples_per_client,
        seed=seed,
    )
    train_texts = _apply_limit(train_texts, data_cfg.get("train_limit"), field_name="train_limit")
    train_meta["sampled_train_sample_count"] = len(train_texts)

    eval_texts = _apply_limit(load_eval_texts(eval_path), data_cfg.get("eval_limit"), field_name="eval_limit")
    initialization_texts = _apply_limit(
        load_initialization_texts(
            initialization_path,
            min_words=int(data_cfg.get("initialization_min_words", 20)),
        ),
        data_cfg.get("initialization_limit"),
        field_name="initialization_limit",
    )
    return DatasetBundle(
        dataset_name=dataset_name,
        train_texts=train_texts,
        eval_texts=eval_texts,
        initialization_texts=initialization_texts,
        train_client_count=int(train_meta["train_client_count"]),
        raw_train_sample_count=int(train_meta["raw_train_sample_count"]),
        sampled_train_sample_count=int(train_meta["sampled_train_sample_count"]),
        max_samples_per_client=max_samples_per_client,
        metadata={
            "train_path": str(train_path),
            "eval_path": str(eval_path),
            "initialization_path": str(initialization_path),
            "train_shape": train_meta["shape"],
            "train_limit": data_cfg.get("train_limit"),
            "eval_limit": data_cfg.get("eval_limit"),
            "initialization_limit": data_cfg.get("initialization_limit"),
        },
    )
