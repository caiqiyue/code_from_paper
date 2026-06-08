#!/usr/bin/env python3
"""Shared helpers for E9 federated config/build scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from round23_runtime_utils import load_yaml_with_inherits


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_NEW_ROUND23_ROOT = Path(__file__).resolve().parents[1]
PAPER_NEW_ROUND19_ROOT = REPO_ROOT / "paper-new-round19"
DEFAULT_TOTAL_PROMPT_BUDGET = 32
E9_REPEAT5_SEEDS = [42, 123, 456, 789, 1024]
E9_ALL6_DATASETS = ["jobs", "congressional", "forums", "microblog", "imdb", "openreview"]
E9_IMBALANCE_WEIGHTS_8 = [0.24, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06]


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    repo_candidate = (REPO_ROOT / candidate).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return repo_candidate


def repo_relative_str(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def write_text(path: str | Path, text: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return target


def load_config_with_inherits(config_path: str | Path) -> dict[str, Any]:
    return load_yaml_with_inherits(_resolve_path(config_path))


def resolve_dataset_train_path(config_path: str | Path) -> tuple[str, Path]:
    config = load_config_with_inherits(config_path)
    data_cfg = dict(config.get("data", {}))
    dataset_name = str(data_cfg.get("dataset_name", "")).strip()
    if not dataset_name:
        raise ValueError(f"Config is missing data.dataset_name: {config_path}")
    train_cfg = str(data_cfg.get("train_path", "")).strip()
    if not train_cfg:
        raise ValueError(f"Config is missing data.train_path: {config_path}")
    return dataset_name, _resolve_path(train_cfg)


def default_partition_output_dir(
    *,
    federated_setting: str,
    dataset_name: str,
    seed: int,
) -> Path:
    return (
        PAPER_NEW_ROUND23_ROOT
        / "artifacts"
        / "e9_partitions"
        / federated_setting
        / dataset_name
        / f"seed{seed}"
    )


def default_partition_manifest_relpath(
    *,
    federated_setting: str,
    dataset_name: str,
    seed: int,
) -> str:
    return repo_relative_str(
        default_partition_output_dir(
            federated_setting=federated_setting,
            dataset_name=dataset_name,
            seed=seed,
        )
        / "partition_manifest.json"
    )


def load_partition_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = _resolve_path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Partition manifest must be a JSON object: {manifest_path}")
    clients = payload.get("clients")
    if not isinstance(clients, list) or not clients:
        raise ValueError(f"Partition manifest has no clients: {manifest_path}")
    return payload


def _normalize_weights(weights: Iterable[float]) -> list[float]:
    normalized = [float(item) for item in weights]
    if not normalized:
        raise ValueError("weights must not be empty")
    total = sum(normalized)
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return [item / total for item in normalized]


def allocate_client_prompt_budget(
    *,
    total_prompt_budget: int,
    num_clients: int,
    client_weights: Iterable[float] | None = None,
) -> list[int]:
    if total_prompt_budget <= 0:
        raise ValueError("total_prompt_budget must be positive")
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if client_weights is None:
        weights = [1.0 / num_clients] * num_clients
    else:
        weights = _normalize_weights(client_weights)
        if len(weights) != num_clients:
            raise ValueError("client_weights length must match num_clients")

    raw = [total_prompt_budget * weight for weight in weights]
    budgets = [int(value) for value in raw]
    remainder = total_prompt_budget - sum(budgets)
    ranking = sorted(
        range(num_clients),
        key=lambda idx: (raw[idx] - budgets[idx], -idx),
        reverse=True,
    )
    for idx in ranking[:remainder]:
        budgets[idx] += 1
    return budgets


def build_client_override_payload(
    *,
    base_config_path: str | Path,
    experiment_id: str,
    client_id: str,
    client_train_path: str,
    output_root: str,
    seed: int,
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "inherits": [repo_relative_str(_resolve_path(base_config_path))],
        "meta": {
            "experiment_id": experiment_id,
            "seed": seed,
            "client_id": client_id,
        },
        "paths": {
            "output_root": output_root,
        },
        "data": {
            "train_path": client_train_path,
        },
    }
    if extra_payload:
        payload.update(extra_payload)
    return payload


def write_client_override_config(
    *,
    base_config_path: str | Path,
    destination_path: str | Path,
    experiment_id: str,
    client_id: str,
    client_train_path: str,
    output_root: str,
    seed: int,
    extra_payload: dict[str, Any] | None = None,
) -> Path:
    payload = build_client_override_payload(
        base_config_path=base_config_path,
        experiment_id=experiment_id,
        client_id=client_id,
        client_train_path=client_train_path,
        output_root=output_root,
        seed=seed,
        extra_payload=extra_payload,
    )
    return write_text(destination_path, yaml.safe_dump(payload, sort_keys=False, allow_unicode=False))


def build_server_eval_payload(
    *,
    base_config_path: str | Path,
    experiment_id: str,
    synthetic_texts_path: str,
    output_root: str,
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "inherits": [repo_relative_str(_resolve_path(base_config_path))],
        "meta": {
            "experiment_id": experiment_id,
        },
        "paths": {
            "output_root": output_root,
        },
        "e9_federated_eval": {
            "synthetic_texts_path": synthetic_texts_path,
        },
    }
    if extra_payload:
        payload.update(extra_payload)
    return payload


def write_server_eval_config(
    *,
    base_config_path: str | Path,
    destination_path: str | Path,
    experiment_id: str,
    synthetic_texts_path: str,
    output_root: str,
    extra_payload: dict[str, Any] | None = None,
) -> Path:
    payload = build_server_eval_payload(
        base_config_path=base_config_path,
        experiment_id=experiment_id,
        synthetic_texts_path=synthetic_texts_path,
        output_root=output_root,
        extra_payload=extra_payload,
    )
    return write_text(destination_path, yaml.safe_dump(payload, sort_keys=False, allow_unicode=False))


def _flatten_texts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(_flatten_texts(item))
        return result
    if isinstance(value, dict):
        if "text" in value and isinstance(value["text"], str):
            return _flatten_texts(value["text"])
        result: list[str] = []
        for item in value.values():
            result.extend(_flatten_texts(item))
        return result
    return _flatten_texts(str(value))


def read_synthetic_texts_file(path: str | Path) -> list[str]:
    payload = json.loads(_resolve_path(path).read_text(encoding="utf-8"))
    return _flatten_texts(payload)


def collect_client_synthetic_texts(paths: Iterable[str | Path]) -> list[str]:
    texts: list[str] = []
    for path in paths:
        texts.extend(read_synthetic_texts_file(path))
    return texts


def write_aggregated_synthetic_texts(path: str | Path, texts: Iterable[str]) -> Path:
    return write_json(path, list(texts))


def build_federated_runtime_sidecar(
    *,
    experiment_id: str,
    federated_setting: str,
    method: str,
    num_clients: int,
    split_mode: str,
    imbalance_mode: str,
    client_success_count: int,
    client_failure_count: int,
    aggregated_synthetic_count: int,
    aggregated_synthetic_count_deduped: int,
    partition_manifest: str,
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "experiment_id": experiment_id,
        "federated_setting": federated_setting,
        "method": method,
        "num_clients": num_clients,
        "split_mode": split_mode,
        "imbalance_mode": imbalance_mode,
        "client_success_count": client_success_count,
        "client_failure_count": client_failure_count,
        "aggregated_synthetic_count": aggregated_synthetic_count,
        "aggregated_synthetic_count_deduped": aggregated_synthetic_count_deduped,
        "partition_manifest": partition_manifest,
    }
    if extra_payload:
        payload.update(extra_payload)
    return payload
