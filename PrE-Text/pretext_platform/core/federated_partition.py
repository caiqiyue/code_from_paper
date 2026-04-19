from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from pretext_platform.core.config import ExperimentConfig


def _ensure_repo_root_on_sys_path(repo_root: Path) -> None:
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def load_samples(*args, **kwargs):
    repo_root = Path(kwargs.pop("repo_root"))
    _ensure_repo_root_on_sys_path(repo_root)
    from thesis_platform.data.loaders import load_samples as _load_samples

    return _load_samples(*args, **kwargs)


def partition_samples(*args, **kwargs):
    repo_root = Path(kwargs.pop("repo_root"))
    _ensure_repo_root_on_sys_path(repo_root)
    from thesis_platform.data.partition import partition_samples as _partition_samples

    return _partition_samples(*args, **kwargs)


def build_federated_client_partitions(config: ExperimentConfig) -> dict[str, dict[str, list[str]]]:
    """Create fixed per-client text partitions by reusing thesis_platform partitioning."""

    federation_cfg = config.federation
    data_cfg = config.data
    train_path = config.resolve_path(data_cfg.get("train_path"))
    if train_path is None:
        raise ValueError("data.train_path must be configured for federated_pretext mode.")

    repo_root = config.repo_root().resolve()
    sample_format = str(data_cfg.get("sample_format", "raw_text"))
    task_type = str(data_cfg.get("task_type", "instruction_tuning"))
    loaded_samples = load_samples(
        train_path,
        dataset_name=str(data_cfg.get("dataset_name", "dataset")),
        source="real",
        task_type=task_type,
        round_id=0,
        client_id="raw",
        prefix="real",
        sample_format=sample_format,
        limit=(
            int(data_cfg.get("train_limit"))
            if data_cfg.get("train_limit") not in (None, "")
            else None
        ),
        repo_root=repo_root,
    )
    partitioned = partition_samples(
        loaded_samples,
        num_clients=int(federation_cfg.get("num_clients", data_cfg.get("num_clients", 1))),
        max_samples_per_client=int(
            federation_cfg.get("max_samples_per_client", data_cfg.get("max_samples_per_client", 8))
        ),
        validation_ratio=float(federation_cfg.get("validation_ratio", data_cfg.get("validation_ratio", 0.0))),
        seed=int(config.meta.get("seed", 42)),
        strategy=str(federation_cfg.get("partition_strategy", data_cfg.get("partition_strategy", "shuffle_round_robin"))),
        repo_root=repo_root,
    )

    partitions: dict[str, dict[str, list[str]]] = {}
    for index, bucket in enumerate(partitioned):
        client_id = f"client_{index:03d}"
        partitions[client_id] = {
            "train_texts": [sample.rendered_text() for sample in bucket.get("train", [])],
            "eval_texts": [sample.rendered_text() for sample in bucket.get("validation", [])],
            "all_texts": [sample.rendered_text() for sample in bucket.get("all", [])],
        }
    return partitions
