from __future__ import annotations

from collections import defaultdict
import random
from typing import Iterable

from thesis_platform.data.loaders import load_samples
from thesis_platform.data.partition import partition_samples

from .paths import resolve_path_from_repo
from .types import ClientPartition


def load_private_samples(
    *,
    dataset_name: str,
    train_path: str,
    train_limit: int | None = None,
) -> list:
    return load_samples(
        resolve_path_from_repo(train_path),
        dataset_name=dataset_name,
        source="private_train",
        task_type="raw_text",
        round_id=0,
        client_id="bootstrap",
        prefix="train",
        limit=train_limit,
    )


def load_eval_samples(
    *,
    dataset_name: str,
    eval_path: str,
    eval_limit: int | None = None,
) -> list:
    return load_samples(
        resolve_path_from_repo(eval_path),
        dataset_name=dataset_name,
        source="private_eval",
        task_type="raw_text",
        round_id=0,
        client_id="eval_client",
        prefix="eval",
        limit=eval_limit,
    )


def detect_partition_mode(samples: Iterable, natural_user_fields: Iterable[str]) -> str:
    for field in natural_user_fields:
        if any(str(sample.meta.get(field, "")).strip() for sample in samples):
            return "natural"
    return "pseudo"


def build_client_partitions(
    samples: list,
    *,
    partition_mode: str,
    num_clients: int,
    max_samples_per_client: int,
    seed: int,
    natural_user_fields: list[str],
) -> list[ClientPartition]:
    if partition_mode == "natural":
        buckets: dict[str, list] = defaultdict(list)
        for sample in samples:
            bucket_value = None
            for field in natural_user_fields:
                value = str(sample.meta.get(field, "")).strip()
                if value:
                    bucket_value = value
                    break
            bucket_key = bucket_value or "ungrouped"
            buckets[bucket_key].append(sample)
        ordered_items = sorted(buckets.items(), key=lambda item: item[0])
        partitions: list[ClientPartition] = []
        for index, (bucket_key, bucket_samples) in enumerate(ordered_items[:num_clients]):
            partitions.append(
                ClientPartition(
                    client_id=f"client_{index}",
                    samples=bucket_samples[:max_samples_per_client],
                    metadata={"bucket_key": bucket_key, "mode": "natural"},
                )
            )
        return partitions

    pseudo = partition_samples(
        samples,
        num_clients=num_clients,
        max_samples_per_client=max_samples_per_client,
        validation_ratio=0.0,
        seed=seed,
        strategy="shuffle_round_robin",
    )
    return [
        ClientPartition(
            client_id=f"client_{index}",
            samples=entry["all"],
            metadata={"mode": "pseudo"},
        )
        for index, entry in enumerate(pseudo)
    ]


def sample_client_ids(
    client_ids: list[str],
    *,
    sample_rate: float,
    seed: int,
) -> list[str]:
    if not client_ids:
        return []
    rng = random.Random(seed)
    selected = [client_id for client_id in client_ids if rng.random() < sample_rate]
    return selected or client_ids[:1]
