from __future__ import annotations

from collections import defaultdict
import random

from thesis_platform.core.schemas import Sample


def _split_train_validation(bucket: list[Sample], validation_ratio: float) -> tuple[list[Sample], list[Sample]]:
    """Split one client bucket into train and validation slices."""

    if not bucket:
        return [], []
    val_count = min(len(bucket) - 1, max(1, int(len(bucket) * validation_ratio))) if len(bucket) > 1 else 0
    validation = bucket[:val_count]
    train = bucket[val_count:]
    return train, validation


def partition_samples(
    samples: list[Sample],
    *,
    num_clients: int,
    max_samples_per_client: int,
    validation_ratio: float,
    seed: int,
    strategy: str = "shuffle_round_robin",
) -> list[dict[str, list[Sample]]]:
    """Split samples into per-client train/validation buckets for experiments."""

    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")

    strategy = strategy.lower()
    if strategy == "shuffle_round_robin":
        rng = random.Random(seed)
        shuffled = list(samples)
        rng.shuffle(shuffled)
        client_buckets: list[list[Sample]] = [[] for _ in range(num_clients)]
        for index, sample in enumerate(shuffled):
            bucket = client_buckets[index % num_clients]
            if len(bucket) < max_samples_per_client:
                bucket.append(sample)

    elif strategy == "preserve_buckets":
        grouped: dict[str, list[Sample]] = defaultdict(list)
        for sample in samples:
            bucket_id = str(sample.meta.get("bucket_id", sample.client_id or "ungrouped"))
            grouped[bucket_id].append(sample)

        ordered_bucket_ids = sorted(grouped.keys(), key=lambda item: int(item) if item.isdigit() else item)
        if len(ordered_bucket_ids) < num_clients:
            raise ValueError(
                "partition_strategy='preserve_buckets' requires at least as many distinct bucket_id values "
                f"as num_clients. Found {len(ordered_bucket_ids)} buckets for {num_clients} clients. "
                "Use shuffle_round_robin or provide dataset bucket metadata."
            )
        client_buckets = [[] for _ in range(num_clients)]
        for index, bucket_id in enumerate(ordered_bucket_ids):
            client_bucket = client_buckets[index % num_clients]
            for sample in grouped[bucket_id]:
                if len(client_bucket) >= max_samples_per_client:
                    break
                client_bucket.append(sample)
    else:
        raise ValueError(f"Unsupported partition strategy '{strategy}'.")

    partitions: list[dict[str, list[Sample]]] = []
    for idx, bucket in enumerate(client_buckets):
        if not bucket:
            partitions.append({"train": [], "validation": [], "all": []})
            continue
        train, validation = _split_train_validation(bucket, validation_ratio)
        for sample in bucket:
            sample.client_id = f"client_{idx}"  # type: ignore[misc]
        partitions.append({"train": train, "validation": validation, "all": bucket})
    return partitions
