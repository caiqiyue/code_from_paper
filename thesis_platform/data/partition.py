from __future__ import annotations

import random

from thesis_platform.core.schemas import Sample


def partition_samples(
    samples: list[Sample],
    *,
    num_clients: int,
    max_samples_per_client: int,
    validation_ratio: float,
    seed: int,
) -> list[dict[str, list[Sample]]]:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    client_buckets: list[list[Sample]] = [[] for _ in range(num_clients)]
    for index, sample in enumerate(shuffled):
        bucket = client_buckets[index % num_clients]
        if len(bucket) < max_samples_per_client:
            bucket.append(sample)

    partitions: list[dict[str, list[Sample]]] = []
    for idx, bucket in enumerate(client_buckets):
        if not bucket:
            partitions.append({"train": [], "validation": [], "all": []})
            continue
        val_count = min(len(bucket) - 1, max(1, int(len(bucket) * validation_ratio))) if len(bucket) > 1 else 0
        validation = bucket[:val_count]
        train = bucket[val_count:]
        for sample in bucket:
            sample.client_id = f"client_{idx}"  # type: ignore[misc]
        partitions.append({"train": train, "validation": validation, "all": bucket})
    return partitions
