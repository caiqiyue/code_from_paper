from __future__ import annotations

import random
from collections import defaultdict

from thesis_platform.core.schemas import ScoredSample


def select_top_k(scored_samples: list[ScoredSample], top_k: int) -> list[ScoredSample]:
    """Select the highest-scoring bad samples independently for each client."""

    grouped: dict[str, list[ScoredSample]] = defaultdict(list)
    for sample in scored_samples:
        grouped[sample.client_id].append(sample)
    selected: list[ScoredSample] = []
    for client_id, client_samples in grouped.items():
        del client_id
        selected.extend(sorted(client_samples, key=lambda item: item.score, reverse=True)[:top_k])
    return selected


def select_random(scored_samples: list[ScoredSample], top_k: int, seed: int = 42) -> list[ScoredSample]:
    """Select random bad samples independently for each client (for ablation experiments)."""

    grouped: dict[str, list[ScoredSample]] = defaultdict(list)
    for sample in scored_samples:
        grouped[sample.client_id].append(sample)
    selected: list[ScoredSample] = []
    rng = random.Random(seed)
    for client_id, client_samples in grouped.items():
        del client_id
        selected.extend(rng.sample(client_samples, min(top_k, len(client_samples))))
    return selected
