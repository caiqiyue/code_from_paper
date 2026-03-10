from __future__ import annotations

from collections import defaultdict

from thesis_platform.core.schemas import ScoredSample


def select_top_k(scored_samples: list[ScoredSample], top_k: int) -> list[ScoredSample]:
    grouped: dict[str, list[ScoredSample]] = defaultdict(list)
    for sample in scored_samples:
        grouped[sample.client_id].append(sample)
    selected: list[ScoredSample] = []
    for client_id, client_samples in grouped.items():
        del client_id
        selected.extend(sorted(client_samples, key=lambda item: item.score, reverse=True)[:top_k])
    return selected
