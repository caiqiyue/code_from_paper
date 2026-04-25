from __future__ import annotations

from .consistency import cosine_similarity


def compute_duplicate_penalty(candidate_vector: list[float], kept_vectors: list[list[float]]) -> float:
    if not kept_vectors:
        return 0.0
    return max(cosine_similarity(candidate_vector, kept_vector) for kept_vector in kept_vectors)
