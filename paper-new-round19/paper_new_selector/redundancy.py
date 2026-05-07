from __future__ import annotations

import math


def _dot(left: list[float], right: list[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right)))


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    denominator = _norm(left) * _norm(right)
    if denominator == 0:
        return 0.0
    return _dot(left, right) / denominator


def compute_dynamic_redundancy_penalty(
    *,
    candidate_vector: list[float],
    selected_vectors: list[list[float]],
) -> float:
    """Return the current redundancy penalty relative to already accepted seeds."""

    if not selected_vectors:
        return 0.0
    return max(0.0, max(cosine_similarity(candidate_vector, vector) for vector in selected_vectors))
