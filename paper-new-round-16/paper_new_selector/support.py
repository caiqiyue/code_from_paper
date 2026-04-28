from __future__ import annotations

import math
import random


def _dot(left: list[float], right: list[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right)))


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _cosine(left: list[float], right: list[float]) -> float:
    denominator = _norm(left) * _norm(right)
    if denominator == 0:
        return 0.0
    return _dot(left, right) / denominator


def compute_private_support(
    *,
    private_vectors: list[list[float]],
    candidate_vectors: list[list[float]],
    private_weights: list[float],
    rank_weights: list[float],
    top_q: int,
) -> list[float]:
    """Compute the weighted Top-Q ranked vote for every candidate."""

    if len(private_vectors) != len(private_weights):
        raise ValueError("private_vectors and private_weights must have the same length.")
    if top_q <= 0:
        raise ValueError("top_q must be positive.")

    support = [0.0 for _ in candidate_vectors]
    if not private_vectors or not candidate_vectors:
        return support

    for private_vector, private_weight in zip(private_vectors, private_weights):
        ranked_indices = sorted(
            range(len(candidate_vectors)),
            key=lambda index: _cosine(private_vector, candidate_vectors[index]),
            reverse=True,
        )[:top_q]
        for rank, candidate_index in enumerate(ranked_indices):
            rank_weight = rank_weights[rank] if rank < len(rank_weights) else rank_weights[-1]
            support[candidate_index] += float(private_weight) * float(rank_weight)
    return support


def apply_gaussian_privacy_noise(
    scores: list[float],
    *,
    enabled: bool,
    sigma: float,
    seed: int,
) -> list[float]:
    """Add deterministic Gaussian noise when the selector is configured to use privacy noise."""

    if not enabled or sigma <= 0:
        return list(scores)
    rng = random.Random(int(seed))
    return [float(score) + rng.gauss(0.0, float(sigma)) for score in scores]
