from __future__ import annotations

import math


def _dot(left: list[float], right: list[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right)))


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _cosine(left: list[float], right: list[float]) -> float:
    denominator = _norm(left) * _norm(right)
    if denominator == 0:
        return 0.0
    return _dot(left, right) / denominator


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def build_private_importance_weights(
    *,
    private_vectors: list[list[float]],
    private_lengths: list[int],
    private_knn_k: int,
    density_lambda: float,
    novelty_lambda: float,
    length_lambda: float,
    length_floor: int,
    length_ceiling: int,
) -> list[float]:
    """Build a lightweight private importance prior w(x)."""

    if len(private_vectors) != len(private_lengths):
        raise ValueError("private_vectors and private_lengths must have the same length.")
    if not private_vectors:
        return []

    centroid = [0.0] * len(private_vectors[0])
    for vector in private_vectors:
        for index, value in enumerate(vector):
            centroid[index] += float(value)
    centroid = [value / len(private_vectors) for value in centroid]

    weights: list[float] = []
    for index, vector in enumerate(private_vectors):
        neighbor_scores = sorted(
            (
                _cosine(vector, other)
                for other_index, other in enumerate(private_vectors)
                if other_index != index
            ),
            reverse=True,
        )
        top_neighbors = neighbor_scores[: max(1, private_knn_k)]
        density_score = _clip((_mean(top_neighbors) + 1.0) / 2.0)
        centroid_similarity = _clip((_cosine(vector, centroid) + 1.0) / 2.0)
        novelty_score = _clip(1.0 - centroid_similarity)

        length = int(private_lengths[index])
        if length <= 0:
            length_score = 0.0
        elif length < length_floor:
            length_score = float(length) / float(max(1, length_floor))
        elif length > length_ceiling:
            length_score = float(length_ceiling) / float(length)
        else:
            length_score = 1.0

        raw_weight = (
            density_lambda * density_score
            + novelty_lambda * novelty_score
            + length_lambda * _clip(length_score)
        )
        weights.append(max(1e-6, raw_weight))
    return weights
