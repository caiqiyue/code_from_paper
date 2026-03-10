from __future__ import annotations

import math


def dot(left: list[float], right: list[float]) -> float:
    return sum(l * r for l, r in zip(left, right))


def l2_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def normalize(vector: list[float]) -> list[float]:
    norm = l2_norm(vector)
    if norm <= 0:
        return list(vector)
    return [value / norm for value in vector]


def mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    accum = [0.0] * dim
    for vector in vectors:
        for idx, value in enumerate(vector):
            accum[idx] += value
    return [value / len(vectors) for value in accum]


def subtract(left: list[float], right: list[float]) -> list[float]:
    return [l - r for l, r in zip(left, right)]


def add(left: list[float], right: list[float]) -> list[float]:
    return [l + r for l, r in zip(left, right)]


def scale(vector: list[float], factor: float) -> list[float]:
    return [value * factor for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    left_norm = l2_norm(left)
    right_norm = l2_norm(right)
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot(left, right) / (left_norm * right_norm)
