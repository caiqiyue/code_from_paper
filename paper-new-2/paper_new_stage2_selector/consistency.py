from __future__ import annotations

import math


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def compute_consistency_score(generated_vector: list[float], seed_vectors: list[list[float]]) -> float:
    if not seed_vectors:
        return 0.0
    return max(cosine_similarity(generated_vector, seed_vector) for seed_vector in seed_vectors)
