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


def _resolve_reference_rank_weights(
    *,
    count: int,
    reference_rank_weights: list[float] | None,
) -> list[float]:
    if count <= 0:
        return []
    if not reference_rank_weights:
        return [1.0] * count
    weights: list[float] = []
    tail_weight = float(reference_rank_weights[-1])
    for index in range(count):
        if index < len(reference_rank_weights):
            weights.append(float(reference_rank_weights[index]))
        else:
            weights.append(tail_weight)
    return weights


def apply_genericity_gate(
    *,
    score: float,
    gate_low: float,
    gate_high: float,
    low_scale: float,
    mid_scale: float,
) -> float:
    if score <= gate_low:
        return float(low_scale)
    if score <= gate_high:
        return float(mid_scale)
    return 1.0


def compute_genericity_penalty(
    *,
    candidate_vector: list[float],
    reference_vectors: list[list[float]],
    reference_top_k: int,
    reference_rank_weights: list[float] | None = None,
    apply_gate: bool = False,
    gate_low: float = 0.0,
    gate_high: float = 1.0,
    low_scale: float = 1.0,
    mid_scale: float = 1.0,
) -> float:
    """Estimate how close a candidate stays to the public initialization distribution."""

    if not reference_vectors:
        return 0.0
    top_scores = sorted(
        (_cosine(candidate_vector, reference) for reference in reference_vectors),
        reverse=True,
    )[: max(1, reference_top_k)]
    weights = _resolve_reference_rank_weights(
        count=len(top_scores),
        reference_rank_weights=reference_rank_weights,
    )
    denominator = float(sum(weights))
    if denominator <= 0.0:
        return 0.0
    weighted_mean = sum(score * weight for score, weight in zip(top_scores, weights)) / denominator
    raw_score = max(0.0, min(1.0, float(weighted_mean)))
    if not apply_gate:
        return raw_score
    gate_scale = apply_genericity_gate(
        score=raw_score,
        gate_low=gate_low,
        gate_high=gate_high,
        low_scale=low_scale,
        mid_scale=mid_scale,
    )
    return raw_score * gate_scale


def compute_genericity_penalties(
    *,
    candidate_vectors: list[list[float]],
    reference_vectors: list[list[float]],
    reference_top_k: int,
    reference_rank_weights: list[float] | None = None,
    apply_gate: bool = False,
    gate_low: float = 0.0,
    gate_high: float = 1.0,
    low_scale: float = 1.0,
    mid_scale: float = 1.0,
) -> list[float]:
    return [
        compute_genericity_penalty(
            candidate_vector=candidate_vector,
            reference_vectors=reference_vectors,
            reference_top_k=reference_top_k,
            reference_rank_weights=reference_rank_weights,
            apply_gate=apply_gate,
            gate_low=gate_low,
            gate_high=gate_high,
            low_scale=low_scale,
            mid_scale=mid_scale,
        )
        for candidate_vector in candidate_vectors
    ]
