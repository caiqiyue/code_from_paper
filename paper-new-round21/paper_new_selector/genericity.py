from __future__ import annotations

import math
import statistics


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
    candidate_length: int | None = None,
    l_ref: float | None = None,
    length_modulation_enabled: bool = False,
    length_alpha: float = 0.0,
    length_factor_min: float = 0.2,
    length_factor_max: float = 5.0,
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
        gated = raw_score
    else:
        gate_scale = apply_genericity_gate(
            score=raw_score,
            gate_low=gate_low,
            gate_high=gate_high,
            low_scale=low_scale,
            mid_scale=mid_scale,
        )
        gated = raw_score * gate_scale

    if (
        length_modulation_enabled
        and candidate_length is not None
        and l_ref is not None
        and length_alpha != 0.0
    ):
        candidate_length_safe = max(int(candidate_length), 1)
        ratio = float(l_ref) / float(candidate_length_safe)
        raw_factor = ratio ** float(length_alpha)
        factor = max(float(length_factor_min), min(float(length_factor_max), raw_factor))
        gated = gated * factor

    return gated


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
    candidate_lengths: list[int] | None = None,
    length_modulation_enabled: bool = False,
    length_alpha: float = 0.0,
    length_factor_min: float = 0.2,
    length_factor_max: float = 5.0,
) -> list[float]:
    if candidate_lengths is not None and len(candidate_lengths) != len(candidate_vectors):
        raise ValueError(
            f"candidate_lengths length ({len(candidate_lengths)}) does not match "
            f"candidate_vectors length ({len(candidate_vectors)})"
        )

    l_ref: float | None = None
    if length_modulation_enabled and candidate_lengths and length_alpha != 0.0:
        l_ref = float(statistics.median(candidate_lengths))

    lengths_iter: list[int | None]
    if candidate_lengths is None:
        lengths_iter = [None] * len(candidate_vectors)
    else:
        lengths_iter = list(candidate_lengths)

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
            candidate_length=candidate_length,
            l_ref=l_ref,
            length_modulation_enabled=length_modulation_enabled,
            length_alpha=length_alpha,
            length_factor_min=length_factor_min,
            length_factor_max=length_factor_max,
        )
        for candidate_vector, candidate_length in zip(candidate_vectors, lengths_iter)
    ]
