from __future__ import annotations

from collections import Counter

from .contracts import BoundaryState


def build_boundary_state(
    *,
    reject_scores: list[float],
    reject_vectors: list[list[float]],
) -> dict:
    """Build a structured boundary-state payload from boundary negatives."""

    if not reject_scores or not reject_vectors:
        state = BoundaryState(
            reject_score_ceiling=0.0,
            reject_score_floor=0.0,
            negative_centroid=[],
            negative_pattern_stats={"count": 0, "mean_reject_score": 0.0},
        )
        return state.to_dict()

    dimensions = len(reject_vectors[0])
    centroid = [0.0] * dimensions
    for vector in reject_vectors:
        for index, value in enumerate(vector):
            centroid[index] += float(value)
    centroid = [value / len(reject_vectors) for value in centroid]

    magnitude_hist = Counter(round(score, 2) for score in reject_scores)
    state = BoundaryState(
        reject_score_ceiling=max(reject_scores),
        reject_score_floor=min(reject_scores),
        negative_centroid=centroid,
        negative_pattern_stats={
            "count": len(reject_scores),
            "mean_reject_score": float(sum(reject_scores)) / float(len(reject_scores)),
            "score_histogram": dict(magnitude_hist),
        },
    )
    return state.to_dict()
