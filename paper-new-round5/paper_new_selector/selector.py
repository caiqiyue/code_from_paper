from __future__ import annotations

from .contracts import CandidateRecord, SelectorDecision
from .redundancy import compute_dynamic_redundancy_penalty


def _score_candidate(
    *,
    index: int,
    candidate_vectors: list[list[float]],
    base_scores: list[float],
    selected_vectors: list[list[float]],
    lambda_redundancy: float,
) -> tuple[float, float]:
    redundancy_penalty = compute_dynamic_redundancy_penalty(
        candidate_vector=candidate_vectors[index],
        selected_vectors=selected_vectors,
    )
    accept_score = base_scores[index] - float(lambda_redundancy) * redundancy_penalty
    return redundancy_penalty, accept_score


def greedy_select_candidates(
    *,
    candidate_vectors: list[list[float]],
    private_support: list[float],
    genericity_penalty: list[float],
    lambda_generic: float,
    lambda_redundancy: float,
    seed_top_k: int,
    hard_negative_top_k: int,
    candidate_texts: list[str] | None = None,
) -> SelectorDecision:
    """Greedily select seeds with a dynamic redundancy penalty."""

    count = len(candidate_vectors)
    if count != len(private_support) or count != len(genericity_penalty):
        raise ValueError(
            "candidate_vectors, private_support, and genericity_penalty must align."
        )
    texts = candidate_texts or [f"candidate_{index}" for index in range(count)]

    base_scores = [
        float(private_support[index])
        * (1.0 - float(lambda_generic) * float(genericity_penalty[index]))
        for index in range(count)
    ]
    accept_scores = [0.0 for _ in range(count)]
    redundancy_penalties = [0.0 for _ in range(count)]
    hard_negative_reason: dict[int, str] = {}
    selected_indices: list[int] = []
    selected_vectors: list[list[float]] = []
    remaining = set(range(count))

    while remaining and len(selected_indices) < max(0, seed_top_k):
        best_index = None
        best_score = None
        for index in remaining:
            redundancy_penalties[index], accept_scores[index] = _score_candidate(
                index=index,
                candidate_vectors=candidate_vectors,
                base_scores=base_scores,
                selected_vectors=selected_vectors,
                lambda_redundancy=lambda_redundancy,
            )
            if best_index is None or accept_scores[index] > best_score:
                best_index = index
                best_score = accept_scores[index]
        assert best_index is not None
        selected_indices.append(best_index)
        selected_vectors.append(candidate_vectors[best_index])
        remaining.remove(best_index)

    for index in remaining:
        redundancy_penalties[index], accept_scores[index] = _score_candidate(
            index=index,
            candidate_vectors=candidate_vectors,
            base_scores=base_scores,
            selected_vectors=selected_vectors,
            lambda_redundancy=lambda_redundancy,
        )

    rejected_indices = sorted(
        remaining,
        key=lambda index: (
            accept_scores[index] if selected_vectors else base_scores[index]
        ),
        reverse=True,
    )

    for index in rejected_indices:
        reason = "low_accept_score_rejected"
        if (
            base_scores[index]
            >= min(base_scores[selected] for selected in selected_indices)
            if selected_indices
            else False
        ):
            reason = "near_boundary_rejected"
        elif redundancy_penalties[index] > 0.8:
            reason = "near_boundary_rejected"
        hard_negative_reason[index] = reason

    hard_negative_indices = rejected_indices[: max(0, hard_negative_top_k)]
    records = [
        CandidateRecord(
            index=index,
            text=texts[index],
            vector=list(candidate_vectors[index]),
            private_support=float(private_support[index]),
            genericity_penalty=float(genericity_penalty[index]),
            redundancy_penalty=float(redundancy_penalties[index]),
            accept_score=float(accept_scores[index]),
            meta={"base_score": float(base_scores[index])},
        )
        for index in range(count)
    ]
    return SelectorDecision(
        selected_indices=selected_indices,
        hard_negative_indices=hard_negative_indices,
        accept_scores=accept_scores,
        base_scores=base_scores,
        hard_negative_reason=hard_negative_reason,
        candidate_records=records,
    )
