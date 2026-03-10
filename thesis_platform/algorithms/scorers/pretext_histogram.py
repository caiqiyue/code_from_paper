from __future__ import annotations

from thesis_platform.algorithms.math_utils import cosine_similarity


def compute_pretext_histogram_scores(
    candidate_vectors: list[list[float]],
    private_vectors: list[list[float]],
) -> tuple[list[float], list[dict[str, float]]]:
    if not candidate_vectors:
        return [], []
    if not private_vectors:
        private_vectors = candidate_vectors

    nearest: list[int] = []
    similarities_by_real: list[list[float]] = []
    for private_vector in private_vectors:
        sims = [cosine_similarity(private_vector, candidate_vector) for candidate_vector in candidate_vectors]
        similarities_by_real.append(sims)
        nearest.append(max(range(len(sims)), key=lambda idx: sims[idx]))

    histogram = [0.0] * len(candidate_vectors)
    for index in nearest:
        histogram[index] += 1.0

    metas: list[dict[str, float]] = []
    bad_scores: list[float] = []
    for idx in range(len(candidate_vectors)):
        assigned = [row[idx] for row, nearest_idx in zip(similarities_by_real, nearest) if nearest_idx == idx]
        mean_similarity = sum(assigned) / len(assigned) if assigned else 0.0
        quality = histogram[idx] + mean_similarity
        bad_score = -quality
        bad_scores.append(bad_score)
        metas.append(
            {
                "raw_histogram": float(histogram[idx]),
                "mean_similarity": mean_similarity,
                "quality_score": quality,
            }
        )
    return bad_scores, metas
