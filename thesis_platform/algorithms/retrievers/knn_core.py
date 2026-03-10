from __future__ import annotations

from thesis_platform.algorithms.math_utils import cosine_similarity


def cosine_top_k(query_vector: list[float], corpus_vectors: list[list[float]], top_k: int) -> list[int]:
    similarities = [(idx, cosine_similarity(vector, query_vector)) for idx, vector in enumerate(corpus_vectors)]
    similarities.sort(key=lambda item: item[1], reverse=True)
    return [idx for idx, _ in similarities[:top_k]]
