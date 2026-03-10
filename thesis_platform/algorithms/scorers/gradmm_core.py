from __future__ import annotations

from collections import Counter
import re

from thesis_platform.algorithms.math_utils import cosine_similarity, l2_norm, mean_vector, subtract

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _rarity_score(text: str, corpus_freq: Counter[str]) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    return float(sum(1.0 / (1.0 + corpus_freq[token]) for token in tokens) / len(tokens))


def compute_gradmm_scores(
    sample_vectors: list[list[float]],
    reference_vectors: list[list[float]],
    *,
    texts: list[str],
    corpus_texts: list[str],
    alpha: float,
) -> tuple[list[float], list[dict[str, float]]]:
    if not sample_vectors:
        return [], []
    if not reference_vectors:
        reference_vectors = sample_vectors

    ref = mean_vector(reference_vectors)
    corpus_freq: Counter[str] = Counter()
    for text in corpus_texts:
        corpus_freq.update(_tokenize(text))

    scores: list[float] = []
    meta: list[dict[str, float]] = []
    for vector, text in zip(sample_vectors, texts):
        rec_loss = l2_norm(subtract(vector, ref))
        cosine_distance = 1.0 - cosine_similarity(vector, ref)
        perplexity = _rarity_score(text, corpus_freq)
        score = rec_loss + cosine_distance + alpha * perplexity
        scores.append(score)
        meta.append(
            {
                "rec_loss_ids": rec_loss,
                "cosine_distance": cosine_distance,
                "perplexity_proxy": perplexity,
            }
        )
    return scores, meta
