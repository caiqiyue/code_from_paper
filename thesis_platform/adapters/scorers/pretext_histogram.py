from __future__ import annotations

from thesis_platform.algorithms.scorers.pretext_histogram import compute_pretext_histogram_scores
from thesis_platform.core.schemas import ScoredSample


class PretextHistogramScorer:
    def __init__(self, config, repo_root):
        del config, repo_root

    def score(self, samples, client_ctx):
        candidate_vectors = client_ctx.embedder.embed_texts([sample.text for sample in samples])
        private_pool = client_ctx.train_samples or client_ctx.all_samples
        private_vectors = client_ctx.embedder.embed_texts([sample.text for sample in private_pool]) if private_pool else candidate_vectors
        scores, metas = compute_pretext_histogram_scores(candidate_vectors, private_vectors)
        return [
            ScoredSample.from_sample(
                sample,
                client_id=client_ctx.client_id,
                score=float(score),
                score_name="pretext_hist",
                score_direction="larger_is_worse",
                meta=meta,
            )
            for sample, score, meta in zip(samples, scores, metas)
        ]
