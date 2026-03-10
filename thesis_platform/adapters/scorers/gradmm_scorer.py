from __future__ import annotations

from thesis_platform.algorithms.scorers.gradmm_core import compute_gradmm_scores
from thesis_platform.core.schemas import ScoredSample


class GradMMScorer:
    def __init__(self, config, repo_root):
        del repo_root
        self.alpha = float(config.get("alpha", 0.5))
        self.score_direction = str(config.get("score_direction", "larger_is_worse"))

    def score(self, samples, client_ctx):
        sample_vectors = client_ctx.embedder.embed_texts([sample.text for sample in samples])
        reference_pool = client_ctx.train_samples or client_ctx.all_samples
        reference_vectors = client_ctx.embedder.embed_texts([sample.text for sample in reference_pool]) if reference_pool else sample_vectors
        scores, metas = compute_gradmm_scores(
            sample_vectors,
            reference_vectors,
            texts=[sample.text for sample in samples],
            corpus_texts=[sample.text for sample in reference_pool],
            alpha=self.alpha,
        )
        return [
            ScoredSample.from_sample(
                sample,
                client_id=client_ctx.client_id,
                score=float(score),
                score_name="gradmm",
                score_direction=self.score_direction,
                meta=meta,
            )
            for sample, score, meta in zip(samples, scores, metas)
        ]
