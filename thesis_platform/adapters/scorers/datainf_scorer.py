from __future__ import annotations

from thesis_platform.algorithms.scorers.datainf_core import compute_datainf_scores
from thesis_platform.core.schemas import ScoredSample


class DataInfScorer:
    """Adapter that exposes the DataInf-style scorer through the platform interface."""

    def __init__(self, config, repo_root):
        """Store the DataInf scoring hyper-parameters."""

        del repo_root
        self.lambda_const_param = float(config.get("lambda_const_param", 10.0))
        self.score_direction = str(config.get("score_direction", "larger_is_worse"))

    def score(self, samples, client_ctx):
        """Score synthetic samples against the client's validation anchor pool."""

        sample_vectors = client_ctx.embedder.embed_texts([sample.text for sample in samples])
        validation_pool = client_ctx.validation_samples or client_ctx.train_samples or client_ctx.all_samples
        val_vectors = client_ctx.embedder.embed_texts([sample.text for sample in validation_pool]) if validation_pool else sample_vectors
        scores = compute_datainf_scores(sample_vectors, val_vectors, lambda_const_param=self.lambda_const_param)
        return [
            ScoredSample.from_sample(
                sample,
                client_id=client_ctx.client_id,
                score=float(score),
                score_name="datainf",
                score_direction=self.score_direction,
                meta={"lambda_const_param": self.lambda_const_param},
            )
            for sample, score in zip(samples, scores)
        ]
