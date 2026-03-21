from __future__ import annotations

from thesis_platform.algorithms.scorers.gradmm_core import compute_gradmm_scores
from thesis_platform.core.schemas import ScoredSample
from thesis_platform.models.features import build_feature_encoder

from thesis_platform.adapters.scorers.datainf_real_scorer import _cache_encoded_texts


class GradMMRealScorer:
    """V3 scorer using real transformer features plus gradient-mismatch ranking."""

    def __init__(self, config, repo_root):
        self.score_direction = str(config.get("score_direction", "larger_is_worse"))
        self.alpha = float(config.get("alpha", 0.25))
        self.feature_encoder = build_feature_encoder(
            config.get("feature_model"),
            repo_root,
            allow_fallback=bool(config.get("allow_hashing_fallback", False)),
            max_length=int(config.get("max_length", 256)),
            device=str(config.get("device", "auto")),
        )

    def score(self, samples, client_ctx):
        """Score synthetic samples by mismatch against the client's real feature distribution."""

        cache = client_ctx.probe_state.setdefault("gradmm_real_cache", {})
        reference_samples = client_ctx.train_samples or client_ctx.all_samples
        reference_texts = [sample.rendered_text() for sample in reference_samples]
        sample_texts = [sample.rendered_text() for sample in samples]

        reference_vectors = _cache_encoded_texts(
            cache=cache,
            cache_key="reference",
            texts=reference_texts,
            encoder=self.feature_encoder,
        )
        sample_vectors = self.feature_encoder.encode_texts(sample_texts)
        scores, metas = compute_gradmm_scores(
            sample_vectors,
            reference_vectors,
            texts=sample_texts,
            corpus_texts=reference_texts,
            alpha=self.alpha,
        )
        client_ctx.probe_state["last_metrics"] = {
            "objective": "real_gradient_mismatch",
            "feature_backend": self.feature_encoder.backend_name,
            "reference_count": len(reference_texts),
        }
        return [
            ScoredSample.from_sample(
                sample,
                client_id=client_ctx.client_id,
                score=float(score),
                score_name="gradmm_real",
                score_direction=self.score_direction,
                meta={**meta, "feature_backend": self.feature_encoder.backend_name},
            )
            for sample, score, meta in zip(samples, scores, metas)
        ]
