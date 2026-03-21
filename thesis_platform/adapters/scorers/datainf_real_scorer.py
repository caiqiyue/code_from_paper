from __future__ import annotations

from thesis_platform.algorithms.math_utils import cosine_similarity, mean_vector
from thesis_platform.algorithms.scorers.datainf_core import compute_datainf_scores
from thesis_platform.core.schemas import ScoredSample
from thesis_platform.models.features import build_feature_encoder


def _cache_encoded_texts(*, cache: dict, cache_key: str, texts: list[str], encoder) -> list[list[float]]:
    cached = cache.get(cache_key)
    if cached is not None and cached.get("texts") == texts:
        return list(cached["vectors"])
    vectors = encoder.encode_texts(texts)
    cache[cache_key] = {"texts": list(texts), "vectors": vectors}
    return vectors


class DataInfRealScorer:
    """V3 scorer using real transformer features plus DataInf-style influence ranking."""

    def __init__(self, config, repo_root):
        self.score_direction = str(config.get("score_direction", "larger_is_worse"))
        self.lambda_const_param = float(config.get("lambda_const_param", 10.0))
        self.feature_encoder = build_feature_encoder(
            config.get("feature_model"),
            repo_root,
            allow_fallback=bool(config.get("allow_hashing_fallback", False)),
            max_length=int(config.get("max_length", 256)),
            device=str(config.get("device", "auto")),
        )

    def score(self, samples, client_ctx):
        """Score synthetic samples against real client slices using real transformer features."""

        cache = client_ctx.probe_state.setdefault("datainf_real_cache", {})
        train_texts = [sample.rendered_text() for sample in (client_ctx.train_samples or client_ctx.all_samples)]
        val_texts = [sample.rendered_text() for sample in (client_ctx.validation_samples or client_ctx.train_samples or client_ctx.all_samples)]
        sample_texts = [sample.rendered_text() for sample in samples]

        train_vectors = _cache_encoded_texts(cache=cache, cache_key="train", texts=train_texts, encoder=self.feature_encoder)
        val_vectors = _cache_encoded_texts(cache=cache, cache_key="val", texts=val_texts, encoder=self.feature_encoder)
        sample_vectors = self.feature_encoder.encode_texts(sample_texts)
        influence_scores = compute_datainf_scores(
            sample_vectors,
            val_vectors or train_vectors,
            lambda_const_param=self.lambda_const_param,
        )
        reference = mean_vector(train_vectors or val_vectors)

        scored_samples: list[ScoredSample] = []
        for sample, vector, influence in zip(samples, sample_vectors, influence_scores):
            domain_gap = 1.0 - cosine_similarity(vector, reference)
            score = float(influence) + max(0.0, domain_gap)
            scored_samples.append(
                ScoredSample.from_sample(
                    sample,
                    client_id=client_ctx.client_id,
                    score=score,
                    score_name="datainf_real",
                    score_direction=self.score_direction,
                    meta={
                        "influence_score": float(influence),
                        "domain_gap": float(domain_gap),
                        "feature_backend": self.feature_encoder.backend_name,
                    },
                )
            )

        client_ctx.probe_state["last_metrics"] = {
            "objective": "real_feature_influence",
            "feature_backend": self.feature_encoder.backend_name,
            "train_count": len(train_texts),
            "validation_count": len(val_texts),
        }
        return scored_samples
