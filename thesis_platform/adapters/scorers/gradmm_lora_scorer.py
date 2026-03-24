"""Real LoRA-based GRADMM scorer.

This implementation uses actual LoRA gradients for gradient mismatch computation,
providing paper-grade quality assessment as described in GRADMM.

Key adaptations from GRADMM source (D:\\学习记录\\导师项目\\研究\\caiqiyue_file\\GRADMM):
- Uses compute_grads_lm approach for gradient extraction
- Implements gradient distance metrics (cos, dlg, tag)
- Calculates mismatch scores between synthetic and real sample gradients

For your innovation algorithm:
- This scorer measures how well synthetic samples match real data distribution
- High gradient mismatch = poor quality synthetic samples
- Compatible with federated client contexts
"""

from __future__ import annotations

import torch
from typing import Any, Dict, List, Optional
import logging

from thesis_platform.core.schemas import ScoredSample
from thesis_platform.core.lora_gradients import (
    LoRAGradientExtractor,
    GradientDistanceCalculator,
)
from thesis_platform.algorithms.scorers.gradmm_core import compute_gradmm_scores
from thesis_platform.models.features import build_feature_encoder
from thesis_platform.adapters.scorers.datainf_lora_scorer import DataInfRealScorer

logger = logging.getLogger(__name__)


class GradMMRealScorer:
    """Real GRADMM scorer using LoRA gradients for gradient matching.

    Adapted from GRADMM paper implementation. Key insight:
    - Synthetic sample quality ∝ gradient similarity to real samples
    - High gradient mismatch = poor quality (needs improvement)

    For your federated algorithm:
    1. Each client computes average gradient on their real samples
    2. Synthetic samples with high gradient mismatch are selected as "bad samples"
    3. These samples are sent for critique and prompt improvement
    """

    def __init__(self, config: Dict[str, Any], repo_root: str):
        """Initialize the GRADMM real scorer.

        Args:
            config: Configuration dictionary
            repo_root: Repository root path
        """
        self.score_direction = str(config.get("score_direction", "larger_is_worse"))
        self.metric = str(config.get("metric", "cos"))  # cos, dlg, tag
        self.grad_clip = str(config.get("grad_clip", ""))  # elem, norm, or ""
        self.device = str(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        # Model configuration
        self.model_name = config.get("model_name", "microsoft/phi-1_5")
        self.lora_rank = int(config.get("lora_rank", 8))
        self.target_modules = config.get("target_modules", ["q_proj", "v_proj"])

        # Feature encoder for fallback
        self.use_real_gradients = bool(config.get("use_real_gradients", True))
        self.feature_encoder = None

        if not self.use_real_gradients:
            self.feature_encoder = build_feature_encoder(
                config.get("feature_model"),
                repo_root,
                allow_fallback=bool(config.get("allow_hashing_fallback", False)),
                max_length=int(config.get("max_length", 256)),
                device=self.device,
            )

        # Gradient extractor (lazily initialized)
        self._gradient_extractor: Optional[LoRAGradientExtractor] = None

        # Cache for gradients
        self._grad_cache: Dict[str, Any] = {}

    def _get_gradient_extractor(self) -> LoRAGradientExtractor:
        """Get or initialize the gradient extractor."""
        if self._gradient_extractor is None:
            try:
                self._gradient_extractor = LoRAGradientExtractor(
                    model_name_or_path=self.model_name,
                    device=self.device,
                    lora_rank=self.lora_rank,
                    target_modules=self.target_modules,
                )
                self._gradient_extractor.load_model(lora_adapter_path=None)
                logger.info("LoRA gradient extractor initialized for GRADMM")
            except Exception as e:
                logger.error(f"Failed to initialize gradient extractor: {e}")
                raise

        return self._gradient_extractor

    def _compute_gradient_distance(
        self,
        real_grads: Dict[str, torch.Tensor],
        syn_grads: Dict[str, torch.Tensor],
    ) -> float:
        """Compute gradient distance using specified metric.

        From GRADMM paper:
        - cos: 1 - cosine_similarity (default)
        - dlg: (g_real - g_syn)^2 (Deep Leakage from Gradients)
        - tag: dlg + λ|g_real - g_syn| (with L1 regularization)

        All metrics measure how different the synthetic sample gradient is
        from the real sample distribution.
        """
        if self.metric == "cos":
            return GradientDistanceCalculator.cosine_distance(real_grads, syn_grads)
        elif self.metric == "euclidean" or self.metric == "dlg":
            return GradientDistanceCalculator.euclidean_distance(real_grads, syn_grads)
        elif self.metric == "l1" or self.metric == "tag":
            return GradientDistanceCalculator.l1_distance(real_grads, syn_grads)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def score(self, samples: List[Any], client_ctx: Any) -> List[ScoredSample]:
        """Score synthetic samples using GRADMM gradient matching.

        Strategy:
        1. Try to use real LoRA gradients
        2. Fall back to feature-based scoring if gradients unavailable

        Args:
            samples: List of synthetic samples to score
            client_ctx: Client context with train/validation data

        Returns:
            List of scored samples
        """
        if self.use_real_gradients:
            try:
                return self._score_with_real_gradients(samples, client_ctx)
            except Exception as e:
                logger.warning(
                    f"Real gradient scoring failed: {e}. Falling back to features."
                )
                return self._score_with_features(samples, client_ctx)
        else:
            return self._score_with_features(samples, client_ctx)

    def _score_with_real_gradients(
        self,
        samples: List[Any],
        client_ctx: Any,
    ) -> List[ScoredSample]:
        """Score samples using real LoRA gradients.

        This is the paper-grade implementation:
        1. Compute average gradient of real samples
        2. Compute gradient of each synthetic sample
        3. Measure gradient distance (mismatch)
        4. Higher mismatch = worse sample
        """
        extractor = self._get_gradient_extractor()

        # Step 1: Compute average gradient of real samples
        real_samples = client_ctx.train_samples or client_ctx.all_samples
        real_texts = [sample.rendered_text() for sample in real_samples]
        real_grads = extractor.compute_average_gradients(real_texts)

        if not real_grads:
            logger.warning("No real gradients computed, falling back to features")
            return self._score_with_features(samples, client_ctx)

        # Step 2 & 3: Compute gradient for each synthetic sample and measure distance
        scored_samples = []

        for sample in samples:
            text = sample.rendered_text()
            syn_grads = extractor.compute_sample_gradients(text)

            if self.grad_clip:
                from thesis_platform.core.lora_gradients import clip_gradients

                syn_grads = clip_gradients(syn_grads, max_norm=1.0)

            # Step 4: Compute gradient distance (GRADMM core)
            mismatch_score = self._compute_gradient_distance(real_grads, syn_grads)

            scored_samples.append(
                ScoredSample.from_sample(
                    sample,
                    client_id=client_ctx.client_id,
                    score=mismatch_score,
                    score_name="gradmm_real_lora",
                    score_direction=self.score_direction,
                    meta={
                        "mismatch_score": mismatch_score,
                        "metric": self.metric,
                        "use_real_gradients": True,
                        "grad_clip": self.grad_clip,
                    },
                )
            )

        client_ctx.probe_state["last_metrics"] = {
            "objective": "lora_gradient_mismatch",
            "metric": self.metric,
            "use_real_gradients": True,
            "real_sample_count": len(real_samples),
        }

        return scored_samples

    def _score_with_features(
        self, samples: List[Any], client_ctx: Any
    ) -> List[ScoredSample]:
        """Fallback: Score using feature vectors (original implementation)."""
        cache = client_ctx.probe_state.setdefault("gradmm_real_cache", {})
        reference_samples = client_ctx.train_samples or client_ctx.all_samples
        reference_texts = [sample.rendered_text() for sample in reference_samples]
        sample_texts = [sample.rendered_text() for sample in samples]

        reference_vectors = self._cache_encoded_texts(
            cache=cache,
            cache_key="reference",
            texts=reference_texts,
            encoder=self.feature_encoder,
        )
        sample_vectors = self.feature_encoder.encode_texts(sample_texts)

        alpha = 0.25  # Default alpha from original implementation
        scores, metas = compute_gradmm_scores(
            sample_vectors,
            reference_vectors,
            texts=sample_texts,
            corpus_texts=reference_texts,
            alpha=alpha,
        )

        scored_samples = [
            ScoredSample.from_sample(
                sample,
                client_id=client_ctx.client_id,
                score=float(score),
                score_name="gradmm_real_lora",
                score_direction=self.score_direction,
                meta={
                    **meta,
                    "feature_backend": self.feature_encoder.backend_name,
                    "use_real_gradients": False,
                    "fallback_reason": "LoRA gradients not available",
                },
            )
            for sample, score, meta in zip(samples, scores, metas)
        ]

        client_ctx.probe_state["last_metrics"] = {
            "objective": "gradient_mismatch_fallback",
            "metric": "feature_based",
            "use_real_gradients": False,
            "feature_backend": self.feature_encoder.backend_name,
        }

        return scored_samples

    @staticmethod
    def _cache_encoded_texts(
        *, cache: dict, cache_key: str, texts: list[str], encoder
    ) -> list[list[float]]:
        """Cache encoded texts to avoid recomputation."""
        cached = cache.get(cache_key)
        if cached is not None and cached.get("texts") == texts:
            return list(cached["vectors"])
        vectors = encoder.encode_texts(texts)
        cache[cache_key] = {"texts": list(texts), "vectors": vectors}
        return vectors

    def __del__(self):
        """Cleanup resources."""
        if self._gradient_extractor is not None:
            self._gradient_extractor.release()


class GreedyGradMMSelector:
    """Greedy gradient selection for top-k bad sample selection (optional enhancement).

    From GRADMM paper: Gradually select samples that minimize gradient distance.
    This can be used as an alternative to individual scoring.

    Note: This is optional and may be overkill for your use case.
    Individual scoring is usually sufficient for "bad sample" selection.
    """

    def __init__(self, scorer: GradMMRealScorer):
        self.scorer = scorer

    def select_top_k(
        self,
        samples: List[Any],
        client_ctx: Any,
        top_k: int = 10,
    ) -> List[int]:
        """Select top-k samples using greedy gradient matching.

        Algorithm:
        1. Start with empty selected set
        2. Iteratively add the sample that minimizes gradient distance
           when combined with already selected samples
        3. Return indices of selected samples

        Args:
            samples: Candidate samples
            client_ctx: Client context
            top_k: Number of samples to select

        Returns:
            Indices of selected samples
        """
        extractor = self.scorer._get_gradient_extractor()

        # Compute real average gradient
        real_samples = client_ctx.train_samples or client_ctx.all_samples
        real_texts = [s.rendered_text() for s in real_samples]
        real_grads = extractor.compute_average_gradients(real_texts)

        # Compute gradients for all synthetic samples
        sample_grads = []
        for sample in samples:
            text = sample.rendered_text()
            grad = extractor.compute_sample_gradients(text)
            sample_grads.append(grad)

        # Greedy selection
        selected_indices = []
        remaining_indices = list(range(len(samples)))
        current_combined_grads = None

        for _ in range(min(top_k, len(samples))):
            best_idx = None
            best_dist = float("inf")

            for idx in remaining_indices:
                # Combine with current selection
                if current_combined_grads is None:
                    combined = sample_grads[idx]
                else:
                    combined = {}
                    n_selected = len(selected_indices)
                    for name in current_combined_grads:
                        combined[name] = (
                            current_combined_grads[name] * n_selected
                            + sample_grads[idx][name]
                        ) / (n_selected + 1)

                # Compute distance
                dist = self.scorer._compute_gradient_distance(real_grads, combined)

                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

                # Update combined gradients
                if current_combined_grads is None:
                    current_combined_grads = sample_grads[best_idx]
                else:
                    n_selected = len(selected_indices)
                    for name in current_combined_grads:
                        current_combined_grads[name] = (
                            current_combined_grads[name] * (n_selected - 1)
                            + sample_grads[best_idx][name]
                        ) / n_selected

        return selected_indices
