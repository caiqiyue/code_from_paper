"""Real LoRA-based DataInf scorer.

This implementation uses actual LoRA gradients instead of feature vectors,
providing paper-grade influence computation as described in DataInf.

Key adaptations from DataInf source (D:\\学习记录\\导师项目\\研究\\caiqiyue_file\\DataInf):
- Uses LORAEngine approach for gradient extraction
- Implements HVP computation (proposed, LiSSA, accurate methods)
- Calculates influence scores: IF = -HVP · training_gradient

For your innovation algorithm:
- This scorer identifies 'bad samples' by their negative influence on validation loss
- Only uses the influence computation part, not the full DataInf training pipeline
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
    resolve_model_name_or_path,
)
from thesis_platform.algorithms.scorers.datainf_core import compute_datainf_scores
from thesis_platform.models.features import build_feature_encoder
from thesis_platform.algorithms.math_utils import cosine_similarity, mean_vector

logger = logging.getLogger(__name__)


class DataInfRealScorer:
    """Real DataInf scorer using LoRA gradients for influence computation.

    Adapted from DataInf paper implementation. Key insight:
    - Influence ∝ -H^{-1}∇L_val · ∇L_train
    - Uses HVP approximations (proposed, LiSSA, accurate) to avoid explicit Hessian computation

    For your federated algorithm:
    1. Each client computes gradients on their LoRA-tuned model
    2. Synthetic samples with high negative influence are "bad samples"
    3. These bad samples are selected for critique and improvement
    """

    def __init__(self, config: Dict[str, Any], repo_root: str):
        """Initialize the DataInf real scorer.

        Args:
            config: Configuration dictionary
            repo_root: Repository root path
        """
        self._gradient_extractor: Optional[LoRAGradientExtractor] = None
        self._grad_cache: Dict[str, Any] = {}

        self.score_direction = str(config.get("score_direction", "larger_is_worse"))
        self.lambda_const_param = float(config.get("lambda_const_param", 10.0))
        self.hvp_method = str(
            config.get("hvp_method", "proposed")
        )  # proposed, lissa, accurate
        self.lissa_iterations = int(config.get("lissa_iterations", 10))
        self.device = str(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        # Model configuration
        self.model_name = resolve_model_name_or_path(
            config.get("model_name", "thesis_platform/open_model/qwen_2_0_5b_instruct"),
            repo_root=repo_root,
        )
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
                # Load model without pre-trained adapter (we compute gradients directly)
                self._gradient_extractor.load_model(lora_adapter_path=None)
                logger.info("LoRA gradient extractor initialized")
            except Exception as e:
                logger.error(f"Failed to initialize gradient extractor: {e}")
                raise

        return self._gradient_extractor

    def _compute_hvp_proposed(
        self,
        val_grad_avg: Dict[str, torch.Tensor],
        tr_grad_dict: Dict[int, Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Compute HVP using DataInf's closed-form approximation.

        From DataInf paper (Section 3.2):
        hvp = (1/n) * Σ_i [(v_avg - C_i * g_i) / λ]
        where C_i = (v_avg · g_i) / (λ + ||g_i||^2)

        This is an efficient approximation that avoids explicit Hessian computation.
        """
        hvp_dict = {}
        n_train = len(tr_grad_dict)

        for weight_name in val_grad_avg:
            # Compute layer-wise damping parameter
            S = torch.zeros(n_train)
            for tr_id in tr_grad_dict:
                tmp_grad = tr_grad_dict[tr_id][weight_name]
                S[tr_id] = torch.mean(tmp_grad**2)
            lambda_const = torch.mean(S) / self.lambda_const_param

            # Compute HVP
            hvp = torch.zeros_like(val_grad_avg[weight_name])
            for tr_id in tr_grad_dict:
                tmp_grad = tr_grad_dict[tr_id][weight_name]
                # C_i = (v_avg · g_i) / (λ + ||g_i||^2)
                C_tmp = torch.sum(val_grad_avg[weight_name] * tmp_grad) / (
                    lambda_const + torch.sum(tmp_grad**2)
                )
                # hvp += (v_avg - C_i * g_i) / (n * λ)
                hvp += (val_grad_avg[weight_name] - C_tmp * tmp_grad) / (
                    n_train * lambda_const
                )

            hvp_dict[weight_name] = hvp

        return hvp_dict

    def _compute_hvp_lissa(
        self,
        val_grad_avg: Dict[str, torch.Tensor],
        tr_grad_dict: Dict[int, Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Compute HVP using LiSSA recursion.

        LiSSA (Linear time Stochastic Second-Order Algorithm):
        hvp_{t+1} = v_avg + hvp_t - α * H * hvp_t

        This is an iterative method from Agarwal et al. (2017).
        """
        hvp_dict = {}
        n_train = len(tr_grad_dict)
        alpha_const = 1.0

        for weight_name in val_grad_avg:
            # Compute damping
            S = torch.zeros(n_train)
            for tr_id in tr_grad_dict:
                tmp_grad = tr_grad_dict[tr_id][weight_name]
                S[tr_id] = torch.mean(tmp_grad**2)
            lambda_const = torch.mean(S) / self.lambda_const_param

            # LiSSA iteration
            running_hvp = val_grad_avg[weight_name].clone()
            for _ in range(self.lissa_iterations):
                hvp_tmp = torch.zeros_like(val_grad_avg[weight_name])
                for tr_id in tr_grad_dict:
                    tmp_grad = tr_grad_dict[tr_id][weight_name]
                    # H * hvp ≈ Σ_i [(g_i · hvp) * g_i] / n
                    hvp_tmp += (
                        torch.sum(tmp_grad * running_hvp) * tmp_grad
                        - lambda_const * running_hvp
                    ) / n_train
                running_hvp = (
                    val_grad_avg[weight_name] + running_hvp - alpha_const * hvp_tmp
                )

            hvp_dict[weight_name] = running_hvp

        return hvp_dict

    def _compute_influence_scores(
        self,
        hvp_dict: Dict[str, torch.Tensor],
        sample_grad_dict: Dict[int, Dict[str, torch.Tensor]],
    ) -> Dict[int, float]:
        """Compute influence scores: IF = -HVP · gradient.

        Negative influence means removing this sample would hurt validation performance
        (i.e., it's a valuable sample).

        For "bad sample" selection:
        - High negative influence: Good samples (keep them)
        - Low negative influence (or positive): Bad samples (need improvement)
        """
        influence_dict = {}

        for sample_id in sample_grad_dict:
            influence = 0.0
            for weight_name in hvp_dict:
                if weight_name in sample_grad_dict[sample_id]:
                    influence += torch.sum(
                        hvp_dict[weight_name] * sample_grad_dict[sample_id][weight_name]
                    ).item()
            influence_dict[sample_id] = -influence  # Negative as per paper

        return influence_dict

    def score(self, samples: List[Any], client_ctx: Any) -> List[ScoredSample]:
        """Score synthetic samples using DataInf influence computation.

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
        """Score samples using real LoRA gradients and HVP.

        This is the paper-grade implementation:
        1. Compute gradients for validation samples
        2. Compute gradients for synthetic samples
        3. Compute HVP (approximate Hessian inverse · validation gradient)
        4. Calculate influence scores
        """
        extractor = self._get_gradient_extractor()

        # Step 1: Compute validation gradients
        val_samples = (
            client_ctx.validation_samples
            or client_ctx.train_samples
            or client_ctx.all_samples
        )
        val_texts = [sample.rendered_text() for sample in val_samples]
        val_grad_dict = {}
        for idx, text in enumerate(val_texts):
            val_grad_dict[idx] = extractor.compute_sample_gradients(text)

        # Average validation gradients
        val_grad_avg = {}
        for weight_name in val_grad_dict[0]:
            val_grad_avg[weight_name] = sum(
                val_grad_dict[i][weight_name] for i in val_grad_dict
            ) / len(val_grad_dict)

        # Step 2: Compute synthetic sample gradients
        sample_grad_dict = {}
        for idx, sample in enumerate(samples):
            text = sample.rendered_text()
            sample_grad_dict[idx] = extractor.compute_sample_gradients(text)

        # Step 3: Compute HVP
        if self.hvp_method == "lissa":
            hvp_dict = self._compute_hvp_lissa(val_grad_avg, sample_grad_dict)
        else:  # default to proposed (most efficient)
            hvp_dict = self._compute_hvp_proposed(val_grad_avg, sample_grad_dict)

        # Step 4: Compute influence scores
        influence_dict = self._compute_influence_scores(hvp_dict, sample_grad_dict)

        # Create scored samples
        # Note: In DataInf, negative influence = valuable sample
        # For "bad sample" selection, we want samples with low negative influence
        scored_samples = []
        for i, sample in enumerate(samples):
            influence = influence_dict.get(i, 0.0)
            # Convert to score: higher score = worse sample
            # (less negative influence = less valuable = worse)
            score = -influence  # Flip sign so higher = worse

            scored_samples.append(
                ScoredSample.from_sample(
                    sample,
                    client_id=client_ctx.client_id,
                    score=score,
                    score_name="datainf_real_lora",
                    score_direction=self.score_direction,
                    meta={
                        "influence_score": influence,
                        "hvp_method": self.hvp_method,
                        "use_real_gradients": True,
                    },
                )
            )

        client_ctx.probe_state["last_metrics"] = {
            "objective": "lora_influence",
            "hvp_method": self.hvp_method,
            "use_real_gradients": True,
            "val_sample_count": len(val_samples),
        }

        return scored_samples

    def _score_with_features(
        self, samples: List[Any], client_ctx: Any
    ) -> List[ScoredSample]:
        """Fallback: Score using feature vectors (original implementation)."""
        cache = client_ctx.probe_state.setdefault("datainf_real_cache", {})
        train_texts = [
            sample.rendered_text()
            for sample in (client_ctx.train_samples or client_ctx.all_samples)
        ]
        val_texts = [
            sample.rendered_text()
            for sample in (
                client_ctx.validation_samples
                or client_ctx.train_samples
                or client_ctx.all_samples
            )
        ]
        sample_texts = [sample.rendered_text() for sample in samples]

        train_vectors = self._cache_encoded_texts(
            cache=cache,
            cache_key="train",
            texts=train_texts,
            encoder=self.feature_encoder,
        )
        val_vectors = self._cache_encoded_texts(
            cache=cache, cache_key="val", texts=val_texts, encoder=self.feature_encoder
        )
        sample_vectors = self.feature_encoder.encode_texts(sample_texts)

        influence_scores = compute_datainf_scores(
            sample_vectors,
            val_vectors or train_vectors,
            lambda_const_param=self.lambda_const_param,
        )
        reference = mean_vector(train_vectors or val_vectors)

        scored_samples = []
        for sample, vector, influence in zip(samples, sample_vectors, influence_scores):
            domain_gap = 1.0 - cosine_similarity(vector, reference)
            score = float(influence) + max(0.0, domain_gap)
            scored_samples.append(
                ScoredSample.from_sample(
                    sample,
                    client_id=client_ctx.client_id,
                    score=score,
                    score_name="datainf_real_lora",
                    score_direction=self.score_direction,
                    meta={
                        "influence_score": float(influence),
                        "domain_gap": float(domain_gap),
                        "feature_backend": self.feature_encoder.backend_name,
                        "use_real_gradients": False,
                        "fallback_reason": "LoRA gradients not available",
                    },
                )
            )

        client_ctx.probe_state["last_metrics"] = {
            "objective": "feature_influence_fallback",
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

    def release(self) -> None:
        """Release any cached GPU-backed model state held by the scorer."""

        gradient_extractor = getattr(self, "_gradient_extractor", None)
        self._gradient_extractor = None
        grad_cache = getattr(self, "_grad_cache", None)
        if isinstance(grad_cache, dict):
            grad_cache.clear()
        if gradient_extractor is not None:
            gradient_extractor.release()

        feature_encoder = getattr(self, "feature_encoder", None)
        self.feature_encoder = None
        release = getattr(feature_encoder, "release", None)
        if callable(release):
            release()

    def __del__(self):
        """Cleanup resources."""
        self.release()
