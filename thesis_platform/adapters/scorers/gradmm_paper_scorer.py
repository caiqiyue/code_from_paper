"""Paper-grade GRADMM scorer using real gradient matching.

Adapted from GRADMM: Gradient Matching for Data Selection
Key insight: Measure synthetic sample quality by gradient mismatch against real samples.

Adaptation notes:
- Uses gradient matching instead of feature-based matching
- Integrates into existing scorer interface
- Does NOT include GRADMM's generation logic (different from our approach)
- Focuses on the core contribution: gradient distance computation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import logging

from thesis_platform.core.schemas import ScoredSample

logger = logging.getLogger(__name__)


def _require_torch() -> Any:
    """Import torch lazily for the real-gradient execution path."""

    try:
        import torch

        return torch
    except Exception as exc:
        raise RuntimeError(
            "torch is required for paper-grade GRADMM real-gradient scoring."
        ) from exc


# ============================================================================
# Core GRADMM Functions (Adapted from GRADMM source)
# ============================================================================

def compute_grads_lm(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    labels: Any,
    create_graph: bool = False,
    grad_clip: str = "",
) -> List[Any]:
    """Compute per-sample gradients w.r.t. model parameters.
    
    Adapted from GRADMM: gradmm/utilities.py:compute_grads_lm
    
    Args:
        model: LoRA-tuned language model
        input_ids: Token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        labels: Target labels [batch_size]
        create_graph: Whether to create computation graph
        grad_clip: Gradient clipping method ("elem", "norm", or "")
        
    Returns:
        List of gradients for each trainable parameter
    """
    torch = _require_torch()
    criterion = torch.nn.CrossEntropyLoss()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, -1, :]  # Last token logits
    
    # Handle batch size
    if labels.shape[0] > 1:
        labels = labels[:1]  # Use first label for all samples
    if input_ids.shape[0] > 1:
        labels = labels.repeat(input_ids.shape[0])
    
    loss = criterion(logits, labels)
    grads = torch.autograd.grad(
        loss,
        (param for param in model.parameters() if param.requires_grad),
        create_graph=create_graph,
        allow_unused=True,
    )
    
    # Gradient clipping
    if grad_clip == "elem":
        grads = [g.clamp_(min=-1, max=1) if g is not None else None for g in grads]
    elif grad_clip == "norm":
        norm = torch.sqrt(sum((g**2).sum() for g in grads if g is not None))
        if norm > 1:
            grads = [g.div_(norm) if g is not None else None for g in grads]
    
    return grads


def compute_average_grads(
    model: Any,
    tokenizer: Any,
    samples: List[Any],
    device: str = "cuda",
) -> List[Any]:
    """Compute average gradients over a set of real samples.
    
    Args:
        model: LoRA-tuned model
        tokenizer: Tokenizer
        samples: List of real samples
        device: Device to use
        
    Returns:
        Average gradients
    """
    all_grads = []
    
    for sample in samples:
        text = sample.rendered_text()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Use a dummy label (for generation task, we use the input itself)
        labels = inputs["input_ids"][:, -1]  # Last token
        
        grads = compute_grads_lm(
            model,
            inputs["input_ids"],
            inputs["attention_mask"],
            labels,
            create_graph=False,
        )
        all_grads.append(grads)
    
    # Average gradients
    avg_grads = []
    n_samples = len(all_grads)
    
    for grad_list in zip(*all_grads):
        avg_grad = sum(g for g in grad_list if g is not None) / n_samples
        avg_grads.append(avg_grad)
    
    return avg_grads


def cos_sim(g1: Any, g2: Any) -> Any:
    """Compute cosine similarity between two gradient tensors.
    
    Adapted from GRADMM: gradmm/utilities.py:cos_sim
    """
    return (g1 * g2).sum() / (g1.norm(p=2) * g2.norm(p=2))


def grad_dist(
    target_grads: List[Any],
    curr_grads: List[Any],
    metric: str = "cos",
) -> Any:
    """Compute gradient distance between two gradient collections.
    
    Adapted from GRADMM: gradmm/filtering.py:grad_dist
    
    Args:
        target_grads: Target gradients (real samples)
        curr_grads: Current gradients (synthetic sample)
        metric: Distance metric ("cos", "dlg", or "tag")
        
    Returns:
        Gradient distance (lower = more similar)
    """
    ret = 0.0
    n_g = 0
    
    for g1, g2 in zip(target_grads, curr_grads):
        if (g1 is not None) and (g2 is not None):
            if metric == "cos":
                # Cosine distance: 1 - cosine similarity
                ret += 1 - cos_sim(g1, g2).item()
            elif metric == "dlg":
                # Deep Leakage from Gradients distance
                ret += (g1 - g2).square().sum().item()
            elif metric == "tag":
                # TAG distance (with L1 regularization)
                ret += (g1 - g2).square().sum().item() + torch.abs(g1 - g2).sum().item()
            n_g += 1
    
    if metric == "cos" and n_g > 0:
        ret /= n_g
    
    torch = _require_torch()
    return torch.tensor(ret)


# ============================================================================
# Paper-Grade GRADMM Scorer
# ============================================================================

class GradMMPaperScorer:
    """Paper-grade GRADMM scorer using real gradient matching.
    
    Core idea from GRADMM:
    - Compute gradients of synthetic samples
    - Compare against average gradients of real samples
    - Higher gradient distance = worse sample = higher score
    
    Differences from GRADMM:
    - Does NOT include GRADMM's generation logic (we use LLM generation)
    - Does NOT include ADMM optimization
    - Focuses only on the gradient matching component
    
    Comparison with current gradmm_core.py:
    - gradmm_core: Uses feature vectors (L2 distance, cosine in feature space)
    - GradMMPaperScorer: Uses real gradients (cosine in gradient space)
    """

    def __init__(self, config: Dict[str, Any], repo_root: str):
        """Initialize the GRADMM paper scorer.
        
        Args:
            config: Configuration dictionary
            repo_root: Repository root path
        """
        self.score_direction = str(config.get("score_direction", "larger_is_worse"))
        self.metric = str(config.get("metric", "cos"))  # cos, dlg, tag
        self.grad_clip = str(config.get("grad_clip", ""))  # elem, norm, or ""
        try:
            torch = _require_torch()
        except RuntimeError:
            torch = None
        default_device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        self.device = str(config.get("device", default_device))
        self.use_real_gradients = bool(config.get("use_real_gradients", True))
        self.repo_root = Path(repo_root) if repo_root is not None else Path(".")
        
        # Model configuration (for loading LoRA model)
        self.model_name = config.get("model_name", "microsoft/phi-1_5")
        self.lora_checkpoint = config.get("lora_checkpoint")
        self.feature_model = config.get("feature_model")
        self.allow_hashing_fallback = bool(config.get("allow_hashing_fallback", False))
        self.max_length = int(config.get("max_length", 256))
        
        # Fallback feature encoder when gradients are disabled or unavailable.
        self.feature_encoder = None
        
        self._model_cache: Dict[str, Any] = {}

    def _get_feature_encoder(self):
        """Lazily initialize the feature fallback used when gradients are unavailable."""

        if self.feature_encoder is None:
            from thesis_platform.models.features import build_feature_encoder

            self.feature_encoder = build_feature_encoder(
                self.feature_model,
                self.repo_root,
                allow_fallback=self.allow_hashing_fallback,
                max_length=self.max_length,
                device=self.device,
            )
        return self.feature_encoder

    def _get_model_for_client(self, client_ctx: Any) -> Tuple[Any, Any, str]:
        """Get or load LoRA model for a client.

        Returns:
            Tuple of (model, tokenizer, device)
            Returns (None, None, None) if model cannot be loaded
        """
        client_id = getattr(client_ctx, 'client_id', 'default')

        if client_id in self._model_cache:
            return self._model_cache[client_id]

        try:
            from thesis_platform.core.lora_gradients import LoRAGradientExtractor

            extractor = LoRAGradientExtractor(
                model_name_or_path=self.model_name,
                device=self.device,
                lora_rank=8,
                target_modules=["q_proj", "v_proj"],
            )
            extractor.load_model(lora_adapter_path=self.lora_checkpoint)
            self._model_cache[client_id] = (extractor.model, extractor.tokenizer, extractor.device)
            return self._model_cache[client_id]
        except Exception as e:
            logger.warning(f"Failed to load LoRA model for client {client_id}: {e}")
            self._model_cache[client_id] = (None, None, None)
            return (None, None, None)

    def score(self, samples: List[Any], client_ctx: Any) -> List[ScoredSample]:
        """Score synthetic samples using paper-grade GRADMM.

        Strategy:
        1. Try to use real gradients from LoRA model
        2. Fall back to feature-based approximation if not available
        """
        fallback_reason = "real_gradients_disabled"
        if self.use_real_gradients:
            try:
                _require_torch()
            except RuntimeError:
                fallback_reason = "LoRA gradients not available"
                return self._score_with_features(samples, client_ctx, fallback_reason=fallback_reason)
            model, tokenizer, device = self._get_model_for_client(client_ctx)
            if model is not None and tokenizer is not None:
                return self._score_with_real_gradients(samples, model, tokenizer, device, client_ctx)
            fallback_reason = "LoRA gradients not available"
        return self._score_with_features(samples, client_ctx, fallback_reason=fallback_reason)

    def _score_with_real_gradients(
        self,
        samples: List[Any],
        model: Any,
        tokenizer: Any,
        device: str,
        client_ctx: Any,
    ) -> List[ScoredSample]:
        """Score using real gradient matching."""
        from thesis_platform.core.lora_gradients import LoRAGradientExtractor

        extractor = LoRAGradientExtractor(
            model_name_or_path=self.model_name,
            device=device,
        )
        extractor.model = model
        extractor.tokenizer = tokenizer

        # Compute average gradient of real samples
        real_samples = client_ctx.train_samples or client_ctx.all_samples
        real_texts = [s.rendered_text() for s in real_samples]
        real_avg_grads = extractor.compute_average_gradients(real_texts)

        # Compute gradient distance for each synthetic sample
        scored_samples = []
        for sample in samples:
            text = sample.rendered_text()
            syn_grads = extractor.compute_sample_gradients(text, return_only_lora=False)

            # Compute gradient distance
            dist = grad_dist(
                [real_avg_grads[k] for k in sorted(real_avg_grads.keys())],
                [syn_grads[k] for k in sorted(syn_grads.keys())],
                metric=self.metric,
            ).item()

            scored_samples.append(
                ScoredSample.from_sample(
                    sample,
                    client_id=client_ctx.client_id,
                    score=float(dist),
                    score_name="gradmm_paper",
                    score_direction=self.score_direction,
                    meta={
                        "gradient_distance": float(dist),
                        "metric": self.metric,
                        "use_real_gradients": True,
                    },
                )
            )

        client_ctx.probe_state["last_metrics"] = {
            "objective": "paper_gradmm_gradient_mismatch",
            "metric": self.metric,
            "use_real_gradients": True,
        }

        return scored_samples

    def _score_with_features(
        self, samples: List[Any], client_ctx: Any, *, fallback_reason: str
    ) -> List[ScoredSample]:
        """Fallback: Score using feature vectors (from gradmm_real_scorer)."""
        from thesis_platform.algorithms.scorers.gradmm_core import compute_gradmm_scores

        feature_encoder = self._get_feature_encoder()
        cache = client_ctx.probe_state.setdefault("gradmm_paper_cache", {})
        reference_samples = client_ctx.train_samples or client_ctx.all_samples
        reference_texts = [sample.rendered_text() for sample in reference_samples]
        sample_texts = [sample.rendered_text() for sample in samples]

        def _cache_encoded_texts(cache_key: str, texts: list[str]) -> list[list[float]]:
            cached = cache.get(cache_key)
            if cached is not None and cached.get("texts") == texts:
                return list(cached["vectors"])
            vectors = feature_encoder.encode_texts(texts)
            cache[cache_key] = {"texts": list(texts), "vectors": vectors}
            return vectors

        reference_vectors = _cache_encoded_texts("reference", reference_texts)
        sample_vectors = feature_encoder.encode_texts(sample_texts)

        scores, metas = compute_gradmm_scores(
            sample_vectors,
            reference_vectors,
            texts=sample_texts,
            corpus_texts=reference_texts,
            alpha=0.25,
        )

        scored_samples = [
            ScoredSample.from_sample(
                sample,
                client_id=client_ctx.client_id,
                score=float(score),
                score_name="gradmm_paper",
                score_direction=self.score_direction,
                meta={
                    **meta,
                    "feature_backend": feature_encoder.backend_name,
                    "use_real_gradients": False,
                    "fallback_reason": fallback_reason,
                },
            )
            for sample, score, meta in zip(samples, scores, metas)
        ]

        client_ctx.probe_state["last_metrics"] = {
            "objective": "paper_gradmm_feature_fallback",
            "feature_backend": feature_encoder.backend_name,
            "use_real_gradients": False,
            "fallback_reason": fallback_reason,
        }

        return scored_samples
