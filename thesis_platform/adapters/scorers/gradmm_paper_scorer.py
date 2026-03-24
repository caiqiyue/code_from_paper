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

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
import logging

from thesis_platform.core.schemas import ScoredSample
from thesis_platform.algorithms.math_utils import cosine_similarity

logger = logging.getLogger(__name__)


# ============================================================================
# Core GRADMM Functions (Adapted from GRADMM source)
# ============================================================================

def compute_grads_lm(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    create_graph: bool = False,
    grad_clip: str = "",
) -> List[torch.Tensor]:
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
    criterion = nn.CrossEntropyLoss()
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
    model: torch.nn.Module,
    tokenizer: Any,
    samples: List[Any],
    device: str = "cuda",
) -> List[torch.Tensor]:
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


def cos_sim(g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between two gradient tensors.
    
    Adapted from GRADMM: gradmm/utilities.py:cos_sim
    """
    return (g1 * g2).sum() / (g1.norm(p=2) * g2.norm(p=2))


def grad_dist(
    target_grads: List[torch.Tensor],
    curr_grads: List[torch.Tensor],
    metric: str = "cos",
) -> torch.Tensor:
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
        self.device = str(config.get("device", "auto"))
        self.use_real_gradients = bool(config.get("use_real_gradients", False))
        
        # Model configuration (for loading LoRA model)
        self.model_name = config.get("model_name", "microsoft/phi-1_5")
        self.lora_checkpoint = config.get("lora_checkpoint")
        
        # Fallback feature encoder (if real gradients not available)
        self.feature_encoder = None
        if not self.use_real_gradients:
            from thesis_platform.models.features import build_feature_encoder
            self.feature_encoder = build_feature_encoder(
                config.get("feature_model"),
                repo_root,
                allow_fallback=bool(config.get("allow_hashing_fallback", False)),
                max_length=int(config.get("max_length", 256)),
                device=self.device,
            )
        
        self._model_cache: Dict[str, Any] = {}

    def _get_model_for_client(self, client_ctx: Any) -> Tuple[Any, Any, str]:
        """Get or load LoRA model for a client.
        
        TODO: Implement proper LoRA model loading
        For now, returns None to trigger fallback
        
        Re
