"""LoRA gradient computation module for real gradient-based scorers.

Adapted from DataInf and GRADMM source code.
Key insight: Only extract gradients from LoRA adapter parameters (lora_A, lora_B),
not from the full model parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

try:
    from peft import PeftModel, LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoModelForSequenceClassification,
    )

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning(
        "peft or transformers not available. LoRA features will be disabled."
    )

logger = logging.getLogger(__name__)


class LoRAGradientExtractor:
    """Extract per-sample gradients from LoRA-tuned models.

    This class handles:
    1. Loading base models with LoRA adapters
    2. Computing per-sample gradients for LoRA parameters only
    3. Aggregating gradients across layers

    Adapted from:
    - DataInf: src/lora_model.py (LORAEngine, LORAEngineGeneration)
    - GRADMM: gradmm/utilities.py (compute_grads_lm)
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        lora_rank: int = 8,
        lora_alpha: int = 8,
        target_modules: Optional[List[str]] = None,
    ):
        """Initialize the LoRA gradient extractor.

        Args:
            model_name_or_path: Path to the base model or HuggingFace model name
            device: Device to run the model on
            lora_rank: LoRA rank (r parameter)
            lora_alpha: LoRA alpha parameter
            target_modules: List of module names to apply LoRA to (e.g., ["q_proj", "v_proj"])
        """
        self.model_name_or_path = model_name_or_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.model = None
        self.tokenizer = None

        if not PEFT_AVAILABLE:
            raise ImportError("peft library is required for LoRA gradient extraction")

    def load_model(
        self,
        lora_adapter_path: Optional[str] = None,
        task_type: str = "CAUSAL_LM",
    ) -> None:
        """Load the base model and optionally a LoRA adapter.

        Args:
            lora_adapter_path: Path to the LoRA adapter checkpoint. If None, will initialize new LoRA.
            task_type: Task type for the model (CAUSAL_LM or SEQ_CLS)
        """
        logger.info(f"Loading base model from {self.model_name_or_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load base model
        if task_type == "CAUSAL_LM":
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )
        else:
            base_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )

        if lora_adapter_path:
            # Load pre-trained LoRA adapter
            logger.info(f"Loading LoRA adapter from {lora_adapter_path}")
            self.model = PeftModel.from_pretrained(
                base_model, lora_adapter_path, is_trainable=True
            )
        else:
            # Initialize new LoRA adapter
            logger.info(f"Initializing new LoRA adapter with rank={self.lora_rank}")
            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules=self.target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type=task_type,
            )
            self.model = get_peft_model(base_model, lora_config)
            self.model.print_trainable_parameters()

        self.model.eval()
        logger.info("Model loaded successfully")

    def compute_sample_gradients(
        self,
        text: str,
        max_length: int = 512,
        return_only_lora: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Compute gradients for a single text sample.

        Args:
            text: Input text
            max_length: Maximum sequence length
            return_only_lora: If True, only return LoRA parameter gradients

        Returns:
            Dictionary mapping parameter names to gradient tensors
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        ).to(self.device)

        # Forward pass
        self.model.zero_grad()
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Extract gradients
        grad_dict = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if return_only_lora:
                    # Only extract LoRA parameter gradients (DataInf approach)
                    if "lora_A" in name or "lora_B" in name:
                        grad_dict[name] = param.grad.cpu().clone()
                else:
                    # Extract all trainable parameter gradients (GRADMM approach)
                    if param.requires_grad:
                        grad_dict[name] = param.grad.cpu().clone()

        return grad_dict

    def compute_batch_gradients(
        self,
        texts: List[str],
        max_length: int = 512,
        return_only_lora: bool = True,
    ) -> List[Dict[str, torch.Tensor]]:
        """Compute gradients for a batch of text samples.

        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            return_only_lora: If True, only return LoRA parameter gradients

        Returns:
            List of gradient dictionaries, one per sample
        """
        grad_list = []
        for text in texts:
            grad_dict = self.compute_sample_gradients(
                text, max_length, return_only_lora
            )
            grad_list.append(grad_dict)
        return grad_list

    def compute_average_gradients(
        self,
        texts: List[str],
        max_length: int = 512,
        return_only_lora: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Compute average gradients across multiple samples.

        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            return_only_lora: If True, only return LoRA parameter gradients

        Returns:
            Dictionary mapping parameter names to averaged gradient tensors
        """
        grad_list = self.compute_batch_gradients(texts, max_length, return_only_lora)

        if not grad_list:
            return {}

        # Average gradients
        avg_grads = {}
        n_samples = len(grad_list)

        for name in grad_list[0].keys():
            avg_grads[name] = (
                sum(grad_dict[name] for grad_dict in grad_list) / n_samples
            )

        return avg_grads

    def release(self):
        """Release model resources."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()
        logger.info("Model resources released")


class GradientDistanceCalculator:
    """Calculate gradient distances between target and current gradients.

    Adapted from:
    - DataInf: src/influence.py (compute_hvp_proposed)
    - GRADMM: gradmm/utilities.py (grad_dist, cos_sim)
    """

    @staticmethod
    def cosine_distance(
        target_grads: Dict[str, torch.Tensor],
        current_grads: Dict[str, torch.Tensor],
    ) -> float:
        """Compute cosine distance between two gradient dictionaries.

        Args:
            target_grads: Target gradients (e.g., from real samples)
            current_grads: Current gradients (e.g., from synthetic samples)

        Returns:
            Cosine distance (1 - cosine_similarity)
        """
        total_sim = 0.0
        n_params = 0

        for name in target_grads:
            if name in current_grads:
                g1 = target_grads[name].reshape(-1)
                g2 = current_grads[name].reshape(-1)

                # Cosine similarity
                sim = torch.sum(g1 * g2) / (torch.norm(g1) * torch.norm(g2) + 1e-8)
                total_sim += sim.item()
                n_params += 1

        if n_params == 0:
            return 1.0

        avg_sim = total_sim / n_params
        return 1.0 - avg_sim  # Cosine distance

    @staticmethod
    def euclidean_distance(
        target_grads: Dict[str, torch.Tensor],
        current_grads: Dict[str, torch.Tensor],
    ) -> float:
        """Compute Euclidean distance between two gradient dictionaries.

        Args:
            target_grads: Target gradients
            current_grads: Current gradients

        Returns:
            Euclidean distance
        """
        total_dist = 0.0
        n_params = 0

        for name in target_grads:
            if name in current_grads:
                g1 = target_grads[name].reshape(-1)
                g2 = current_grads[name].reshape(-1)

                dist = torch.sum((g1 - g2) ** 2)
                total_dist += dist.item()
                n_params += 1

        if n_params == 0:
            return float("inf")

        return (total_dist / n_params) ** 0.5

    @staticmethod
    def gradient_mismatch_score(
        real_grads: Dict[str, torch.Tensor],
        syn_grads: Dict[str, torch.Tensor],
        metric: str = "cosine",
    ) -> float:
        """Compute gradient mismatch score.

        Args:
            real_grads: Gradients from real samples
            syn_grads: Gradients from synthetic samples
            metric: Distance metric ("cosine", "euclidean", "l1")

        Returns:
            Mismatch score (higher = more different)
        """
        if metric == "cosine":
            return GradientDistanceCalculator.cosine_distance(real_grads, syn_grads)
        elif metric == "euclidean":
            return GradientDistanceCalculator.euclidean_distance(real_grads, syn_grads)
        elif metric == "l1":
            return GradientDistanceCalculator.l1_distance(real_grads, syn_grads)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    @staticmethod
    def l1_distance(
        target_grads: Dict[str, torch.Tensor],
        current_grads: Dict[str, torch.Tensor],
    ) -> float:
        """Compute L1 distance between two gradient dictionaries."""
        total_dist = 0.0
        n_params = 0

        for name in target_grads:
            if name in current_grads:
                g1 = target_grads[name].reshape(-1)
                g2 = current_grads[name].reshape(-1)

                dist = torch.sum(torch.abs(g1 - g2))
                total_dist += dist.item()
                n_params += 1

        if n_params == 0:
            return float("inf")

        return total_dist / n_params


# Utility functions for gradient manipulation
def flatten_gradients(grad_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten all gradients into a single vector."""
    return torch.cat([g.reshape(-1) for g in grad_dict.values()])


def gradient_norm(grad_dict: Dict[str, torch.Tensor]) -> float:
    """Compute the L2 norm of all gradients."""
    flat_grad = flatten_gradients(grad_dict)
    return torch.norm(flat_grad).item()


def clip_gradients(
    grad_dict: Dict[str, torch.Tensor],
    max_norm: float,
) -> Dict[str, torch.Tensor]:
    """Clip gradients by norm.

    Args:
        grad_dict: Gradient dictionary
        max_norm: Maximum allowed norm

    Returns:
        Clipped gradient dictionary
    """
    norm = gradient_norm(grad_dict)
    if norm > max_norm:
        scale = max_norm / norm
        return {k: v * scale for k, v in grad_dict.items()}
    return grad_dict


def add_noise_to_gradients(
    grad_dict: Dict[str, torch.Tensor],
    noise_multiplier: float,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Add Gaussian noise to gradients (for DP).

    Args:
        grad_dict: Gradient dictionary
        noise_multiplier: Noise scale
        device: Device for noise generation

    Returns:
        Noisy gradient dictionary
    """
    noisy_grads = {}
    for name, grad in grad_dict.items():
        noise = torch.randn_like(grad, device=device) * noise_multiplier
        noisy_grads[name] = grad + noise
    return noisy_grads
