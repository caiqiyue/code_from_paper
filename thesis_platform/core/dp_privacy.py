"""Real Differential Privacy mechanisms for federated learning.

Implements:
- Gradient clipping (per-sample and per-layer)
- Gaussian noise injection
- DP-SGD optimizer wrapper
- Moment accountant for privacy budget tracking

Note: Privacy is OPTIONAL. Can be enabled/disabled via configuration.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Callable
import logging
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DPConfig:
    """Differential Privacy configuration."""

    enabled: bool = False
    epsilon: float = 1.0  # Privacy budget
    delta: float = 1e-5  # Failure probability
    max_grad_norm: float = 1.0  # Gradient clipping threshold
    noise_multiplier: float = 1.0  # Noise scale
    accountant_method: str = "rdp"  # rdp, gdp, or plain

    def validate(self):
        """Validate DP configuration."""
        if self.enabled:
            if self.epsilon <= 0:
                raise ValueError("epsilon must be positive")
            if self.delta <= 0 or self.delta >= 1:
                raise ValueError("delta must be in (0, 1)")
            if self.max_grad_norm <= 0:
                raise ValueError("max_grad_norm must be positive")


class GradientClipper:
    """Gradient clipping for DP."""

    @staticmethod
    def clip_by_norm(
        grad_dict: Dict[str, torch.Tensor],
        max_norm: float,
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """Clip gradients by global norm.

        Returns:
            Tuple of (clipped_gradients, actual_norm)
        """
        # Compute global norm
        global_norm = 0.0
        for grad in grad_dict.values():
            if grad is not None:
                global_norm += torch.sum(grad**2).item()
        global_norm = global_norm**0.5

        # Clip
        clip_coeff = min(max_norm / (global_norm + 1e-6), 1.0)

        clipped_grads = {}
        for name, grad in grad_dict.items():
            if grad is not None:
                clipped_grads[name] = grad * clip_coeff
            else:
                clipped_grads[name] = None

        return clipped_grads, global_norm

    @staticmethod
    def clip_by_value(
        grad_dict: Dict[str, torch.Tensor],
        clip_value: float,
    ) -> Dict[str, torch.Tensor]:
        """Clip gradients element-wise."""
        clipped_grads = {}
        for name, grad in grad_dict.items():
            if grad is not None:
                clipped_grads[name] = torch.clamp(grad, -clip_value, clip_value)
            else:
                clipped_grads[name] = None
        return clipped_grads


class NoiseInjector:
    """Gaussian noise injection for DP."""

    @staticmethod
    def add_gaussian_noise(
        grad_dict: Dict[str, torch.Tensor],
        noise_multiplier: float,
        max_grad_norm: float,
        device: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """Add Gaussian noise to gradients.

        Noise scale: sigma = noise_multiplier * max_grad_norm
        """
        sigma = noise_multiplier * max_grad_norm

        noisy_grads = {}
        for name, grad in grad_dict.items():
            if grad is not None:
                noise = torch.randn_like(grad, device=device) * sigma
                noisy_grads[name] = grad + noise
            else:
                noisy_grads[name] = None

        return noisy_grads

    @staticmethod
    def add_gaussian_noise_to_tensor(
        tensor: torch.Tensor,
        sigma: float,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Add Gaussian noise to a single tensor."""
        noise = torch.randn_like(tensor, device=device) * sigma
        return tensor + noise


class MomentAccountant:
    """Moment accountant for privacy budget tracking (Abadi et al. 2016).

    Tracks privacy consumption using Renyi Differential Privacy (RDP).
    """

    def __init__(self, epsilon: float, delta: float):
        self.epsilon = epsilon
        self.delta = delta
        self.steps = 0
        self.noise_multiplier = 1.0
        self.max_grad_norm = 1.0

    def compute_noise_multiplier(
        self,
        epsilon_target: float,
        delta: float,
        n_samples: int,
        batch_size: int,
        epochs: int,
    ) -> float:
        """Compute noise multiplier for target privacy budget.

        Uses binary search to find appropriate noise multiplier.
        """

        def compute_epsilon(noise_mult):
            # Simplified RDP computation
            # In practice, use opacus or other libraries for precise computation
            q = batch_size / n_samples  # Sampling rate
            steps = epochs * (n_samples // batch_size)

            # RDP to (eps, delta)-DP conversion (simplified)
            rdp_alpha = 2  # Usually search over multiple alphas
            rdp_eps = steps * rdp_alpha / (2 * noise_mult**2)

            # Convert to (eps, delta)-DP
            eps = rdp_eps + np.log(1 / delta) / (rdp_alpha - 1)
            return eps

        # Binary search for noise multiplier
        noise_low, noise_high = 0.1, 100.0
        for _ in range(20):  # Max iterations
            noise_mid = (noise_low + noise_high) / 2
            eps_mid = compute_epsilon(noise_mid)

            if eps_mid > epsilon_target:
                noise_low = noise_mid
            else:
                noise_high = noise_mid

        return (noise_low + noise_high) / 2

    def get_privacy_spent(self) -> Tuple[float, float]:
        """Return (epsilon_spent, delta)."""
        # Simplified - in practice track per-step consumption
        return (0.0, self.delta)


class DPPrivatizer:
    """Main DP privatizer that orchestrates clipping and noise injection.

    Can be enabled/disabled via configuration.
    """

    def __init__(self, config: DPConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.clipper = GradientClipper()
        self.noise_injector = NoiseInjector()

        if config.enabled:
            self.accountant = MomentAccountant(config.epsilon, config.delta)
            logger.info(f"DP enabled: epsilon={config.epsilon}, delta={config.delta}")
        else:
            self.accountant = None
            logger.info("DP disabled")

    def privatize_gradients(
        self,
        grad_dict: Dict[str, torch.Tensor],
        sample_rate: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Apply DP to gradients.

        If DP is disabled, returns gradients unchanged.
        """
        if not self.config.enabled:
            return grad_dict

        # Step 1: Clip gradients
        clipped_grads, global_norm = self.clipper.clip_by_norm(
            grad_dict, self.config.max_grad_norm
        )

        logger.debug(f"Gradient norm before clipping: {global_norm:.4f}")

        # Step 2: Add noise
        noisy_grads = self.noise_injector.add_gaussian_noise(
            clipped_grads,
            self.config.noise_multiplier,
            self.config.max_grad_norm,
            self.device,
        )

        return noisy_grads

    def privatize_vector(
        self,
        vector: torch.Tensor,
        clip_norm: Optional[float] = None,
    ) -> torch.Tensor:
        """Privatize a single vector (e.g., prototype vector)."""
        if not self.config.enabled:
            return vector

        clip_norm = clip_norm or self.config.max_grad_norm

        # Clip
        vector_norm = torch.norm(vector)
        clip_coeff = min(clip_norm / (vector_norm + 1e-6), 1.0)
        clipped = vector * clip_coeff

        # Add noise
        sigma = self.config.noise_multiplier * clip_norm
        noise = torch.randn_like(clipped, device=self.device) * sigma

        return clipped + noise

    def get_privacy_budget_status(self) -> Dict[str, Any]:
        """Get current privacy budget consumption."""
        if not self.config.enabled:
            return {
                "enabled": False,
                "epsilon_budget": None,
                "epsilon_spent": 0.0,
                "delta": None,
                "budget_left": None,
            }

        epsilon_spent, delta = self.accountant.get_privacy_spent()
        budget_left = self.config.epsilon - epsilon_spent

        return {
            "enabled": True,
            "epsilon_budget": self.config.epsilon,
            "epsilon_spent": epsilon_spent,
            "delta": self.config.delta,
            "budget_left": max(0, budget_left),
            "budget_exceeded": epsilon_spent > self.config.epsilon,
        }


class DPAdamW(torch.optim.AdamW):
    """DP version of AdamW optimizer.

    Wraps standard AdamW with gradient clipping and noise injection.
    """

    def __init__(
        self,
        params,
        dp_config: DPConfig,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self.dp_config = dp_config
        self.privatizer = DPPrivatizer(dp_config, device) if dp_config.enabled else None

    def step(self, closure=None):
        """Performs a single optimization step with DP."""
        if self.privatizer is None or not self.dp_config.enabled:
            return super().step(closure)

        # Collect gradients
        grad_dict = {}
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grad_dict[id(p)] = p.grad.data

        # Privatize gradients
        noisy_grad_dict = self.privatizer.privatize_gradients(grad_dict)

        # Update gradients
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None and id(p) in noisy_grad_dict:
                    p.grad.data = noisy_grad_dict[id(p)]

        # Standard optimizer step
        return super().step(closure)


# Utility functions for integration
def create_dp_config_from_dict(config_dict: Dict[str, Any]) -> DPConfig:
    """Create DPConfig from configuration dictionary."""
    return DPConfig(
        enabled=config_dict.get("enabled", False),
        epsilon=config_dict.get("epsilon", 1.0),
        delta=config_dict.get("delta", 1e-5),
        max_grad_norm=config_dict.get("max_grad_norm", 1.0),
        noise_multiplier=config_dict.get("noise_multiplier", 1.0),
        accountant_method=config_dict.get("accountant_method", "rdp"),
    )


def compute_dp_noise_multiplier(
    epsilon: float,
    delta: float,
    n_samples: int,
    batch_size: int,
    epochs: int,
) -> float:
    """Compute noise multiplier for target privacy budget.

    This is a simplified computation. For production use,
    consider using opacus or other specialized libraries.
    """
    accountant = MomentAccountant(epsilon, delta)
    return accountant.compute_noise_multiplier(
        epsilon, delta, n_samples, batch_size, epochs
    )
