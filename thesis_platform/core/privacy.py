from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:
    from thesis_platform.core.dp_privacy import (
        DPConfig,
        DPPrivatizer,
        create_dp_config_from_dict,
    )
    DP_PRIVACY_AVAILABLE = True
except ImportError:
    DP_PRIVACY_AVAILABLE = False
    DPPrivatizer = None
    create_dp_config_from_dict = None


@dataclass(slots=True)
class PrivacyPolicy:
    """Runtime privacy accounting policy for one thesis-platform experiment."""

    enabled: bool
    mode: str
    epsilon: float | None
    delta: float | None
    sample_cost: float
    critique_cost: float
    upload_token_cost: float
    enforce_budget: bool = False
    # Real DP configuration (when mode includes "real_dp")
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.0
    dp_enabled: bool = False

    @classmethod
    def from_config(cls, privacy_cfg: dict[str, Any]) -> "PrivacyPolicy":
        """Normalize one privacy config block into a runtime policy."""

        enabled = bool(privacy_cfg.get("enabled", False))
        epsilon_raw = privacy_cfg.get("epsilon", 1.29)
        delta_raw = privacy_cfg.get("delta", 3e-6)
        epsilon = float(epsilon_raw) if epsilon_raw not in (None, "") else None
        delta = float(delta_raw) if delta_raw not in (None, "") else None
        sample_cost = float(privacy_cfg.get("sample_cost", 0.0))
        critique_cost = float(privacy_cfg.get("critique_cost", 0.0))
        upload_token_cost = float(privacy_cfg.get("upload_token_cost", 0.0))
        mode = str(privacy_cfg.get("mode", "disabled" if not enabled else "sample_critique_upload_proxy"))
        enforce_budget = bool(privacy_cfg.get("enforce_budget", False))
        max_grad_norm = float(privacy_cfg.get("max_grad_norm", 1.0))
        noise_multiplier = float(privacy_cfg.get("noise_multiplier", 1.0))
        # Check if real DP should be enabled
        dp_enabled = enabled and bool(
            privacy_cfg.get("enable_real_dp", privacy_cfg.get("dp_enabled", False))
        )

        return cls(
            enabled=enabled,
            mode=mode,
            epsilon=epsilon,
            delta=delta,
            sample_cost=sample_cost,
            critique_cost=critique_cost,
            upload_token_cost=upload_token_cost,
            enforce_budget=enforce_budget,
            max_grad_norm=max_grad_norm,
            noise_multiplier=noise_multiplier,
            dp_enabled=dp_enabled,
        )

    def validate(self) -> None:
        """Validate the privacy policy configuration."""

        if self.enabled:
            if self.epsilon is None or self.epsilon <= 0:
                raise ValueError("privacy.epsilon must be a positive number when privacy is enabled.")
            if self.delta is None or self.delta <= 0:
                raise ValueError("privacy.delta must be a positive number when privacy is enabled.")
        for field_name, value in {
            "privacy.sample_cost": self.sample_cost,
            "privacy.critique_cost": self.critique_cost,
            "privacy.upload_token_cost": self.upload_token_cost,
        }.items():
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative.")
        if self.dp_enabled:
            if self.max_grad_norm <= 0:
                raise ValueError("privacy.max_grad_norm must be positive when real DP is enabled.")
            if self.noise_multiplier <= 0:
                raise ValueError("privacy.noise_multiplier must be positive when real DP is enabled.")

    def snapshot(self) -> dict[str, Any]:
        """Return the stable public config snapshot for reports and manifests."""

        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "sample_cost": self.sample_cost,
            "critique_cost": self.critique_cost,
            "upload_token_cost": self.upload_token_cost,
            "enforce_budget": self.enforce_budget,
            "max_grad_norm": self.max_grad_norm,
            "noise_multiplier": self.noise_multiplier,
            "dp_enabled": self.dp_enabled,
        }

    def to_dp_config(self) -> Optional["DPConfig"]:
        """Convert to DPConfig for real DP operations."""
        if not DP_PRIVACY_AVAILABLE or not self.dp_enabled:
            return None
        return create_dp_config_from_dict({
            "enabled": self.dp_enabled,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "max_grad_norm": self.max_grad_norm,
            "noise_multiplier": self.noise_multiplier,
        })


@dataclass(slots=True)
class PrivacyLedger:
    """Accumulate proxy privacy spend across experiment rounds.

    When real DP is enabled via policy.dp_enabled, this ledger also manages
    gradient privatization using DPPrivatizer.
    """

    policy: PrivacyPolicy
    device: str = "cpu"
    entries: list[dict[str, Any]] = field(default_factory=list)
    cumulative_spent: float = 0.0
    _dp_privatizer: Optional["DPPrivatizer"] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize DP privatizer if real DP is enabled."""
        self._dp_privatizer = None
        if self.policy.dp_enabled and DP_PRIVACY_AVAILABLE:
            dp_config = self.policy.to_dp_config()
            if dp_config is not None:
                try:
                    device = str(self.device or "cpu")
                except Exception:
                    device = "cpu"
                self._dp_privatizer = DPPrivatizer(dp_config, device=device)

    def record_round(
        self,
        *,
        round_id: int,
        sample_count: int,
        critique_count: int,
        upload_token_count: int,
    ) -> dict[str, Any]:
        """Record one round-level privacy accounting event."""

        if not self.policy.enabled:
            summary = {
                "round_id": round_id,
                "privacy_enabled": False,
                "privacy_mode": "disabled",
                "epsilon_budget": self.policy.epsilon,
                "delta_budget": self.policy.delta,
                "privacy_spent": 0.0,
                "privacy_spent_cumulative": 0.0,
                "privacy_budget_left": None,
                "privacy_budget_exceeded": False,
                "privacy_event_counts": {
                    "sample_count": int(sample_count),
                    "critique_count": int(critique_count),
                    "upload_token_count": int(upload_token_count),
                },
                "privacy_spend_breakdown": {
                    "samples": 0.0,
                    "critiques": 0.0,
                    "upload_tokens": 0.0,
                },
            }
            self.entries.append(summary)
            return summary

        spend_breakdown = {
            "samples": round(float(sample_count) * self.policy.sample_cost, 12),
            "critiques": round(float(critique_count) * self.policy.critique_cost, 12),
            "upload_tokens": round(float(upload_token_count) * self.policy.upload_token_cost, 12),
        }
        round_spent = round(sum(spend_breakdown.values()), 12)
        self.cumulative_spent = round(self.cumulative_spent + round_spent, 12)
        epsilon_budget = float(self.policy.epsilon or 0.0)
        budget_left = round(max(epsilon_budget - self.cumulative_spent, 0.0), 12)
        exceeded = self.cumulative_spent > epsilon_budget if self.policy.epsilon is not None else False
        if exceeded and self.policy.enforce_budget:
            raise ValueError(
                f"Privacy budget exceeded after round {round_id}: "
                f"spent={self.cumulative_spent} epsilon={self.policy.epsilon}"
            )
        summary = {
            "round_id": round_id,
            "privacy_enabled": True,
            "privacy_mode": self.policy.mode,
            "epsilon_budget": self.policy.epsilon,
            "delta_budget": self.policy.delta,
            "privacy_spent": round_spent,
            "privacy_spent_cumulative": self.cumulative_spent,
            "privacy_budget_left": budget_left,
            "privacy_budget_exceeded": exceeded,
            "real_dp_enabled": self.policy.dp_enabled,
            "privacy_event_counts": {
                "sample_count": int(sample_count),
                "critique_count": int(critique_count),
                "upload_token_count": int(upload_token_count),
            },
            "privacy_spend_breakdown": spend_breakdown,
        }
        if self.policy.dp_enabled:
            real_status = self.get_privacy_budget_status()
            previous_real_total = 0.0
            if self.entries:
                previous_real_total = float(
                    self.entries[-1].get("real_dp_epsilon_spent", 0.0)
                )
            real_total = float(real_status.get("epsilon_spent", 0.0))
            summary.update(
                {
                    "real_dp_enabled": True,
                    "real_dp_epsilon_spent": real_total,
                    "real_dp_epsilon_spent_increment": round(
                        max(real_total - previous_real_total, 0.0), 12
                    ),
                    "real_dp_budget_left": real_status.get("budget_left"),
                    "real_dp_budget_exceeded": bool(
                        real_status.get("budget_exceeded", False)
                    ),
                    "real_dp_query_count": int(real_status.get("query_count", 0)),
                    "proxy_privacy_spent": round_spent,
                    "proxy_privacy_spent_cumulative": self.cumulative_spent,
                }
            )
        self.entries.append(summary)
        return summary

    def privatize_gradients(
        self,
        grad_dict: Dict[str, torch.Tensor],
        sample_rate: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Apply real DP privatization to gradients if enabled.

        Args:
            grad_dict: Dictionary of gradients to privatize
            sample_rate: Sampling rate for DP accountant

        Returns:
            Privatized gradients (unchanged if DP is disabled)
        """
        if self._dp_privatizer is None:
            return grad_dict
        return self._dp_privatizer.privatize_gradients(grad_dict, sample_rate)

    def privatize_vector(
        self,
        vector: torch.Tensor,
        clip_norm: Optional[float] = None,
    ) -> torch.Tensor:
        """Apply real DP privatization to a vector if enabled.

        Args:
            vector: Vector to privatize
            clip_norm: Optional clip norm override

        Returns:
            Privatized vector (unchanged if DP is disabled)
        """
        if self._dp_privatizer is None:
            return vector
        return self._dp_privatizer.privatize_vector(vector, clip_norm)

    def privatize_scores(
        self,
        scores: list[float],
        clip_bound: Optional[float] = None,
    ) -> list[float]:
        """Apply real DP privatization to a list of scalar scores if enabled.

        This is used for influence scores and other numeric signals that leak
        information about client private data when sent to the server.

        Algorithm:
        1. Clip each score to [-clip_bound, clip_bound]
        2. Add Gaussian noise with sigma = noise_multiplier * clip_bound

        Args:
            scores: List of scalar scores to privatize
            clip_bound: Clip bound (defaults to max_grad_norm from policy)

        Returns:
            Privatized scores (unchanged if DP is disabled)
        """
        if self._dp_privatizer is None:
            return scores
        return self._dp_privatizer.privatize_scores(scores, clip_bound=clip_bound)

    def privatize_scalar(
        self,
        value: float,
        clip_bound: Optional[float] = None,
    ) -> float:
        """Apply real DP privatization to a single scalar value if enabled.

        Args:
            value: Scalar value to privatize
            clip_bound: Clip bound (defaults to max_grad_norm from policy)

        Returns:
            Privatized scalar (unchanged if DP is disabled)
        """
        if self._dp_privatizer is None:
            return value
        return self._dp_privatizer.privatize_scalar(value, clip_bound=clip_bound)

    def get_privacy_budget_status(self) -> dict[str, Any]:
        """Get current privacy budget status.

        Returns:
            Dictionary with privacy budget information
        """
        if self._dp_privatizer is None:
            proxy_budget_left = (
                self.policy.epsilon - self.cumulative_spent
                if self.policy.enabled and self.policy.epsilon
                else None
            )
            return {
                "enabled": self.policy.enabled,
                "real_dp_enabled": False,
                "epsilon_budget": self.policy.epsilon,
                "epsilon_spent": self.cumulative_spent if self.policy.enabled else 0.0,
                "delta": self.policy.delta,
                "budget_left": proxy_budget_left,
                "budget_exceeded": self.cumulative_spent > self.policy.epsilon if self.policy.enabled and self.policy.epsilon else False,
                "proxy_epsilon_spent": self.cumulative_spent if self.policy.enabled else 0.0,
                "proxy_budget_left": proxy_budget_left,
                "proxy_budget_exceeded": self.cumulative_spent > self.policy.epsilon if self.policy.enabled and self.policy.epsilon else False,
            }
        proxy_budget_left = (
            self.policy.epsilon - self.cumulative_spent
            if self.policy.enabled and self.policy.epsilon
            else None
        )
        real_status = dict(self._dp_privatizer.get_privacy_budget_status())
        real_status.update(
            {
                "enabled": self.policy.enabled,
                "real_dp_enabled": True,
                "proxy_epsilon_spent": self.cumulative_spent,
                "proxy_budget_left": proxy_budget_left,
                "proxy_budget_exceeded": self.cumulative_spent > self.policy.epsilon
                if self.policy.enabled and self.policy.epsilon
                else False,
            }
        )
        return real_status

    def summary(self) -> dict[str, Any]:
        """Return the experiment-level privacy summary."""

        latest = self.entries[-1] if self.entries else {}
        budget_status = self.get_privacy_budget_status()
        spent_total = (
            float(budget_status.get("epsilon_spent", 0.0))
            if self.policy.dp_enabled
            else self.cumulative_spent if self.policy.enabled else 0.0
        )
        return {
            "enabled": self.policy.enabled,
            "mode": self.policy.mode if self.policy.enabled else "disabled",
            "epsilon": self.policy.epsilon,
            "delta": self.policy.delta,
            "spent_total": spent_total,
            "proxy_spent_total": self.cumulative_spent if self.policy.enabled else 0.0,
            "budget_left": budget_status.get("budget_left"),
            "budget_exceeded": bool(budget_status.get("budget_exceeded", False)),
            "real_dp_enabled": self.policy.dp_enabled,
            "real_dp_epsilon_spent": budget_status.get("epsilon_spent")
            if self.policy.dp_enabled
            else None,
            "real_dp_query_count": budget_status.get("query_count")
            if self.policy.dp_enabled
            else None,
            "round_count": len(self.entries),
            "last_round_proxy_budget_left": latest.get("privacy_budget_left"),
        }

    def report(self) -> dict[str, Any]:
        """Return the full serializable ledger payload."""

        return {
            "policy": self.policy.snapshot(),
            "summary": self.summary(),
            "cumulative_spent": self.cumulative_spent,
            "device": self.device,
            "dp_runtime_state": self._dp_privatizer.export_state()
            if self._dp_privatizer is not None
            else None,
            "entries": list(self.entries),
        }

    @classmethod
    def restore_from_report(cls, report_data: dict[str, Any]) -> "PrivacyLedger":
        """Restore a PrivacyLedger from a checkpoint report.

        Args:
            report_data: The data previously returned by report()

        Returns:
            A new PrivacyLedger with restored state
        """
        policy = PrivacyPolicy.from_config(report_data["policy"])
        ledger = cls(policy=policy, device=str(report_data.get("device", "cpu")))
        ledger.cumulative_spent = report_data.get("cumulative_spent", 0.0)
        ledger.entries = list(report_data.get("entries", []))
        if ledger._dp_privatizer is not None:
            dp_runtime_state = report_data.get("dp_runtime_state")
            if dp_runtime_state:
                ledger._dp_privatizer.restore_state(dp_runtime_state)
            elif ledger.entries:
                legacy_query_count = int(
                    ledger.entries[-1].get(
                        "real_dp_query_count",
                        report_data.get("summary", {}).get("real_dp_query_count", 0),
                    )
                )
                if legacy_query_count > 0:
                    ledger._dp_privatizer.restore_state(
                        {
                            "enabled": True,
                            "accountant": {
                                "query_counter": legacy_query_count,
                                "steps": legacy_query_count,
                            },
                        }
                    )
        return ledger
