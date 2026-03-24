from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
        return cls(
            enabled=enabled,
            mode=mode,
            epsilon=epsilon,
            delta=delta,
            sample_cost=sample_cost,
            critique_cost=critique_cost,
            upload_token_cost=upload_token_cost,
            enforce_budget=enforce_budget,
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
        }


@dataclass(slots=True)
class PrivacyLedger:
    """Accumulate proxy privacy spend across experiment rounds."""

    policy: PrivacyPolicy
    entries: list[dict[str, Any]] = field(default_factory=list)
    cumulative_spent: float = 0.0

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
            "privacy_event_counts": {
                "sample_count": int(sample_count),
                "critique_count": int(critique_count),
                "upload_token_count": int(upload_token_count),
            },
            "privacy_spend_breakdown": spend_breakdown,
        }
        self.entries.append(summary)
        return summary

    def summary(self) -> dict[str, Any]:
        """Return the experiment-level privacy summary."""

        latest = self.entries[-1] if self.entries else {}
        return {
            "enabled": self.policy.enabled,
            "mode": self.policy.mode if self.policy.enabled else "disabled",
            "epsilon": self.policy.epsilon,
            "delta": self.policy.delta,
            "spent_total": self.cumulative_spent if self.policy.enabled else 0.0,
            "budget_left": latest.get("privacy_budget_left"),
            "budget_exceeded": bool(latest.get("privacy_budget_exceeded", False)),
            "round_count": len(self.entries),
        }

    def report(self) -> dict[str, Any]:
        """Return the full serializable ledger payload."""

        return {
            "policy": self.policy.snapshot(),
            "summary": self.summary(),
            "entries": list(self.entries),
        }
