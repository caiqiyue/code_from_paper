from __future__ import annotations

from typing import Any


def _coverage_tuple(candidate: dict[str, Any]) -> tuple[float, float]:
    return (
        float(candidate["coverage_p25"]),
        float(candidate["coverage_mean"]),
    )


def _stability_score(candidate: dict[str, Any]) -> float:
    value = candidate.get("stability_score")
    if value is None:
        return float("inf")
    return float(value)


def select_uncertain_budget_by_policy_arbitration(
    *,
    broad_candidate: dict[str, Any],
    compact_candidate: dict[str, Any],
    uncertain_cfg: dict[str, Any],
) -> dict[str, Any]:
    coverage_epsilon = float(uncertain_cfg.get("coverage_epsilon", 0.002))
    support_epsilon = float(uncertain_cfg.get("support_epsilon", 0.002))
    stability_epsilon = float(uncertain_cfg.get("stability_epsilon", 0.002))
    prefer_smaller_budget_on_tie = bool(
        uncertain_cfg.get("prefer_smaller_budget_on_tie", True)
    )

    broad_cov = _coverage_tuple(broad_candidate)
    compact_cov = _coverage_tuple(compact_candidate)

    broad_wins_coverage = (
        broad_cov[0] > compact_cov[0] + coverage_epsilon
        or (
            abs(broad_cov[0] - compact_cov[0]) <= coverage_epsilon
            and broad_cov[1] > compact_cov[1] + coverage_epsilon
        )
    )
    compact_wins_coverage = (
        compact_cov[0] > broad_cov[0] + coverage_epsilon
        or (
            abs(compact_cov[0] - broad_cov[0]) <= coverage_epsilon
            and compact_cov[1] > broad_cov[1] + coverage_epsilon
        )
    )

    if broad_wins_coverage:
        selected = dict(broad_candidate)
        selected["arbitration_reason"] = "coverage"
        selected["arbitration_winner_policy"] = "broad_tail"
    elif compact_wins_coverage:
        selected = dict(compact_candidate)
        selected["arbitration_reason"] = "coverage"
        selected["arbitration_winner_policy"] = "compact_structured"
    else:
        broad_support = float(broad_candidate["support_mean"])
        compact_support = float(compact_candidate["support_mean"])
        if broad_support > compact_support + support_epsilon:
            selected = dict(broad_candidate)
            selected["arbitration_reason"] = "support"
            selected["arbitration_winner_policy"] = "broad_tail"
        elif compact_support > broad_support + support_epsilon:
            selected = dict(compact_candidate)
            selected["arbitration_reason"] = "support"
            selected["arbitration_winner_policy"] = "compact_structured"
        else:
            broad_stability = _stability_score(broad_candidate)
            compact_stability = _stability_score(compact_candidate)
            broad_has_stability = broad_candidate.get("stability_score") is not None
            compact_has_stability = compact_candidate.get("stability_score") is not None
            if broad_has_stability and compact_has_stability and (
                broad_stability + stability_epsilon < compact_stability
            ):
                selected = dict(broad_candidate)
                selected["arbitration_reason"] = "stability"
                selected["arbitration_winner_policy"] = "broad_tail"
            elif broad_has_stability and compact_has_stability and (
                compact_stability + stability_epsilon < broad_stability
            ):
                selected = dict(compact_candidate)
                selected["arbitration_reason"] = "stability"
                selected["arbitration_winner_policy"] = "compact_structured"
            else:
                broad_budget = int(broad_candidate["resolved_seed_top_k"])
                compact_budget = int(compact_candidate["resolved_seed_top_k"])
                if prefer_smaller_budget_on_tie and compact_budget <= broad_budget:
                    selected = dict(compact_candidate)
                    selected["arbitration_winner_policy"] = "compact_structured"
                else:
                    selected = dict(broad_candidate)
                    selected["arbitration_winner_policy"] = "broad_tail"
                selected["arbitration_reason"] = "compactness"

    selected["arbitration_triggered"] = True
    selected["selection_stage"] = "uncertainty_policy_arbitration"
    selected["arbitration_broad_budget"] = int(broad_candidate["resolved_seed_top_k"])
    selected["arbitration_compact_budget"] = int(
        compact_candidate["resolved_seed_top_k"]
    )
    selected["arbitration_broad_stability"] = broad_candidate.get("stability_score")
    selected["arbitration_compact_stability"] = compact_candidate.get(
        "stability_score"
    )
    return selected
