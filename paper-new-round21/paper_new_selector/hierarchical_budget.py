from __future__ import annotations

from typing import Any

from .regime_router import route_budget_regime
from .shape_descriptor import compute_shape_descriptor


def _filter_budget_metrics(
    metrics_by_budget: dict[int, dict[str, Any]],
    candidate_seed_top_k: list[int],
) -> dict[int, dict[str, Any]]:
    return {
        int(budget): dict(metrics_by_budget[int(budget)])
        for budget in candidate_seed_top_k
        if int(budget) in metrics_by_budget
    }


def _select_broad_tail_budget(
    metrics_by_budget: dict[int, dict[str, Any]],
    policy_cfg: dict[str, Any],
) -> dict[str, Any]:
    if not metrics_by_budget:
        return {"selection_stage": "broad_tail_policy", "feasible_budgets": [], "resolved_seed_top_k": None}
    best_coverage_p25 = max(float(metrics["coverage_p25"]) for metrics in metrics_by_budget.values())
    best_coverage_mean = max(float(metrics["coverage_mean"]) for metrics in metrics_by_budget.values())
    feasible = [
        int(budget)
        for budget, metrics in sorted(metrics_by_budget.items())
        if float(metrics["coverage_p25"]) >= float(policy_cfg["coverage_p25_ratio"]) * best_coverage_p25
        and float(metrics["coverage_mean"]) >= float(policy_cfg["coverage_mean_ratio"]) * best_coverage_mean
    ]
    if not feasible:
        return {"selection_stage": "broad_tail_policy", "feasible_budgets": [], "resolved_seed_top_k": None}
    epsilon = float(policy_cfg.get("epsilon", 0.0))
    ranked = sorted(
        feasible,
        key=lambda budget: (
            float(metrics_by_budget[budget]["coverage_p25"]),
            float(metrics_by_budget[budget]["coverage_mean"]),
            float(metrics_by_budget[budget]["support_mean"]),
            int(budget),
        ),
        reverse=True,
    )
    best_budget = int(ranked[0])
    if len(ranked) > 1:
        runner_up_budget = int(ranked[1])
        best_metrics = metrics_by_budget[best_budget]
        runner_up_metrics = metrics_by_budget[runner_up_budget]
        if (
            abs(
                float(best_metrics["coverage_p25"])
                - float(runner_up_metrics["coverage_p25"])
            )
            <= epsilon
            and abs(
                float(best_metrics["coverage_mean"])
                - float(runner_up_metrics["coverage_mean"])
            )
            <= epsilon
        ):
            best_budget = max(best_budget, runner_up_budget)
    return {
        "selection_stage": "broad_tail_policy",
        "feasible_budgets": feasible,
        "resolved_seed_top_k": int(best_budget),
    }


def _select_compact_budget(
    metrics_by_budget: dict[int, dict[str, Any]],
    policy_cfg: dict[str, Any],
) -> dict[str, Any]:
    from .budget_calibration import combine_feasible_budget_metrics

    if not metrics_by_budget:
        return {"selection_stage": "compact_structured_policy", "feasible_budgets": [], "resolved_seed_top_k": None}
    best_coverage_p25 = max(float(metrics["coverage_p25"]) for metrics in metrics_by_budget.values())
    feasible = [
        int(budget)
        for budget, metrics in sorted(metrics_by_budget.items())
        if float(metrics["coverage_p25"]) >= float(policy_cfg["coverage_p25_ratio"]) * best_coverage_p25
    ]
    if not feasible:
        return {"selection_stage": "compact_structured_policy", "feasible_budgets": [], "resolved_seed_top_k": None}
    epsilon = float(policy_cfg.get("epsilon", 0.0))
    enriched = combine_feasible_budget_metrics(
        metrics_by_budget=metrics_by_budget,
        feasible_budgets=feasible,
        calibration_cfg={"utility": dict(policy_cfg["utility"])},
    )
    ranked = sorted(
        feasible,
        key=lambda budget: (
            float(enriched[budget]["feasible_utility"]),
            -int(budget),
        ),
        reverse=True,
    )
    best_budget = int(ranked[0])
    if len(ranked) > 1:
        runner_up_budget = int(ranked[1])
        utility_gap = float(
            enriched[best_budget]["feasible_utility"]
            - enriched[runner_up_budget]["feasible_utility"]
        )
        if utility_gap <= epsilon:
            best_budget = min(best_budget, runner_up_budget)
    return {
        "selection_stage": "compact_structured_policy",
        "feasible_budgets": feasible,
        "resolved_seed_top_k": int(best_budget),
    }


def _compute_local_budget_stability(
    metrics_by_budget: dict[int, dict[str, Any]],
    budget: int,
) -> float | None:
    ordered_budgets = sorted(int(key) for key in metrics_by_budget)
    if budget not in ordered_budgets:
        return None
    index = ordered_budgets.index(int(budget))
    neighbor_budgets: list[int] = []
    if index > 0:
        neighbor_budgets.append(ordered_budgets[index - 1])
    if index + 1 < len(ordered_budgets):
        neighbor_budgets.append(ordered_budgets[index + 1])
    if not neighbor_budgets:
        return None

    anchor = metrics_by_budget[int(budget)]
    diffs: list[float] = []
    for neighbor_budget in neighbor_budgets:
        neighbor = metrics_by_budget[int(neighbor_budget)]
        diffs.extend(
            [
                abs(float(anchor["coverage_p25"]) - float(neighbor["coverage_p25"])),
                abs(float(anchor["coverage_mean"]) - float(neighbor["coverage_mean"])),
                abs(float(anchor["support_mean"]) - float(neighbor["support_mean"])),
            ]
        )
    return sum(diffs) / float(len(diffs))


def resolve_hierarchical_budget(
    *,
    private_lengths: list[int],
    metrics_by_budget: dict[int, dict[str, Any]],
    rule_cfg: dict[str, Any],
) -> dict[str, Any]:
    router_cfg = dict(rule_cfg["router"])
    descriptor = compute_shape_descriptor(
        private_lengths,
        tail_threshold=int(router_cfg["tail_threshold"]),
        short_threshold=int(router_cfg["short_threshold"]),
    )
    route = route_budget_regime(descriptor, router_cfg)
    policies_cfg = dict(rule_cfg["policies"])
    if route.regime == "broad_tail":
        subset = _filter_budget_metrics(
            metrics_by_budget,
            list(policies_cfg["broad_tail"]["candidate_seed_top_k"]),
        )
        selected = _select_broad_tail_budget(subset, dict(policies_cfg["broad_tail"]))
    elif route.regime == "compact_structured":
        subset = _filter_budget_metrics(
            metrics_by_budget,
            list(policies_cfg["compact_structured"]["candidate_seed_top_k"]),
        )
        selected = _select_compact_budget(subset, dict(policies_cfg["compact_structured"]))
    else:
        uncertain_cfg = dict(policies_cfg["uncertain"])
        uncertain_mode = str(uncertain_cfg.get("mode", "fallback"))
        broad_subset = _filter_budget_metrics(
            metrics_by_budget,
            list(policies_cfg["broad_tail"]["candidate_seed_top_k"]),
        )
        compact_subset = _filter_budget_metrics(
            metrics_by_budget,
            list(policies_cfg["compact_structured"]["candidate_seed_top_k"]),
        )
        broad_selected = _select_broad_tail_budget(
            broad_subset,
            dict(policies_cfg["broad_tail"]),
        )
        compact_selected = _select_compact_budget(
            compact_subset,
            dict(policies_cfg["compact_structured"]),
        )

        if uncertain_mode == "policy_arbitration" and bool(
            uncertain_cfg.get("arbitration_enabled", True)
        ):
            from .uncertainty_arbitration import (
                select_uncertain_budget_by_policy_arbitration,
            )

            broad_budget = broad_selected.get("resolved_seed_top_k")
            compact_budget = compact_selected.get("resolved_seed_top_k")
            if broad_budget is not None and compact_budget is not None:
                broad_candidate = dict(broad_subset[int(broad_budget)])
                broad_candidate.update(broad_selected)
                broad_candidate["policy_name"] = "broad_tail"
                broad_candidate["stability_score"] = _compute_local_budget_stability(
                    broad_subset,
                    int(broad_budget),
                )

                compact_candidate = dict(compact_subset[int(compact_budget)])
                compact_candidate.update(compact_selected)
                compact_candidate["policy_name"] = "compact_structured"
                compact_candidate["stability_score"] = _compute_local_budget_stability(
                    compact_subset,
                    int(compact_budget),
                )

                selected = select_uncertain_budget_by_policy_arbitration(
                    broad_candidate=broad_candidate,
                    compact_candidate=compact_candidate,
                    uncertain_cfg=uncertain_cfg,
                )
            else:
                selected = {
                    "resolved_seed_top_k": None,
                    "selection_stage": "uncertainty_policy_arbitration",
                    "arbitration_triggered": False,
                    "policy_fallback_used": True,
                    "policy_fallback_reason": "missing_policy_candidate",
                    "feasible_budgets": [],
                }
        elif uncertain_mode in {"policy_arbitration", "fallback"}:
            from .budget_calibration import (
                select_budget_by_constrained_utility,
                select_budget_with_recheck,
            )

            compact_utility = dict(policies_cfg["compact_structured"]["utility"])
            fallback_mode = str(
                uncertain_cfg.get("fallback_mode", "self_calibrated_constrained")
            )
            if fallback_mode == "self_calibrated":
                selected = select_budget_with_recheck(
                    metrics_by_budget=metrics_by_budget,
                    calibration_cfg={
                        "utility": compact_utility,
                        "tiebreak": {"epsilon": 0.01, "prefer_smaller_budget": True},
                        "near_boundary_recheck": {"enabled": False},
                    },
                )
            elif fallback_mode == "self_calibrated_constrained":
                selected = select_budget_by_constrained_utility(
                    metrics_by_budget=metrics_by_budget,
                    calibration_cfg={
                        "coverage_constraint": dict(uncertain_cfg["coverage_constraint"]),
                        "utility": compact_utility,
                        "tiebreak": {"epsilon": 0.01, "prefer_smaller_budget": True},
                        "constrained_recheck": {"enabled": False},
                    },
                )
            else:
                raise ValueError(
                    "Unsupported uncertain fallback_mode for hierarchical routing: "
                    f"{fallback_mode}"
                )
            selected["selection_stage"] = "uncertain_fallback_policy"
            selected["arbitration_triggered"] = False
        else:
            raise ValueError(
                "Unsupported uncertain mode for hierarchical routing: "
                f"{uncertain_mode}"
            )
    selected["regime"] = route.regime
    selected["shape_score"] = float(route.shape_score)
    selected["descriptor"] = descriptor.to_dict()
    return selected
