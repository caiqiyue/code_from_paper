from __future__ import annotations

from typing import Any

from .redundancy import cosine_similarity
from .selector import greedy_select_candidates

VALID_COVERAGE_METRICS = {
    "coverage_mean",
    "coverage_p25",
    "coverage_min",
}


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _percentile_nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if percentile <= 0:
        return float(min(values))
    if percentile >= 100:
        return float(max(values))
    sorted_values = sorted(float(value) for value in values)
    rank = int(-(-len(sorted_values) * float(percentile) // 100))
    return float(sorted_values[max(0, rank - 1)])


def compute_selected_support_score(
    *, selected_indices: list[int], private_support: list[float]
) -> float:
    return _mean([float(private_support[index]) for index in selected_indices])


def compute_selected_genericity_score(
    *, selected_indices: list[int], genericity_penalty: list[float]
) -> float:
    return _mean([float(genericity_penalty[index]) for index in selected_indices])


def compute_selected_redundancy_score(*, selected_vectors: list[list[float]]) -> float:
    if len(selected_vectors) < 2:
        return 0.0
    similarities: list[float] = []
    for left_index in range(len(selected_vectors)):
        for right_index in range(left_index + 1, len(selected_vectors)):
            similarities.append(
                cosine_similarity(
                    selected_vectors[left_index],
                    selected_vectors[right_index],
                )
            )
    return _mean(similarities)


def compute_selected_coverage_score(
    *,
    private_vectors: list[list[float]],
    selected_vectors: list[list[float]],
) -> dict[str, float]:
    if not private_vectors or not selected_vectors:
        return {"coverage_mean": 0.0, "coverage_p25": 0.0, "coverage_min": 0.0}
    coverage_values = [
        max(cosine_similarity(private_vector, selected) for selected in selected_vectors)
        for private_vector in private_vectors
    ]
    return {
        "coverage_mean": _mean(coverage_values),
        "coverage_p25": _percentile_nearest_rank(coverage_values, 25),
        "coverage_min": float(min(coverage_values)),
    }


def compute_budget_cost(
    *, seed_top_k: int, candidate_seed_top_k: list[int]
) -> float:
    if not candidate_seed_top_k:
        return 0.0
    min_budget = min(int(value) for value in candidate_seed_top_k)
    max_budget = max(int(value) for value in candidate_seed_top_k)
    if min_budget == max_budget:
        return 0.0
    return float((int(seed_top_k) - min_budget) / (max_budget - min_budget))


def _normalize_metric_series(metric_values: dict[int, float]) -> dict[int, float]:
    if not metric_values:
        return {}
    low = min(float(value) for value in metric_values.values())
    high = max(float(value) for value in metric_values.values())
    if abs(high - low) <= 1e-8:
        return {int(key): 0.0 for key in metric_values}
    return {
        int(key): float((float(value) - low) / (high - low))
        for key, value in metric_values.items()
    }


def combine_budget_metrics(
    *,
    metrics_by_budget: dict[int, dict[str, Any]],
    calibration_cfg: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    utility_cfg = dict(calibration_cfg.get("utility", {}))
    support_weight = float(utility_cfg.get("support_weight", 1.0))
    genericity_weight = float(utility_cfg.get("genericity_weight", 0.5))
    redundancy_weight = float(utility_cfg.get("redundancy_weight", 0.3))
    coverage_weight = float(utility_cfg.get("coverage_weight", 0.4))
    budget_weight = float(utility_cfg.get("budget_weight", 0.1))

    support_values = {
        budget: float(metrics["support_score"])
        for budget, metrics in metrics_by_budget.items()
    }
    genericity_values = {
        budget: float(metrics["genericity_score"])
        for budget, metrics in metrics_by_budget.items()
    }
    redundancy_values = {
        budget: float(metrics["redundancy_score"])
        for budget, metrics in metrics_by_budget.items()
    }
    coverage_values = {
        budget: float(metrics["coverage_mean"])
        for budget, metrics in metrics_by_budget.items()
    }
    budget_cost_values = {
        budget: float(metrics["budget_cost"])
        for budget, metrics in metrics_by_budget.items()
    }

    normalized_support = _normalize_metric_series(support_values)
    normalized_genericity = _normalize_metric_series(genericity_values)
    normalized_redundancy = _normalize_metric_series(redundancy_values)
    normalized_coverage = _normalize_metric_series(coverage_values)
    normalized_budget_cost = _normalize_metric_series(budget_cost_values)

    sorted_budgets = sorted(metrics_by_budget)
    previous_coverage_mean = 0.0
    previous_coverage_normalized = 0.0
    enriched: dict[int, dict[str, Any]] = {}
    for budget in sorted_budgets:
        raw_metrics = dict(metrics_by_budget[budget])
        coverage_mean = float(raw_metrics["coverage_mean"])
        normalized_metrics = {
            "support_score": normalized_support[budget],
            "genericity_score": normalized_genericity[budget],
            "redundancy_score": normalized_redundancy[budget],
            "coverage_score": normalized_coverage[budget],
            "budget_cost": normalized_budget_cost[budget],
        }
        coverage_gain = coverage_mean - previous_coverage_mean
        normalized_coverage_gain = normalized_metrics["coverage_score"] - previous_coverage_normalized
        utility = (
            support_weight * normalized_metrics["support_score"]
            - genericity_weight * normalized_metrics["genericity_score"]
            - redundancy_weight * normalized_metrics["redundancy_score"]
            + coverage_weight * normalized_metrics["coverage_score"]
            - budget_weight * normalized_metrics["budget_cost"]
        )
        raw_metrics["normalized_metrics"] = normalized_metrics
        raw_metrics["coverage_gain"] = float(coverage_gain)
        raw_metrics["normalized_coverage_gain"] = float(normalized_coverage_gain)
        raw_metrics["utility"] = float(utility)
        enriched[budget] = raw_metrics
        previous_coverage_mean = coverage_mean
        previous_coverage_normalized = normalized_metrics["coverage_score"]
    return enriched


def compute_relative_coverage_threshold(
    *, best_coverage_p25: float, relative_ratio: float
) -> float:
    return float(best_coverage_p25) * float(relative_ratio)


def _normalize_metric_name(metric_name: str) -> str:
    return str(metric_name).strip()


def _extract_metric_value(metrics: dict[str, Any], metric_name: str) -> float:
    normalized_name = _normalize_metric_name(metric_name)
    if normalized_name not in VALID_COVERAGE_METRICS:
        raise ValueError(f"Unsupported coverage constraint metric: {normalized_name}")
    if normalized_name not in metrics:
        raise ValueError(f"Missing metric in calibration summary: {normalized_name}")
    return float(metrics[normalized_name])


def _build_coverage_metric_specs(coverage_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    specs = coverage_cfg.get("metrics")
    if isinstance(specs, list) and specs:
        normalized_specs: list[dict[str, Any]] = []
        for raw_spec in specs:
            if not isinstance(raw_spec, dict):
                continue
            name = _normalize_metric_name(raw_spec.get("name", "coverage_p25"))
            if name not in VALID_COVERAGE_METRICS:
                raise ValueError(f"Unsupported coverage constraint metric: {name}")
            normalized_specs.append(
                {
                    "name": name,
                    "relative_ratio": float(
                        raw_spec.get(
                            "relative_ratio",
                            coverage_cfg.get("relative_ratio", 0.99),
                        )
                    ),
                    "required": bool(raw_spec.get("required", True)),
                    "weight": float(raw_spec.get("weight", 1.0)),
                }
            )
        if normalized_specs:
            return normalized_specs
    default_name = _normalize_metric_name(coverage_cfg.get("metric", "coverage_p25"))
    if default_name not in VALID_COVERAGE_METRICS:
        raise ValueError(f"Unsupported coverage constraint metric: {default_name}")
    return [
        {
            "name": default_name,
            "relative_ratio": float(coverage_cfg.get("relative_ratio", 0.99)),
            "required": True,
            "weight": 1.0,
        }
    ]


def _compute_family_score_by_budget(
    *,
    metrics_by_budget: dict[int, dict[str, Any]],
    metric_specs: list[dict[str, Any]],
) -> dict[int, float]:
    if not metric_specs:
        return {int(budget): 0.0 for budget in metrics_by_budget}

    normalized_by_metric: dict[str, dict[int, float]] = {}
    for metric_spec in metric_specs:
        metric_name = str(metric_spec["name"])
        normalized_by_metric[metric_name] = _normalize_metric_series(
            {
                int(budget): _extract_metric_value(metrics, metric_name)
                for budget, metrics in metrics_by_budget.items()
            }
        )

    family_scores: dict[int, float] = {}
    total_weight = sum(float(metric_spec.get("weight", 1.0)) for metric_spec in metric_specs)
    normalizer = total_weight if total_weight > 0.0 else float(len(metric_specs))
    if normalizer <= 0.0:
        normalizer = 1.0

    for budget in metrics_by_budget:
        weighted_sum = 0.0
        for metric_spec in metric_specs:
            metric_name = str(metric_spec["name"])
            metric_weight = float(metric_spec.get("weight", 1.0))
            weighted_sum += metric_weight * normalized_by_metric[metric_name][int(budget)]
        family_scores[int(budget)] = float(weighted_sum / normalizer)
    return family_scores


def select_feasible_budgets_by_coverage_constraint(
    *,
    metrics_by_budget: dict[int, dict[str, Any]],
    calibration_cfg: dict[str, Any],
) -> dict[str, Any]:
    coverage_cfg = dict(calibration_cfg.get("coverage_constraint", {}))
    metric_specs = _build_coverage_metric_specs(coverage_cfg)
    per_metric: list[dict[str, Any]] = []

    feasible_budgets = sorted(int(budget) for budget in metrics_by_budget)
    for metric_spec in metric_specs:
        metric_name = str(metric_spec["name"])
        metric_values = {
            int(budget): _extract_metric_value(metrics, metric_name)
            for budget, metrics in metrics_by_budget.items()
        }
        best_value = max(metric_values.values()) if metric_values else 0.0
        threshold = compute_relative_coverage_threshold(
            best_coverage_p25=best_value,
            relative_ratio=float(metric_spec["relative_ratio"]),
        )
        metric_feasible = [
            int(budget)
            for budget, value in sorted(metric_values.items())
            if float(value) >= float(threshold)
        ]
        if bool(metric_spec.get("required", True)):
            feasible_budgets = [
                int(budget) for budget in feasible_budgets if int(budget) in set(metric_feasible)
            ]
        per_metric.append(
            {
                "name": metric_name,
                "relative_ratio": float(metric_spec["relative_ratio"]),
                "required": bool(metric_spec.get("required", True)),
                "weight": float(metric_spec.get("weight", 1.0)),
                "best_value": float(best_value),
                "threshold": float(threshold),
                "feasible_budgets": metric_feasible,
            }
        )

    primary_metric = per_metric[0] if per_metric else {
        "name": str(coverage_cfg.get("metric", "coverage_p25")),
        "relative_ratio": float(coverage_cfg.get("relative_ratio", 0.99)),
        "best_value": 0.0,
        "threshold": 0.0,
    }
    family_scores = _compute_family_score_by_budget(
        metrics_by_budget=metrics_by_budget,
        metric_specs=metric_specs,
    )
    return {
        "mode": str(coverage_cfg.get("mode", "single_metric_relative")),
        "metric": str(primary_metric["name"]),
        "relative_ratio": float(primary_metric["relative_ratio"]),
        "best_coverage_p25": float(primary_metric["best_value"]),
        "threshold": float(primary_metric["threshold"]),
        "feasible_budgets": [int(budget) for budget in feasible_budgets],
        "metrics": per_metric,
        "family_score_by_budget": {
            str(budget): float(score) for budget, score in sorted(family_scores.items())
        },
    }


def select_feasible_budgets_by_coverage_p25(
    *,
    metrics_by_budget: dict[int, dict[str, Any]],
    calibration_cfg: dict[str, Any],
) -> dict[str, Any]:
    return select_feasible_budgets_by_coverage_constraint(
        metrics_by_budget=metrics_by_budget,
        calibration_cfg=calibration_cfg,
    )


def combine_feasible_budget_metrics(
    *,
    metrics_by_budget: dict[int, dict[str, Any]],
    feasible_budgets: list[int],
    calibration_cfg: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    utility_cfg = dict(calibration_cfg.get("utility", {}))
    support_weight = float(utility_cfg.get("support_weight", 1.0))
    genericity_weight = float(utility_cfg.get("genericity_weight", 0.5))
    redundancy_weight = float(utility_cfg.get("redundancy_weight", 0.3))
    budget_weight = float(utility_cfg.get("budget_weight", 0.1))

    subset = {
        int(budget): dict(metrics_by_budget[int(budget)])
        for budget in feasible_budgets
    }
    normalized_support = _normalize_metric_series(
        {budget: float(metrics["support_score"]) for budget, metrics in subset.items()}
    )
    normalized_genericity = _normalize_metric_series(
        {budget: float(metrics["genericity_score"]) for budget, metrics in subset.items()}
    )
    normalized_redundancy = _normalize_metric_series(
        {budget: float(metrics["redundancy_score"]) for budget, metrics in subset.items()}
    )
    normalized_budget_cost = _normalize_metric_series(
        {budget: float(metrics["budget_cost"]) for budget, metrics in subset.items()}
    )

    for budget, raw_metrics in subset.items():
        feasible_normalized_metrics = {
            "support_score": normalized_support[budget],
            "genericity_score": normalized_genericity[budget],
            "redundancy_score": normalized_redundancy[budget],
            "budget_cost": normalized_budget_cost[budget],
        }
        feasible_utility = (
            support_weight * feasible_normalized_metrics["support_score"]
            - genericity_weight * feasible_normalized_metrics["genericity_score"]
            - redundancy_weight * feasible_normalized_metrics["redundancy_score"]
            - budget_weight * feasible_normalized_metrics["budget_cost"]
        )
        raw_metrics["feasible_normalized_metrics"] = feasible_normalized_metrics
        raw_metrics["feasible_utility"] = float(feasible_utility)
        raw_metrics["utility"] = float(feasible_utility)
    return subset


def _select_budget_from_feasible_metrics(
    *,
    metrics_by_budget: dict[int, dict[str, Any]],
    calibration_cfg: dict[str, Any],
) -> dict[str, Any]:
    tiebreak_cfg = dict(calibration_cfg.get("tiebreak", {}))
    epsilon = float(tiebreak_cfg.get("epsilon", 0.01))
    prefer_smaller = bool(tiebreak_cfg.get("prefer_smaller_budget", True))

    ranked = sorted(
        metrics_by_budget.items(),
        key=lambda item: (float(item[1]["utility"]), -int(item[0])),
        reverse=True,
    )
    best_budget, best_metrics = ranked[0]
    runner_up_budget = best_budget
    runner_up_metrics = best_metrics
    utility_gap = 0.0
    tiebreak_applied = False
    tiebreak_reason = "argmax_feasible_utility"

    if len(ranked) > 1:
        runner_up_budget, runner_up_metrics = ranked[1]
        utility_gap = float(best_metrics["utility"] - runner_up_metrics["utility"])
        if utility_gap <= epsilon and prefer_smaller:
            smaller_budget = min(int(best_budget), int(runner_up_budget))
            larger_budget = max(int(best_budget), int(runner_up_budget))
            best_budget = smaller_budget
            best_metrics = metrics_by_budget[best_budget]
            runner_up_budget = larger_budget
            runner_up_metrics = metrics_by_budget[runner_up_budget]
            tiebreak_applied = True
            tiebreak_reason = "prefer_smaller_feasible_budget_within_epsilon"

    return {
        "resolved_seed_top_k": int(best_budget),
        "selected_utility": float(best_metrics["utility"]),
        "runner_up_seed_top_k": int(runner_up_budget),
        "runner_up_utility": float(runner_up_metrics["utility"]),
        "utility_gap": float(utility_gap),
        "tiebreak_applied": bool(tiebreak_applied),
        "tiebreak_reason": tiebreak_reason,
    }


def _build_default_constrained_recheck_summary(*, enabled: bool) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "triggered": False,
        "selected_budget": None,
        "candidate_budget": None,
        "promoted_budget": None,
        "support_drop": 0.0,
        "support_drop_normalized": 0.0,
        "coverage_mean_gain": 0.0,
        "coverage_p25_gain": 0.0,
        "coverage_min_gain": 0.0,
        "family_score_gain": 0.0,
        "pass_recheck": False,
        "reason": "disabled" if not enabled else "not_triggered",
    }


def evaluate_constrained_recheck(
    *,
    metrics_by_budget: dict[int, dict[str, Any]],
    feasible_budgets: list[int],
    selected_budget: int,
    calibration_cfg: dict[str, Any],
    coverage_constraint: dict[str, Any],
) -> dict[str, Any]:
    recheck_cfg = dict(calibration_cfg.get("constrained_recheck", {}))
    summary = _build_default_constrained_recheck_summary(
        enabled=bool(recheck_cfg.get("enabled", False))
    )
    if not summary["enabled"]:
        return summary

    larger_candidates = [
        int(budget) for budget in sorted(feasible_budgets) if int(budget) > int(selected_budget)
    ]
    if not larger_candidates:
        summary.update(
            {
                "selected_budget": int(selected_budget),
                "reason": "no_larger_feasible_budget",
            }
        )
        return summary

    candidate_mode = str(recheck_cfg.get("candidate_mode", "all_larger"))
    candidates_to_check = (
        [larger_candidates[0]] if candidate_mode == "adjacent_only" else list(larger_candidates)
    )

    support_drop_ratio_max = float(
        recheck_cfg.get(
            "support_drop_ratio_max",
            recheck_cfg.get("support_drop_max", 0.02),
        )
    )
    coverage_mean_gain_min = float(recheck_cfg.get("coverage_mean_gain_min", 0.0))
    coverage_p25_gain_min = float(recheck_cfg.get("coverage_p25_gain_min", 0.0))
    coverage_min_gain_min = float(recheck_cfg.get("coverage_min_gain_min", 0.0))
    family_score_gain_min = float(recheck_cfg.get("family_score_gain_min", 0.0))

    smaller_metrics = metrics_by_budget[int(selected_budget)]
    family_scores = dict(coverage_constraint.get("family_score_by_budget", {}))
    smaller_family_score = float(family_scores.get(str(int(selected_budget)), 0.0))
    last_candidate_budget = int(candidates_to_check[-1])
    last_trace: dict[str, Any] | None = None
    for candidate_budget in candidates_to_check:
        larger_metrics = metrics_by_budget[int(candidate_budget)]
        larger_family_score = float(family_scores.get(str(int(candidate_budget)), 0.0))

        support_drop = float(smaller_metrics["support_mean"] - larger_metrics["support_mean"])
        support_drop_normalized = float(
            support_drop / max(abs(float(smaller_metrics["support_mean"])), 1e-8)
        )
        coverage_mean_gain = float(larger_metrics["coverage_mean"] - smaller_metrics["coverage_mean"])
        coverage_p25_gain = float(larger_metrics["coverage_p25"] - smaller_metrics["coverage_p25"])
        coverage_min_gain = float(larger_metrics["coverage_min"] - smaller_metrics["coverage_min"])
        family_score_gain = float(larger_family_score - smaller_family_score)
        pass_recheck = (
            support_drop_normalized <= support_drop_ratio_max
            and coverage_mean_gain >= coverage_mean_gain_min
            and coverage_p25_gain >= coverage_p25_gain_min
            and coverage_min_gain >= coverage_min_gain_min
            and family_score_gain >= family_score_gain_min
        )
        last_trace = {
            "triggered": True,
            "selected_budget": int(selected_budget),
            "candidate_budget": int(candidate_budget),
            "support_drop": float(support_drop),
            "support_drop_normalized": float(support_drop_normalized),
            "coverage_mean_gain": float(coverage_mean_gain),
            "coverage_p25_gain": float(coverage_p25_gain),
            "coverage_min_gain": float(coverage_min_gain),
            "family_score_gain": float(family_score_gain),
        }
        if pass_recheck:
            summary.update(
                {
                    **last_trace,
                    "promoted_budget": int(candidate_budget),
                    "pass_recheck": True,
                    "reason": (
                        "promoted_adjacent_feasible_budget"
                        if candidate_mode == "adjacent_only"
                        else "promoted_larger_feasible_budget"
                    ),
                }
            )
            return summary

    if last_trace is not None:
        summary.update(
            {
                **last_trace,
                "promoted_budget": None,
                "pass_recheck": False,
                "reason": (
                    "adjacent_feasible_budget_failed_guards"
                    if candidate_mode == "adjacent_only"
                    else "no_larger_budget_passed_guards"
                ),
            }
        )
    else:
        summary.update(
            {
                "triggered": True,
                "selected_budget": int(selected_budget),
                "candidate_budget": int(last_candidate_budget),
                "promoted_budget": None,
                "reason": "no_larger_budget_passed_guards",
            }
        )
    return summary


def select_budget_by_constrained_utility(
    *,
    metrics_by_budget: dict[int, dict[str, Any]],
    calibration_cfg: dict[str, Any],
) -> dict[str, Any]:
    coverage_constraint = select_feasible_budgets_by_coverage_constraint(
        metrics_by_budget=metrics_by_budget,
        calibration_cfg=calibration_cfg,
    )
    feasible_budgets = list(coverage_constraint["feasible_budgets"])
    for budget, metrics in metrics_by_budget.items():
        metrics["coverage_feasible"] = int(budget) in feasible_budgets

    if not feasible_budgets:
        fallback = _select_budget_with_tiebreak(
            metrics_by_budget=metrics_by_budget,
            calibration_cfg=calibration_cfg,
        )
        fallback["constrained_recheck"] = _build_default_constrained_recheck_summary(
            enabled=bool(dict(calibration_cfg.get("constrained_recheck", {})).get("enabled", False))
        )
        fallback["coverage_constraint"] = coverage_constraint
        fallback["selection_stage"] = "fallback_argmax_utility"
        fallback["fallback_used"] = True
        return fallback

    feasible_metrics = combine_feasible_budget_metrics(
        metrics_by_budget=metrics_by_budget,
        feasible_budgets=feasible_budgets,
        calibration_cfg=calibration_cfg,
    )
    for budget in feasible_budgets:
        metrics_by_budget[int(budget)]["base_utility"] = float(
            metrics_by_budget[int(budget)].get(
                "utility",
                feasible_metrics[int(budget)]["feasible_utility"],
            )
        )
        metrics_by_budget[int(budget)]["feasible_normalized_metrics"] = dict(
            feasible_metrics[int(budget)]["feasible_normalized_metrics"]
        )
        metrics_by_budget[int(budget)]["feasible_utility"] = float(
            feasible_metrics[int(budget)]["feasible_utility"]
        )
        metrics_by_budget[int(budget)]["utility"] = float(
            feasible_metrics[int(budget)]["utility"]
        )
    selected = _select_budget_from_feasible_metrics(
        metrics_by_budget=feasible_metrics,
        calibration_cfg=calibration_cfg,
    )
    constrained_recheck = evaluate_constrained_recheck(
        metrics_by_budget=metrics_by_budget,
        feasible_budgets=feasible_budgets,
        selected_budget=int(selected["resolved_seed_top_k"]),
        calibration_cfg=calibration_cfg,
        coverage_constraint=coverage_constraint,
    )
    if constrained_recheck["enabled"] and constrained_recheck["pass_recheck"]:
        previous_selected_budget = int(selected["resolved_seed_top_k"])
        promoted_budget = int(constrained_recheck["promoted_budget"])
        selected["resolved_seed_top_k"] = promoted_budget
        selected["selected_utility"] = float(metrics_by_budget[promoted_budget]["utility"])
        selected["runner_up_seed_top_k"] = previous_selected_budget
        selected["runner_up_utility"] = float(metrics_by_budget[previous_selected_budget]["utility"])
        selected["utility_gap"] = float(
            selected["selected_utility"] - selected["runner_up_utility"]
        )
        selected["tiebreak_applied"] = True
        selected["tiebreak_reason"] = "constrained_recheck_promoted_larger_budget"
    selected["coverage_constraint"] = coverage_constraint
    selected["constrained_recheck"] = constrained_recheck
    selected["selection_stage"] = "feasible_set_utility"
    selected["fallback_used"] = False
    return selected


def _select_budget_with_tiebreak(
    *,
    metrics_by_budget: dict[int, dict[str, Any]],
    calibration_cfg: dict[str, Any],
) -> dict[str, Any]:
    tiebreak_cfg = dict(calibration_cfg.get("tiebreak", {}))
    epsilon = float(tiebreak_cfg.get("epsilon", 0.01))
    coverage_gain_min = float(tiebreak_cfg.get("coverage_gain_min", 0.005))
    prefer_smaller = bool(tiebreak_cfg.get("prefer_smaller_budget", True))

    ranked = sorted(
        metrics_by_budget.items(),
        key=lambda item: (float(item[1]["utility"]), -int(item[0])),
        reverse=True,
    )
    best_budget, best_metrics = ranked[0]
    runner_up_budget = best_budget
    runner_up_metrics = best_metrics
    utility_gap = 0.0
    tiebreak_applied = False
    tiebreak_reason = "argmax_utility"

    if len(ranked) > 1:
        original_best_budget, original_best_metrics = best_budget, best_metrics
        runner_up_budget, runner_up_metrics = ranked[1]
        utility_gap = float(best_metrics["utility"] - runner_up_metrics["utility"])
        if utility_gap <= epsilon and prefer_smaller:
            smaller_budget = min(int(best_budget), int(runner_up_budget))
            larger_budget = max(int(best_budget), int(runner_up_budget))
            larger_coverage_gain = float(
                metrics_by_budget[larger_budget]["coverage_mean"]
                - metrics_by_budget[smaller_budget]["coverage_mean"]
            )
            if larger_coverage_gain < coverage_gain_min:
                best_budget = smaller_budget
                best_metrics = metrics_by_budget[best_budget]
                runner_up_budget = larger_budget
                runner_up_metrics = metrics_by_budget[runner_up_budget]
                tiebreak_applied = True
                tiebreak_reason = "prefer_smaller_budget_within_epsilon"
            else:
                best_budget = larger_budget
                best_metrics = metrics_by_budget[best_budget]
                runner_up_budget = smaller_budget
                runner_up_metrics = metrics_by_budget[runner_up_budget]
                tiebreak_applied = True
                tiebreak_reason = "prefer_larger_budget_for_coverage_gain"
        else:
            best_budget = original_best_budget
            best_metrics = original_best_metrics

    return {
        "resolved_seed_top_k": int(best_budget),
        "selected_utility": float(best_metrics["utility"]),
        "runner_up_seed_top_k": int(runner_up_budget),
        "runner_up_utility": float(runner_up_metrics["utility"]),
        "utility_gap": float(utility_gap),
        "tiebreak_applied": bool(tiebreak_applied),
        "tiebreak_reason": tiebreak_reason,
    }


def _build_default_near_boundary_recheck_summary(*, enabled: bool) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "triggered": False,
        "utility_gap": 0.0,
        "smaller_budget": None,
        "larger_budget": None,
        "coverage_mean_gain": 0.0,
        "coverage_p25_gain": 0.0,
        "support_drop": 0.0,
        "pass_recheck": False,
        "final_budget": None,
        "reason": "disabled" if not enabled else "not_triggered",
    }


def should_trigger_near_boundary_recheck(
    *,
    selected_budget: int,
    runner_up_budget: int,
    utility_gap: float,
    calibration_cfg: dict[str, Any],
) -> bool:
    recheck_cfg = dict(calibration_cfg.get("near_boundary_recheck", {}))
    if not bool(recheck_cfg.get("enabled", False)):
        return False
    trigger_gap = float(recheck_cfg.get("trigger_gap", 0.12))
    return (
        int(runner_up_budget) > int(selected_budget)
        and float(utility_gap) <= trigger_gap
    )


def evaluate_near_boundary_recheck(
    *,
    metrics_by_budget: dict[int, dict[str, Any]],
    smaller_budget: int,
    larger_budget: int,
    utility_gap: float,
    calibration_cfg: dict[str, Any],
) -> dict[str, Any]:
    recheck_cfg = dict(calibration_cfg.get("near_boundary_recheck", {}))
    coverage_mean_gain_min = float(recheck_cfg.get("coverage_mean_gain_min", 0.004))
    coverage_p25_gain_min = float(recheck_cfg.get("coverage_p25_gain_min", 0.008))
    support_drop_max = float(recheck_cfg.get("support_drop_max", 0.015))

    smaller_metrics = metrics_by_budget[int(smaller_budget)]
    larger_metrics = metrics_by_budget[int(larger_budget)]

    coverage_mean_gain = float(larger_metrics["coverage_mean"] - smaller_metrics["coverage_mean"])
    coverage_p25_gain = float(larger_metrics["coverage_p25"] - smaller_metrics["coverage_p25"])
    support_drop = float(smaller_metrics["support_mean"] - larger_metrics["support_mean"])
    pass_recheck = (
        coverage_mean_gain >= coverage_mean_gain_min
        and coverage_p25_gain >= coverage_p25_gain_min
        and support_drop <= support_drop_max
    )

    return {
        "enabled": True,
        "triggered": True,
        "utility_gap": float(utility_gap),
        "smaller_budget": int(smaller_budget),
        "larger_budget": int(larger_budget),
        "coverage_mean_gain": coverage_mean_gain,
        "coverage_p25_gain": coverage_p25_gain,
        "support_drop": support_drop,
        "pass_recheck": bool(pass_recheck),
        "final_budget": int(larger_budget if pass_recheck else smaller_budget),
        "reason": (
            "coverage_guard_passed"
            if pass_recheck
            else "coverage_guard_failed"
        ),
    }


def select_budget_with_recheck(
    *,
    metrics_by_budget: dict[int, dict[str, Any]],
    calibration_cfg: dict[str, Any],
) -> dict[str, Any]:
    selected = _select_budget_with_tiebreak(
        metrics_by_budget=metrics_by_budget,
        calibration_cfg=calibration_cfg,
    )
    recheck_cfg = dict(calibration_cfg.get("near_boundary_recheck", {}))
    recheck_summary = _build_default_near_boundary_recheck_summary(
        enabled=bool(recheck_cfg.get("enabled", False))
    )

    selected_budget = int(selected["resolved_seed_top_k"])
    runner_up_budget = int(selected["runner_up_seed_top_k"])
    utility_gap = float(selected["utility_gap"])

    if not should_trigger_near_boundary_recheck(
        selected_budget=selected_budget,
        runner_up_budget=runner_up_budget,
        utility_gap=utility_gap,
        calibration_cfg=calibration_cfg,
    ):
        if recheck_summary["enabled"]:
            recheck_summary.update(
                {
                    "utility_gap": utility_gap,
                    "final_budget": selected_budget,
                    "reason": (
                        "runner_up_not_larger"
                        if runner_up_budget <= selected_budget
                        else "utility_gap_above_trigger"
                    ),
                }
            )
        selected["near_boundary_recheck"] = recheck_summary
        return selected

    recheck_summary = evaluate_near_boundary_recheck(
        metrics_by_budget=metrics_by_budget,
        smaller_budget=selected_budget,
        larger_budget=runner_up_budget,
        utility_gap=utility_gap,
        calibration_cfg=calibration_cfg,
    )
    final_budget = int(recheck_summary["final_budget"])
    if final_budget != selected_budget:
        previous_selected_budget = selected_budget
        selected["resolved_seed_top_k"] = final_budget
        selected["selected_utility"] = float(metrics_by_budget[final_budget]["utility"])
        selected["runner_up_seed_top_k"] = previous_selected_budget
        selected["runner_up_utility"] = float(
            metrics_by_budget[previous_selected_budget]["utility"]
        )
        selected["tiebreak_applied"] = True
        selected["tiebreak_reason"] = "near_boundary_recheck_promoted_larger_budget"
    selected["near_boundary_recheck"] = recheck_summary
    return selected


def resolve_seed_top_k_by_self_calibration(
    *,
    selector_cfg: dict[str, Any],
    candidate_vectors: list[list[float]],
    candidate_texts: list[str],
    private_vectors: list[list[float]],
    private_support: list[float],
    genericity_penalty: list[float],
) -> dict[str, Any]:
    calibration_cfg = dict(selector_cfg.get("seed_budget_rule", {}))
    candidate_seed_top_k = [
        int(value) for value in calibration_cfg.get("candidate_seed_top_k", [int(selector_cfg["seed_top_k"])])
    ]
    candidate_seed_top_k = sorted(set(candidate_seed_top_k))

    decisions_by_budget: dict[int, Any] = {}
    metrics_by_budget: dict[int, dict[str, Any]] = {}
    for seed_top_k in candidate_seed_top_k:
        decision = greedy_select_candidates(
            candidate_vectors=candidate_vectors,
            candidate_texts=candidate_texts,
            private_support=private_support,
            genericity_penalty=genericity_penalty,
            lambda_generic=float(selector_cfg["lambda_generic"]),
            lambda_redundancy=float(selector_cfg["lambda_redundancy"]),
            seed_top_k=int(seed_top_k),
            hard_negative_top_k=int(selector_cfg["hard_negative_top_k"]),
        )
        selected_indices = list(decision.selected_indices)
        selected_vectors = [candidate_vectors[index] for index in selected_indices]
        coverage_stats = compute_selected_coverage_score(
            private_vectors=private_vectors,
            selected_vectors=selected_vectors,
        )
        decisions_by_budget[int(seed_top_k)] = decision
        metrics_by_budget[int(seed_top_k)] = {
            "selected_count": len(selected_indices),
            "selected_indices": selected_indices,
            "support_score": compute_selected_support_score(
                selected_indices=selected_indices,
                private_support=private_support,
            ),
            "support_mean": compute_selected_support_score(
                selected_indices=selected_indices,
                private_support=private_support,
            ),
            "genericity_score": compute_selected_genericity_score(
                selected_indices=selected_indices,
                genericity_penalty=genericity_penalty,
            ),
            "redundancy_score": compute_selected_redundancy_score(
                selected_vectors=selected_vectors,
            ),
            "coverage_mean": coverage_stats["coverage_mean"],
            "coverage_p25": coverage_stats["coverage_p25"],
            "coverage_min": coverage_stats["coverage_min"],
            "budget_cost": compute_budget_cost(
                seed_top_k=int(seed_top_k),
                candidate_seed_top_k=candidate_seed_top_k,
            ),
        }

    enriched_metrics = combine_budget_metrics(
        metrics_by_budget=metrics_by_budget,
        calibration_cfg=calibration_cfg,
    )
    mode = str(calibration_cfg.get("mode", "self_calibrated"))
    if mode == "self_calibrated":
        selected = select_budget_with_recheck(
            metrics_by_budget=enriched_metrics,
            calibration_cfg=calibration_cfg,
        )
    elif mode == "self_calibrated_constrained":
        selected = select_budget_by_constrained_utility(
            metrics_by_budget=enriched_metrics,
            calibration_cfg=calibration_cfg,
        )
    else:
        raise ValueError(f"Unsupported seed_budget_rule.mode: {mode}")
    resolved_seed_top_k = int(selected["resolved_seed_top_k"])
    decision = decisions_by_budget[resolved_seed_top_k]

    seed_budget_summary = {
        "configured_seed_top_k": int(selector_cfg["seed_top_k"]),
        "resolved_seed_top_k": resolved_seed_top_k,
        "rule": calibration_cfg,
        "mode": mode,
        "candidate_seed_top_k": candidate_seed_top_k,
        "per_budget_metrics": {
            str(budget): enriched_metrics[budget]
            for budget in candidate_seed_top_k
        },
        "selected_utility": float(selected["selected_utility"]),
        "runner_up_seed_top_k": int(selected["runner_up_seed_top_k"]),
        "runner_up_utility": float(selected["runner_up_utility"]),
        "utility_gap": float(selected["utility_gap"]),
        "tiebreak_applied": bool(selected["tiebreak_applied"]),
        "tiebreak_reason": str(selected["tiebreak_reason"]),
    }
    if "near_boundary_recheck" in selected:
        seed_budget_summary["near_boundary_recheck"] = dict(selected["near_boundary_recheck"])
    if "coverage_constraint" in selected:
        seed_budget_summary["coverage_constraint"] = dict(selected["coverage_constraint"])
    if "constrained_recheck" in selected:
        seed_budget_summary["constrained_recheck"] = dict(selected["constrained_recheck"])
    if "selection_stage" in selected:
        seed_budget_summary["selection_stage"] = str(selected["selection_stage"])
    if "fallback_used" in selected:
        seed_budget_summary["fallback_used"] = bool(selected["fallback_used"])

    return {
        "decision": decision,
        "seed_budget_summary": seed_budget_summary,
    }
