from __future__ import annotations

from typing import Any

from .redundancy import cosine_similarity
from .selector import greedy_select_candidates


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
                tiebreak_applied = True
                tiebreak_reason = "prefer_smaller_budget_within_epsilon"
            else:
                best_budget = larger_budget
                best_metrics = metrics_by_budget[best_budget]
                tiebreak_applied = True
                tiebreak_reason = "prefer_larger_budget_for_coverage_gain"

    return {
        "resolved_seed_top_k": int(best_budget),
        "selected_utility": float(best_metrics["utility"]),
        "runner_up_seed_top_k": int(runner_up_budget),
        "runner_up_utility": float(runner_up_metrics["utility"]),
        "utility_gap": float(utility_gap),
        "tiebreak_applied": bool(tiebreak_applied),
        "tiebreak_reason": tiebreak_reason,
    }


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
    selected = _select_budget_with_tiebreak(
        metrics_by_budget=enriched_metrics,
        calibration_cfg=calibration_cfg,
    )
    resolved_seed_top_k = int(selected["resolved_seed_top_k"])
    decision = decisions_by_budget[resolved_seed_top_k]

    return {
        "decision": decision,
        "seed_budget_summary": {
            "configured_seed_top_k": int(selector_cfg["seed_top_k"]),
            "resolved_seed_top_k": resolved_seed_top_k,
            "rule": calibration_cfg,
            "mode": str(calibration_cfg.get("mode", "self_calibrated")),
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
        },
    }
