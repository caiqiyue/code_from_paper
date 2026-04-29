import unittest

from paper_new_selector.budget_calibration import (
    _select_budget_with_tiebreak,
    combine_budget_metrics,
    compute_budget_cost,
    compute_selected_coverage_score,
    compute_selected_redundancy_score,
    evaluate_near_boundary_recheck,
    resolve_seed_top_k_by_self_calibration,
    select_budget_with_recheck,
    should_trigger_near_boundary_recheck,
)


class BudgetCalibrationTests(unittest.TestCase):
    def test_compute_selected_coverage_score_reports_aggregate_stats(self):
        result = compute_selected_coverage_score(
            private_vectors=[[1.0, 0.0], [0.0, 1.0]],
            selected_vectors=[[1.0, 0.0]],
        )
        self.assertAlmostEqual(result["coverage_mean"], 0.5)
        self.assertAlmostEqual(result["coverage_p25"], 0.0)
        self.assertAlmostEqual(result["coverage_min"], 0.0)

    def test_compute_selected_redundancy_score_is_zero_for_single_vector(self):
        self.assertEqual(
            compute_selected_redundancy_score(selected_vectors=[[1.0, 0.0]]),
            0.0,
        )

    def test_compute_budget_cost_scales_to_candidate_range(self):
        self.assertEqual(
            compute_budget_cost(seed_top_k=20, candidate_seed_top_k=[18, 19, 20, 21, 22]),
            0.5,
        )

    def test_combine_budget_metrics_normalizes_and_computes_utility(self):
        enriched = combine_budget_metrics(
            metrics_by_budget={
                18: {
                    "support_score": 0.8,
                    "genericity_score": 0.4,
                    "redundancy_score": 0.3,
                    "coverage_mean": 0.6,
                    "budget_cost": 0.0,
                },
                22: {
                    "support_score": 0.7,
                    "genericity_score": 0.5,
                    "redundancy_score": 0.4,
                    "coverage_mean": 0.9,
                    "budget_cost": 1.0,
                },
            },
            calibration_cfg={
                "utility": {
                    "support_weight": 1.0,
                    "genericity_weight": 0.5,
                    "redundancy_weight": 0.3,
                    "coverage_weight": 0.4,
                    "budget_weight": 0.1,
                }
            },
        )
        self.assertIn("normalized_metrics", enriched[18])
        self.assertIn("utility", enriched[22])
        self.assertGreater(enriched[18]["utility"], enriched[22]["utility"])

    def test_self_calibration_prefers_smaller_budget_when_utility_is_tied(self):
        result = resolve_seed_top_k_by_self_calibration(
            selector_cfg={
                "seed_top_k": 20,
                "lambda_generic": 0.0,
                "lambda_redundancy": 0.0,
                "hard_negative_top_k": 1,
                "seed_budget_rule": {
                    "enabled": True,
                    "mode": "self_calibrated",
                    "candidate_seed_top_k": [18, 19],
                    "utility": {
                        "support_weight": 1.0,
                        "genericity_weight": 0.0,
                        "redundancy_weight": 0.0,
                        "coverage_weight": 0.0,
                        "budget_weight": 0.0,
                    },
                    "tiebreak": {
                        "epsilon": 1.0,
                        "coverage_gain_min": 0.5,
                        "prefer_smaller_budget": True,
                    },
                },
            },
            candidate_vectors=[[1.0, 0.0], [0.9, 0.1]],
            candidate_texts=["alpha", "beta"],
            private_vectors=[[1.0, 0.0], [0.9, 0.1]],
            private_support=[1.0, 0.9],
            genericity_penalty=[0.0, 0.0],
        )
        self.assertEqual(result["seed_budget_summary"]["resolved_seed_top_k"], 18)
        self.assertTrue(result["seed_budget_summary"]["tiebreak_applied"])
        self.assertIn("near_boundary_recheck", result["seed_budget_summary"])

    def test_tiebreak_uses_pairwise_coverage_gain_for_sparse_budget_sets(self):
        selected = _select_budget_with_tiebreak(
            metrics_by_budget={
                18: {"utility": 0.9, "coverage_mean": 0.20},
                20: {"utility": 0.1, "coverage_mean": 0.24},
                22: {"utility": 0.9005, "coverage_mean": 0.30},
            },
            calibration_cfg={
                "tiebreak": {
                    "epsilon": 1.0,
                    "coverage_gain_min": 0.05,
                    "prefer_smaller_budget": True,
                }
            },
        )
        self.assertEqual(selected["resolved_seed_top_k"], 22)
        self.assertEqual(selected["runner_up_seed_top_k"], 18)
        self.assertTrue(selected["tiebreak_applied"])

    def test_tiebreak_reorders_runner_up_when_preferring_smaller_budget(self):
        selected = _select_budget_with_tiebreak(
            metrics_by_budget={
                18: {"utility": 0.91, "coverage_mean": 0.20},
                20: {"utility": 0.905, "coverage_mean": 0.204},
            },
            calibration_cfg={
                "tiebreak": {
                    "epsilon": 0.01,
                    "coverage_gain_min": 0.05,
                    "prefer_smaller_budget": True,
                }
            },
        )
        self.assertEqual(selected["resolved_seed_top_k"], 18)
        self.assertEqual(selected["runner_up_seed_top_k"], 20)
        self.assertTrue(selected["tiebreak_applied"])

    def test_should_trigger_near_boundary_recheck_requires_larger_runner_up_within_gap(self):
        self.assertTrue(
            should_trigger_near_boundary_recheck(
                selected_budget=18,
                runner_up_budget=20,
                utility_gap=0.08,
                calibration_cfg={
                    "near_boundary_recheck": {
                        "enabled": True,
                        "trigger_gap": 0.12,
                    }
                },
            )
        )
        self.assertFalse(
            should_trigger_near_boundary_recheck(
                selected_budget=20,
                runner_up_budget=18,
                utility_gap=0.08,
                calibration_cfg={
                    "near_boundary_recheck": {
                        "enabled": True,
                        "trigger_gap": 0.12,
                    }
                },
            )
        )

    def test_evaluate_near_boundary_recheck_requires_tail_coverage_gain_and_support_guard(self):
        result = evaluate_near_boundary_recheck(
            metrics_by_budget={
                18: {
                    "coverage_mean": 0.71,
                    "coverage_p25": 0.55,
                    "support_mean": 0.82,
                },
                20: {
                    "coverage_mean": 0.716,
                    "coverage_p25": 0.564,
                    "support_mean": 0.81,
                },
            },
            smaller_budget=18,
            larger_budget=20,
            utility_gap=0.07,
            calibration_cfg={
                "near_boundary_recheck": {
                    "enabled": True,
                    "trigger_gap": 0.12,
                    "coverage_mean_gain_min": 0.004,
                    "coverage_p25_gain_min": 0.008,
                    "support_drop_max": 0.015,
                }
            },
        )
        self.assertTrue(result["pass_recheck"])
        self.assertEqual(result["final_budget"], 20)

    def test_select_budget_with_recheck_promotes_larger_budget_when_guard_passes(self):
        selected = select_budget_with_recheck(
            metrics_by_budget={
                18: {
                    "utility": 0.91,
                    "coverage_mean": 0.71,
                    "coverage_p25": 0.55,
                    "support_mean": 0.82,
                },
                20: {
                    "utility": 0.85,
                    "coverage_mean": 0.716,
                    "coverage_p25": 0.564,
                    "support_mean": 0.81,
                },
            },
            calibration_cfg={
                "tiebreak": {
                    "epsilon": 0.01,
                    "coverage_gain_min": 0.05,
                    "prefer_smaller_budget": True,
                },
                "near_boundary_recheck": {
                    "enabled": True,
                    "trigger_gap": 0.12,
                    "coverage_mean_gain_min": 0.004,
                    "coverage_p25_gain_min": 0.008,
                    "support_drop_max": 0.015,
                },
            },
        )
        self.assertEqual(selected["resolved_seed_top_k"], 20)
        self.assertEqual(selected["runner_up_seed_top_k"], 18)
        self.assertTrue(selected["near_boundary_recheck"]["triggered"])
        self.assertTrue(selected["near_boundary_recheck"]["pass_recheck"])
        self.assertEqual(
            selected["tiebreak_reason"],
            "near_boundary_recheck_promoted_larger_budget",
        )

    def test_select_budget_with_recheck_keeps_smaller_budget_when_guard_fails(self):
        selected = select_budget_with_recheck(
            metrics_by_budget={
                18: {
                    "utility": 0.91,
                    "coverage_mean": 0.71,
                    "coverage_p25": 0.55,
                    "support_mean": 0.82,
                },
                20: {
                    "utility": 0.85,
                    "coverage_mean": 0.712,
                    "coverage_p25": 0.553,
                    "support_mean": 0.79,
                },
            },
            calibration_cfg={
                "tiebreak": {
                    "epsilon": 0.01,
                    "coverage_gain_min": 0.05,
                    "prefer_smaller_budget": True,
                },
                "near_boundary_recheck": {
                    "enabled": True,
                    "trigger_gap": 0.12,
                    "coverage_mean_gain_min": 0.004,
                    "coverage_p25_gain_min": 0.008,
                    "support_drop_max": 0.015,
                },
            },
        )
        self.assertEqual(selected["resolved_seed_top_k"], 18)
        self.assertTrue(selected["near_boundary_recheck"]["triggered"])
        self.assertFalse(selected["near_boundary_recheck"]["pass_recheck"])


if __name__ == "__main__":
    unittest.main()
