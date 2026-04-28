import unittest

from paper_new_selector.budget_calibration import (
    _select_budget_with_tiebreak,
    combine_budget_metrics,
    compute_budget_cost,
    compute_selected_coverage_score,
    compute_selected_redundancy_score,
    resolve_seed_top_k_by_self_calibration,
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
        self.assertTrue(selected["tiebreak_applied"])


if __name__ == "__main__":
    unittest.main()
