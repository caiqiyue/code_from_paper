import unittest

from paper_new_selector.hierarchical_budget import resolve_hierarchical_budget


class HierarchicalBudgetTests(unittest.TestCase):
    def test_broad_tail_policy_prefers_high_budget_band(self):
        result = resolve_hierarchical_budget(
            private_lengths=[150, 205, 396],
            metrics_by_budget={
                21: {
                    "coverage_p25": 0.1358,
                    "coverage_mean": 0.2480,
                    "support_mean": 0.82,
                    "genericity_score": 0.18,
                    "redundancy_score": 0.09,
                    "budget_cost": 0.75,
                },
                22: {
                    "coverage_p25": 0.1358,
                    "coverage_mean": 0.2490,
                    "support_mean": 0.81,
                    "genericity_score": 0.18,
                    "redundancy_score": 0.09,
                    "budget_cost": 1.0,
                },
            },
            rule_cfg={
                "router": {
                    "tail_threshold": 350,
                    "short_threshold": 120,
                    "tau_center": 0.0,
                    "delta_router": 0.35,
                    "screening_reference": {
                        "median_len": {"mean": 150.0, "std": 50.0},
                        "p75_len": {"mean": 335.0, "std": 90.0},
                        "iqr_len": {"mean": 180.0, "std": 60.0},
                    },
                },
                "policies": {
                    "broad_tail": {
                        "candidate_seed_top_k": [21, 22],
                        "coverage_p25_ratio": 0.98,
                        "coverage_mean_ratio": 0.998,
                        "epsilon": 0.002,
                    },
                    "compact_structured": {
                        "candidate_seed_top_k": [18, 19, 20],
                        "coverage_p25_ratio": 0.98,
                        "utility": {
                            "support_weight": 1.0,
                            "genericity_weight": 0.5,
                            "redundancy_weight": 0.3,
                            "budget_weight": 0.1,
                        },
                        "epsilon": 0.01,
                    },
                    "uncertain": {
                        "fallback_mode": "self_calibrated_constrained",
                        "coverage_constraint": {
                            "mode": "tail_family_relative",
                            "metrics": [
                                {
                                    "name": "coverage_p25",
                                    "relative_ratio": 0.98,
                                    "required": True,
                                    "weight": 0.7,
                                },
                                {
                                    "name": "coverage_mean",
                                    "relative_ratio": 0.998,
                                    "required": True,
                                    "weight": 0.3,
                                },
                            ],
                        },
                    },
                },
            },
        )
        self.assertEqual(result["regime"], "broad_tail")
        self.assertIn(result["resolved_seed_top_k"], [21, 22])

    def test_uncertain_policy_uses_global_fallback(self):
        result = resolve_hierarchical_budget(
            private_lengths=[120, 150, 200],
            metrics_by_budget={
                18: {
                    "coverage_p25": 0.80,
                    "coverage_mean": 0.90,
                    "support_score": 0.90,
                    "support_mean": 0.90,
                    "genericity_score": 0.20,
                    "redundancy_score": 0.20,
                    "budget_cost": 0.0,
                    "utility": 0.60,
                },
                20: {
                    "coverage_p25": 0.91,
                    "coverage_mean": 0.93,
                    "support_score": 0.88,
                    "support_mean": 0.88,
                    "genericity_score": 0.15,
                    "redundancy_score": 0.10,
                    "budget_cost": 0.5,
                    "utility": 0.55,
                },
                22: {
                    "coverage_p25": 0.92,
                    "coverage_mean": 0.94,
                    "support_score": 0.84,
                    "support_mean": 0.84,
                    "genericity_score": 0.14,
                    "redundancy_score": 0.09,
                    "budget_cost": 1.0,
                    "utility": 0.52,
                },
            },
            rule_cfg={
                "router": {
                    "tail_threshold": 350,
                    "short_threshold": 120,
                    "tau_center": 0.0,
                    "delta_router": 10.0,
                    "screening_reference": {
                        "median_len": {"mean": 150.0, "std": 50.0},
                        "p75_len": {"mean": 335.0, "std": 90.0},
                        "iqr_len": {"mean": 180.0, "std": 60.0},
                    },
                },
                "policies": {
                    "broad_tail": {
                        "candidate_seed_top_k": [21, 22],
                        "coverage_p25_ratio": 0.98,
                        "coverage_mean_ratio": 0.998,
                        "epsilon": 0.002,
                    },
                    "compact_structured": {
                        "candidate_seed_top_k": [18, 19, 20],
                        "coverage_p25_ratio": 0.98,
                        "utility": {
                            "support_weight": 1.0,
                            "genericity_weight": 0.5,
                            "redundancy_weight": 0.3,
                            "budget_weight": 0.1,
                        },
                        "epsilon": 0.01,
                    },
                    "uncertain": {
                        "fallback_mode": "self_calibrated_constrained",
                        "coverage_constraint": {
                            "mode": "tail_family_relative",
                            "metrics": [
                                {
                                    "name": "coverage_p25",
                                    "relative_ratio": 0.98,
                                    "required": True,
                                    "weight": 0.7,
                                },
                                {
                                    "name": "coverage_mean",
                                    "relative_ratio": 0.998,
                                    "required": True,
                                    "weight": 0.3,
                                },
                            ],
                        },
                    },
                },
            },
        )
        self.assertEqual(result["regime"], "uncertain")
        self.assertEqual(result["selection_stage"], "uncertain_fallback_policy")

    def test_uncertain_policy_can_use_self_calibrated_fallback_mode(self):
        result = resolve_hierarchical_budget(
            private_lengths=[120, 150, 200],
            metrics_by_budget={
                18: {
                    "coverage_p25": 0.80,
                    "coverage_mean": 0.90,
                    "support_score": 0.90,
                    "support_mean": 0.90,
                    "genericity_score": 0.20,
                    "redundancy_score": 0.20,
                    "budget_cost": 0.0,
                    "utility": 0.60,
                },
                20: {
                    "coverage_p25": 0.81,
                    "coverage_mean": 0.901,
                    "support_score": 0.89,
                    "support_mean": 0.89,
                    "genericity_score": 0.19,
                    "redundancy_score": 0.19,
                    "budget_cost": 0.5,
                    "utility": 0.595,
                },
            },
            rule_cfg={
                "router": {
                    "tail_threshold": 350,
                    "short_threshold": 120,
                    "tau_center": 0.0,
                    "delta_router": 10.0,
                    "screening_reference": {
                        "median_len": {"mean": 150.0, "std": 50.0},
                        "p75_len": {"mean": 335.0, "std": 90.0},
                        "iqr_len": {"mean": 180.0, "std": 60.0},
                    },
                },
                "policies": {
                    "broad_tail": {
                        "candidate_seed_top_k": [21, 22],
                        "coverage_p25_ratio": 0.98,
                        "coverage_mean_ratio": 0.998,
                        "epsilon": 0.002,
                    },
                    "compact_structured": {
                        "candidate_seed_top_k": [18, 19, 20],
                        "coverage_p25_ratio": 0.98,
                        "utility": {
                            "support_weight": 1.0,
                            "genericity_weight": 0.5,
                            "redundancy_weight": 0.3,
                            "budget_weight": 0.1,
                        },
                        "epsilon": 0.01,
                    },
                    "uncertain": {
                        "fallback_mode": "self_calibrated",
                    },
                },
            },
        )
        self.assertEqual(result["regime"], "uncertain")
        self.assertEqual(result["selection_stage"], "uncertain_fallback_policy")
        self.assertEqual(result["resolved_seed_top_k"], 18)

    def test_compact_policy_prefers_smaller_budget_within_epsilon(self):
        result = resolve_hierarchical_budget(
            private_lengths=[60, 75, 95],
            metrics_by_budget={
                18: {
                    "coverage_p25": 0.95,
                    "coverage_mean": 0.97,
                    "support_score": 0.90,
                    "support_mean": 0.90,
                    "genericity_score": 0.20,
                    "redundancy_score": 0.10,
                    "budget_cost": 0.0,
                },
                19: {
                    "coverage_p25": 0.951,
                    "coverage_mean": 0.971,
                    "support_score": 0.899,
                    "support_mean": 0.899,
                    "genericity_score": 0.20,
                    "redundancy_score": 0.10,
                    "budget_cost": 0.25,
                },
            },
            rule_cfg={
                "router": {
                    "tail_threshold": 350,
                    "short_threshold": 120,
                    "tau_center": 0.0,
                    "delta_router": 0.35,
                    "screening_reference": {
                        "median_len": {"mean": 150.0, "std": 50.0},
                        "p75_len": {"mean": 335.0, "std": 90.0},
                        "iqr_len": {"mean": 180.0, "std": 60.0},
                    },
                },
                "policies": {
                    "broad_tail": {
                        "candidate_seed_top_k": [21, 22],
                        "coverage_p25_ratio": 0.98,
                        "coverage_mean_ratio": 0.998,
                        "epsilon": 0.002,
                    },
                    "compact_structured": {
                        "candidate_seed_top_k": [18, 19],
                        "coverage_p25_ratio": 0.98,
                        "utility": {
                            "support_weight": 1.0,
                            "genericity_weight": 0.5,
                            "redundancy_weight": 0.3,
                            "budget_weight": 0.1,
                        },
                        "epsilon": 0.01,
                    },
                    "uncertain": {
                        "fallback_mode": "self_calibrated_constrained",
                        "coverage_constraint": {
                            "mode": "tail_family_relative",
                            "metrics": [
                                {
                                    "name": "coverage_p25",
                                    "relative_ratio": 0.98,
                                    "required": True,
                                    "weight": 0.7,
                                },
                            ],
                        },
                    },
                },
            },
        )
        self.assertEqual(result["regime"], "compact_structured")
        self.assertEqual(result["resolved_seed_top_k"], 18)


if __name__ == "__main__":
    unittest.main()
