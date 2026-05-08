import unittest

from paper_new_selector.uncertainty_arbitration import (
    select_uncertain_budget_by_policy_arbitration,
)


class UncertaintyArbitrationTests(unittest.TestCase):
    def test_prefers_higher_coverage_candidate(self):
        broad = {
            "policy_name": "broad_tail",
            "resolved_seed_top_k": 22,
            "coverage_p25": 0.912,
            "coverage_mean": 0.955,
            "support_mean": 0.701,
        }
        compact = {
            "policy_name": "compact_structured",
            "resolved_seed_top_k": 19,
            "coverage_p25": 0.904,
            "coverage_mean": 0.948,
            "support_mean": 0.722,
        }
        selected = select_uncertain_budget_by_policy_arbitration(
            broad_candidate=broad,
            compact_candidate=compact,
            uncertain_cfg={
                "coverage_epsilon": 0.002,
                "support_epsilon": 0.002,
                "prefer_smaller_budget_on_tie": True,
            },
        )
        self.assertEqual(selected["resolved_seed_top_k"], 22)
        self.assertEqual(selected["arbitration_winner_policy"], "broad_tail")
        self.assertEqual(selected["arbitration_reason"], "coverage")

    def test_uses_support_when_coverage_is_close(self):
        broad = {
            "policy_name": "broad_tail",
            "resolved_seed_top_k": 22,
            "coverage_p25": 0.910,
            "coverage_mean": 0.951,
            "support_mean": 0.700,
        }
        compact = {
            "policy_name": "compact_structured",
            "resolved_seed_top_k": 19,
            "coverage_p25": 0.9095,
            "coverage_mean": 0.9505,
            "support_mean": 0.715,
        }
        selected = select_uncertain_budget_by_policy_arbitration(
            broad_candidate=broad,
            compact_candidate=compact,
            uncertain_cfg={
                "coverage_epsilon": 0.002,
                "support_epsilon": 0.002,
                "prefer_smaller_budget_on_tie": True,
            },
        )
        self.assertEqual(selected["resolved_seed_top_k"], 19)
        self.assertEqual(selected["arbitration_winner_policy"], "compact_structured")
        self.assertEqual(selected["arbitration_reason"], "support")

    def test_prefers_smaller_budget_on_full_tie(self):
        broad = {
            "policy_name": "broad_tail",
            "resolved_seed_top_k": 21,
            "coverage_p25": 0.910,
            "coverage_mean": 0.952,
            "support_mean": 0.710,
        }
        compact = {
            "policy_name": "compact_structured",
            "resolved_seed_top_k": 19,
            "coverage_p25": 0.9105,
            "coverage_mean": 0.9515,
            "support_mean": 0.7095,
        }
        selected = select_uncertain_budget_by_policy_arbitration(
            broad_candidate=broad,
            compact_candidate=compact,
            uncertain_cfg={
                "coverage_epsilon": 0.002,
                "support_epsilon": 0.002,
                "prefer_smaller_budget_on_tie": True,
            },
        )
        self.assertEqual(selected["resolved_seed_top_k"], 19)
        self.assertEqual(selected["arbitration_winner_policy"], "compact_structured")
        self.assertEqual(selected["arbitration_reason"], "compactness")


if __name__ == "__main__":
    unittest.main()
