import unittest

from paper_new_selector.stage1_runner import (
    compute_private_length_stats,
    resolve_seed_top_k,
)


class SeedBudgetRuleTests(unittest.TestCase):
    def test_disabled_rule_keeps_configured_seed_top_k(self):
        self.assertEqual(
            resolve_seed_top_k(
                {"seed_top_k": 21, "seed_budget_rule": {"enabled": False}},
                [100, 200, 300],
            ),
            21,
        )

    def test_enabled_rule_with_no_lengths_keeps_configured_seed_top_k(self):
        self.assertEqual(
            resolve_seed_top_k(
                {"seed_top_k": 20, "seed_budget_rule": {"enabled": True}},
                [],
            ),
            20,
        )

    def test_short_structured_lengths_resolve_to_19(self):
        lengths = [80, 90, 103, 110, 186, 240]
        self.assertEqual(
            resolve_seed_top_k(
                {"seed_top_k": 20, "seed_budget_rule": {"enabled": True}},
                lengths,
            ),
            19,
        )

    def test_broad_mixed_lengths_resolve_to_22(self):
        lengths = [150, 180, 190, 250, 396, 520]
        self.assertEqual(
            resolve_seed_top_k(
                {
                    "seed_top_k": 20,
                    "seed_budget_rule": {"enabled": True, "mode": "length_family"},
                },
                lengths,
            ),
            22,
        )

    def test_long_social_lengths_resolve_to_18(self):
        lengths = [100, 186, 186, 374, 900]
        self.assertEqual(
            resolve_seed_top_k(
                {"seed_top_k": 20, "seed_budget_rule": {"enabled": True}},
                lengths,
            ),
            18,
        )

    def test_fallback_lengths_resolve_to_20(self):
        lengths = [130, 150, 170, 250, 300, 330]
        self.assertEqual(
            resolve_seed_top_k(
                {"seed_top_k": 20, "seed_budget_rule": {"enabled": True}},
                lengths,
            ),
            20,
        )

    def test_round15_observed_stats_resolve_to_round14_success_budgets(self):
        cfg = {"seed_top_k": 20, "seed_budget_rule": {"enabled": True}}
        cases = [
            ([130, 177, 178, 300, 349], 20),
            ([80, 99, 99, 173, 498], 19),
            ([100, 203, 204, 396, 500], 22),
            ([100, 186, 186, 374, 900], 18),
        ]
        for lengths, expected in cases:
            with self.subTest(lengths=lengths):
                self.assertEqual(resolve_seed_top_k(cfg, lengths), expected)

    def test_unsupported_mode_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Unsupported seed_budget_rule.mode"):
            resolve_seed_top_k(
                {
                    "seed_top_k": 20,
                    "seed_budget_rule": {"enabled": True, "mode": "dataset_name"},
                },
                [100, 200, 300],
            )

    def test_length_stats_use_nearest_rank_p75(self):
        self.assertEqual(
            compute_private_length_stats([10, 20, 30, 40])["p75"],
            30.0,
        )


if __name__ == "__main__":
    unittest.main()
