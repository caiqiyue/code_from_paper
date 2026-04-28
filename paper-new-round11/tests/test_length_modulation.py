import math
import unittest

from paper_new_selector.genericity import compute_length_factors


class LengthFactorTests(unittest.TestCase):
    def test_length_factor_neutral_when_alpha_zero(self):
        factors = compute_length_factors(
            lengths=[5, 10, 20, 50],
            alpha=0.0,
            l_ref_strategy="batch_median",
            factor_min=0.2,
            factor_max=5.0,
        )
        self.assertEqual(len(factors), 4)
        for factor in factors:
            self.assertAlmostEqual(factor, 1.0, places=9)

    def test_length_factor_protects_longer_when_alpha_positive(self):
        factors = compute_length_factors(
            lengths=[5, 10, 20, 50],
            alpha=0.3,
            l_ref_strategy="batch_median",
            factor_min=0.01,
            factor_max=100.0,
        )
        # batch median of [5, 10, 20, 50] = (10 + 20) / 2 = 15
        self.assertGreater(factors[0], factors[1])
        self.assertGreater(factors[1], factors[2])
        self.assertGreater(factors[2], factors[3])
        self.assertAlmostEqual(factors[0], 3.0 ** 0.3, places=6)
        self.assertAlmostEqual(factors[3], 0.3 ** 0.3, places=6)

    def test_length_factor_protects_shorter_when_alpha_negative(self):
        factors = compute_length_factors(
            lengths=[5, 10, 20, 50],
            alpha=-0.3,
            l_ref_strategy="batch_median",
            factor_min=0.01,
            factor_max=100.0,
        )
        self.assertLess(factors[0], factors[1])
        self.assertLess(factors[1], factors[2])
        self.assertLess(factors[2], factors[3])
        self.assertAlmostEqual(factors[0], (5/15) ** 0.3, places=6)

    def test_length_factor_clipped_to_min_max(self):
        factors_short = compute_length_factors(
            lengths=[1, 100, 100],
            alpha=0.6,
            l_ref_strategy="batch_median",
            factor_min=0.2,
            factor_max=5.0,
        )
        self.assertAlmostEqual(factors_short[0], 5.0, places=9)

        factors_long = compute_length_factors(
            lengths=[10, 10, 1000],
            alpha=0.6,
            l_ref_strategy="batch_median",
            factor_min=0.2,
            factor_max=5.0,
        )
        self.assertAlmostEqual(factors_long[2], 0.2, places=9)

    def test_genericity_with_length_disabled_matches_round4(self):
        from paper_new_selector.genericity import compute_genericity_penalties

        candidate_vectors = [
            [1.0, 0.0],
            [0.7, 0.714142842854285],
            [0.0, 1.0],
        ]
        reference_vectors = [[1.0, 0.0], [0.99, 0.01], [0.98, 0.02]]
        common_kwargs = dict(
            candidate_vectors=candidate_vectors,
            reference_vectors=reference_vectors,
            reference_top_k=3,
            reference_rank_weights=[1.0, 0.5, 0.1],
            apply_gate=True,
            gate_low=0.78,
            gate_high=0.90,
            low_scale=0.10,
            mid_scale=0.45,
        )

        baseline = compute_genericity_penalties(**common_kwargs)

        with_disabled = compute_genericity_penalties(
            **common_kwargs,
            candidate_lengths=[5, 10, 20],
            length_modulation_enabled=False,
            length_alpha=0.6,
            length_factor_min=0.2,
            length_factor_max=5.0,
        )

        with_alpha_zero = compute_genericity_penalties(
            **common_kwargs,
            candidate_lengths=[5, 10, 20],
            length_modulation_enabled=True,
            length_alpha=0.0,
            length_factor_min=0.2,
            length_factor_max=5.0,
        )

        for i in range(3):
            self.assertAlmostEqual(baseline[i], with_disabled[i], places=12)
            self.assertAlmostEqual(baseline[i], with_alpha_zero[i], places=12)


if __name__ == "__main__":
    unittest.main()
