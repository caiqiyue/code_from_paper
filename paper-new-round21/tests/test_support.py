import unittest

from paper_new_selector.genericity import (
    apply_genericity_gate,
    compute_genericity_penalties,
    compute_genericity_penalty,
)
from paper_new_selector.support import apply_gaussian_privacy_noise, compute_private_support


class SupportTests(unittest.TestCase):
    def test_private_support_uses_topq_ranked_votes(self):
        scores = compute_private_support(
            private_vectors=[[1.0, 0.0]],
            candidate_vectors=[[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
            private_weights=[1.0],
            rank_weights=[1.0, 0.6],
            top_q=2,
        )
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])

    def test_genericity_penalty_is_high_for_public_template_like_candidates(self):
        penalty = compute_genericity_penalty(
            candidate_vector=[1.0, 0.0],
            reference_vectors=[[0.99, 0.01], [0.98, 0.02]],
            reference_top_k=2,
        )
        self.assertGreater(penalty, 0.9)

    def test_genericity_penalty_supports_rank_weighted_reference_mean(self):
        penalty = compute_genericity_penalty(
            candidate_vector=[1.0, 0.0],
            reference_vectors=[[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
            reference_top_k=3,
            reference_rank_weights=[1.0, 0.5, 0.1],
        )
        expected = (1.0 * 1.0 + 0.9701425001453318 * 0.5 + 0.0 * 0.1) / (1.0 + 0.5 + 0.1)
        self.assertAlmostEqual(penalty, expected, places=6)

    def test_genericity_penalty_reuses_last_weight_when_topk_exceeds_weight_count(self):
        penalty = compute_genericity_penalty(
            candidate_vector=[1.0, 0.0],
            reference_vectors=[[1.0, 0.0], [0.8, 0.2], [0.6, 0.4]],
            reference_top_k=3,
            reference_rank_weights=[1.0, 0.5],
        )
        expected = (
            1.0 * 1.0
            + 0.9701425001453318 * 0.5
            + 0.8320502943378436 * 0.5
        ) / (1.0 + 0.5 + 0.5)
        self.assertAlmostEqual(penalty, expected, places=6)

    def test_genericity_gate_uses_low_mid_high_scales(self):
        self.assertEqual(
            apply_genericity_gate(
                score=0.70,
                gate_low=0.78,
                gate_high=0.90,
                low_scale=0.10,
                mid_scale=0.45,
            ),
            0.10,
        )
        self.assertEqual(
            apply_genericity_gate(
                score=0.84,
                gate_low=0.78,
                gate_high=0.90,
                low_scale=0.10,
                mid_scale=0.45,
            ),
            0.45,
        )
        self.assertEqual(
            apply_genericity_gate(
                score=0.95,
                gate_low=0.78,
                gate_high=0.90,
                low_scale=0.10,
                mid_scale=0.45,
            ),
            1.0,
        )

    def test_genericity_penalty_applies_gate_to_raw_score(self):
        penalty = compute_genericity_penalty(
            candidate_vector=[1.0, 0.0],
            reference_vectors=[[1.0, 0.0], [0.7, 0.714142842854285], [0.0, 1.0]],
            reference_top_k=3,
            reference_rank_weights=[1.0, 0.5, 0.1],
            gate_low=0.78,
            gate_high=0.90,
            low_scale=0.10,
            mid_scale=0.45,
            apply_gate=True,
        )
        raw_score = (1.0 * 1.0 + 0.7 * 0.5 + 0.0 * 0.1) / (1.0 + 0.5 + 0.1)
        expected = raw_score * 0.45
        self.assertAlmostEqual(penalty, expected, places=6)

    def test_genericity_penalties_accept_length_kwargs_without_changing_disabled_behavior(self):
        baseline = compute_genericity_penalties(
            candidate_vectors=[[1.0, 0.0], [0.0, 1.0]],
            reference_vectors=[[1.0, 0.0], [0.7, 0.714142842854285], [0.0, 1.0]],
            reference_top_k=3,
            reference_rank_weights=[1.0, 0.5, 0.1],
            gate_low=0.78,
            gate_high=0.90,
            low_scale=0.10,
            mid_scale=0.45,
            apply_gate=True,
        )
        length_disabled = compute_genericity_penalties(
            candidate_vectors=[[1.0, 0.0], [0.0, 1.0]],
            reference_vectors=[[1.0, 0.0], [0.7, 0.714142842854285], [0.0, 1.0]],
            reference_top_k=3,
            reference_rank_weights=[1.0, 0.5, 0.1],
            gate_low=0.78,
            gate_high=0.90,
            low_scale=0.10,
            mid_scale=0.45,
            apply_gate=True,
            candidate_lengths=[5, 20],
            length_modulation_enabled=False,
            length_alpha=0.6,
            length_factor_min=0.2,
            length_factor_max=5.0,
        )
        self.assertEqual(length_disabled, baseline)

    def test_genericity_penalties_apply_length_modulation_against_batch_median(self):
        penalties = compute_genericity_penalties(
            candidate_vectors=[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            reference_vectors=[[1.0, 0.0], [0.0, 1.0]],
            reference_top_k=1,
            candidate_lengths=[5, 10, 20],
            length_modulation_enabled=True,
            length_alpha=1.0,
            length_factor_min=0.2,
            length_factor_max=5.0,
        )
        self.assertAlmostEqual(penalties[0], 2.0, places=6)
        self.assertAlmostEqual(penalties[1], 1.0, places=6)
        self.assertAlmostEqual(penalties[2], 0.5, places=6)

    def test_privacy_noise_is_deterministic_for_the_same_seed(self):
        base_scores = [1.0, 2.0, 3.0]
        first = apply_gaussian_privacy_noise(base_scores, enabled=True, sigma=0.5, seed=42)
        second = apply_gaussian_privacy_noise(base_scores, enabled=True, sigma=0.5, seed=42)
        self.assertEqual(first, second)
        self.assertNotEqual(first, base_scores)

    def test_privacy_noise_is_skipped_when_disabled(self):
        base_scores = [1.0, 2.0, 3.0]
        disabled = apply_gaussian_privacy_noise(base_scores, enabled=False, sigma=160.0, seed=42)
        self.assertEqual(disabled, base_scores)


if __name__ == "__main__":
    unittest.main()
