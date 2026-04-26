import unittest

from paper_new_selector.genericity import compute_genericity_penalty
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
