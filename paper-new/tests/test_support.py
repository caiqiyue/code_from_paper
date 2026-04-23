import unittest

from paper_new_selector.genericity import compute_genericity_penalty
from paper_new_selector.support import compute_private_support


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


if __name__ == "__main__":
    unittest.main()
