import unittest

from paper_new_stage2_selector.consistency import compute_consistency_score


class ConsistencyTests(unittest.TestCase):
    def test_consistency_prefers_text_close_to_any_prompt_seed(self):
        score = compute_consistency_score(
            generated_vector=[1.0, 0.0],
            seed_vectors=[[0.99, 0.01], [0.0, 1.0], [0.2, 0.8]],
        )
        self.assertGreater(score, 0.95)


if __name__ == "__main__":
    unittest.main()
