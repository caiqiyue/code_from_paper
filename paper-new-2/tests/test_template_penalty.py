import unittest

from paper_new_stage2_selector.dedup import compute_duplicate_penalty
from paper_new_stage2_selector.template_penalty import compute_template_penalty


class TemplatePenaltyTests(unittest.TestCase):
    def test_template_penalty_hits_prompt_echo_and_low_diversity(self):
        penalty = compute_template_penalty(
            text="List of 3 diverse original text samples original text sample original text sample",
            prompt_text="List of 3 diverse original text samples Original Text Sample 1 alpha",
            seed_texts=["alpha", "beta", "gamma"],
            min_words=4,
            prompt_echo_ngram=6,
            unique_token_ratio_floor=0.45,
        )
        self.assertGreaterEqual(penalty, 1.0)

    def test_duplicate_penalty_is_high_for_near_duplicate_vectors(self):
        penalty = compute_duplicate_penalty(
            candidate_vector=[1.0, 0.0],
            kept_vectors=[[0.999, 0.001], [0.0, 1.0]],
        )
        self.assertGreater(penalty, 0.95)


if __name__ == "__main__":
    unittest.main()
