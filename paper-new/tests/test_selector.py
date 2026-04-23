import unittest

from paper_new_selector.selector import greedy_select_candidates


class SelectorDecisionTests(unittest.TestCase):
    def test_redundancy_penalty_prevents_duplicate_seeds_and_marks_boundary_negative(self):
        result = greedy_select_candidates(
            candidate_vectors=[[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]],
            private_support=[1.0, 0.98, 0.92],
            genericity_penalty=[0.1, 0.1, 0.1],
            lambda_generic=0.35,
            lambda_redundancy=0.25,
            seed_top_k=2,
            hard_negative_top_k=1,
        )
        self.assertEqual(result.selected_indices, [0, 2])
        self.assertEqual(result.hard_negative_indices, [1])
        self.assertEqual(result.hard_negative_reason[1], "near_boundary_rejected")

    def test_rejected_candidates_are_rescored_against_final_selected_seed_set(self):
        result = greedy_select_candidates(
            candidate_vectors=[[1.0, 0.0], [0.0, 1.0], [0.0, 0.99]],
            private_support=[1.0, 0.95, 0.94],
            genericity_penalty=[0.0, 0.0, 0.0],
            lambda_generic=0.0,
            lambda_redundancy=1.0,
            seed_top_k=2,
            hard_negative_top_k=1,
        )
        self.assertEqual(result.selected_indices, [0, 1])
        self.assertEqual(result.hard_negative_indices, [2])
        self.assertGreater(result.candidate_records[2].redundancy_penalty, 0.9)
        self.assertLess(result.accept_scores[2], result.base_scores[2])


if __name__ == "__main__":
    unittest.main()
