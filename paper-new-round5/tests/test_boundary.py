import unittest

from paper_new_selector.boundary import build_boundary_state


class BoundaryTests(unittest.TestCase):
    def test_boundary_state_keeps_more_than_raw_negative_texts(self):
        boundary = build_boundary_state(
            reject_scores=[0.11, 0.08, 0.03],
            reject_vectors=[[1.0, 0.0], [0.95, 0.05], [0.9, 0.1]],
        )
        self.assertIn("reject_score_ceiling", boundary)
        self.assertIn("reject_score_floor", boundary)
        self.assertIn("negative_centroid", boundary)


if __name__ == "__main__":
    unittest.main()
