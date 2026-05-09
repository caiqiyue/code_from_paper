import unittest

from paper_new_selector.importance import build_private_importance_weights


class ImportanceTests(unittest.TestCase):
    def test_importance_weights_downweight_short_generic_private_samples(self):
        weights = build_private_importance_weights(
            private_vectors=[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
            private_lengths=[32, 28, 2],
            private_knn_k=2,
            density_lambda=0.5,
            novelty_lambda=0.3,
            length_lambda=0.2,
            length_floor=12,
            length_ceiling=128,
        )
        self.assertGreater(weights[0], weights[2])


if __name__ == "__main__":
    unittest.main()
