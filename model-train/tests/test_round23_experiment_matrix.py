from __future__ import annotations

import sys
import unittest
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
MODEL_TRAIN_ROOT = THIS_DIR.parent
if str(MODEL_TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_TRAIN_ROOT))

from round23_experiment_matrix import build_experiment_matrix
from round23_feature_sets import get_feature_spec
from round23_model_zoo import FORMAL_MODEL_FAMILIES


class Round23ExperimentMatrixTests(unittest.TestCase):
    def test_formal_feature_versions_are_explicit(self) -> None:
        with_dataset = get_feature_spec("with_dataset")
        no_dataset = get_feature_spec("no_dataset")

        self.assertTrue(with_dataset.include_dataset_onehot)
        self.assertFalse(no_dataset.include_dataset_onehot)
        self.assertNotEqual(with_dataset.onehot_order, [])
        self.assertEqual(no_dataset.onehot_order, [])
        self.assertEqual(with_dataset.feature_fields, no_dataset.feature_fields)
        self.assertNotIn("best_top1_at_k20", with_dataset.feature_fields)

    def test_formal_experiment_matrix_covers_seven_by_two(self) -> None:
        matrix = build_experiment_matrix()
        self.assertEqual(list(FORMAL_MODEL_FAMILIES), [
            "lightgbm",
            "xgboost",
            "catboost",
            "randomforest",
            "extratrees",
            "mlp",
            "linear_baseline",
        ])
        self.assertEqual(len(matrix), 14)
        seen = {(row["model_family"], row["feature_version"]) for row in matrix}
        self.assertIn(("lightgbm", "with_dataset"), seen)
        self.assertIn(("linear_baseline", "no_dataset"), seen)


if __name__ == "__main__":
    unittest.main()
