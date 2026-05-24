from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
MODEL_TRAIN_ROOT = THIS_DIR.parent
if str(MODEL_TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_TRAIN_ROOT))

from scripts.export_round23_controller_bundle import export_round23_controller_bundle


class Round23ControllerBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="round23_controller_bundle_"))
        self.model_dir = self.tmpdir / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        for suffix in ("neg2", "neg1", "0", "pos1", "pos2"):
            (self.model_dir / f"model_dk_{suffix}.txt").write_text(f"dummy-{suffix}", encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_export_bundle_writes_schema_metadata_and_model_files(self) -> None:
        bundle_dir = self.tmpdir / "bundle"
        report = export_round23_controller_bundle(
            trained_model_dir=self.model_dir,
            output_dir=bundle_dir,
            bundle_version="round23_test_bundle",
            model_family="lightgbm",
            training_data_version="controller-fixture-v1",
            feature_version="no_dataset",
            feature_names=[
                "shape_score",
                "private_mean_length",
                "private_p75_length",
                "private_length_iqr",
            ],
            include_dataset_onehot=False,
            onehot_order=[],
        )
        self.assertEqual(report["bundle_version"], "round23_test_bundle")
        feature_schema = json.loads((bundle_dir / "feature_schema.json").read_text(encoding="utf-8"))
        metadata = json.loads((bundle_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(feature_schema["version"], "round23_controller_feature_schema_v1")
        self.assertIn("feature_names", feature_schema)
        self.assertIn("total_features", feature_schema)
        self.assertFalse(feature_schema["include_dataset_onehot"])
        self.assertEqual(feature_schema["feature_version"], "no_dataset")
        self.assertEqual(feature_schema["onehot_order"], [])
        self.assertEqual(feature_schema["total_features"], 4)
        self.assertEqual(metadata["learner_family"], "lightgbm")
        self.assertEqual(metadata["action_space"], [-2, -1, 0, 1, 2])
        self.assertEqual(metadata["model_family"], "lightgbm")
        self.assertEqual(metadata["feature_version"], "no_dataset")
        self.assertFalse(metadata["include_dataset_onehot"])
        self.assertTrue((bundle_dir / "feature_schema.json").exists())
        self.assertTrue((bundle_dir / "model_dk_neg2.txt").exists())
        self.assertTrue((bundle_dir / "model_dk_pos2.txt").exists())

    def test_export_bundle_records_controller_scope_audit_metadata(self) -> None:
        bundle_dir = self.tmpdir / "bundle_audit"
        report = export_round23_controller_bundle(
            trained_model_dir=self.model_dir,
            output_dir=bundle_dir,
            bundle_version="round23_4seen_test_bundle",
            model_family="extratrees",
            training_data_version="controller-fixture-v2",
            feature_version="no_dataset",
            feature_names=["shape_score"],
            include_dataset_onehot=False,
            onehot_order=[],
            controller_scope="4seen",
            included_datasets=["jobs", "congressional", "forums", "microblog"],
            excluded_datasets=["imdb", "openreview"],
            record_count=800,
            training_data_hash="sha256:abc123",
            selection_protocol="frozen_pre_e2_protocol",
        )

        metadata = json.loads((bundle_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["controller_scope"], "4seen")
        self.assertEqual(metadata["included_datasets"], ["jobs", "congressional", "forums", "microblog"])
        self.assertEqual(metadata["excluded_datasets"], ["imdb", "openreview"])
        self.assertEqual(metadata["record_count"], 800)
        self.assertEqual(metadata["training_data_hash"], "sha256:abc123")
        self.assertEqual(metadata["selection_protocol"], "frozen_pre_e2_protocol")
        self.assertEqual(report["controller_scope"], "4seen")
        self.assertEqual(report["record_count"], 800)


if __name__ == "__main__":
    unittest.main()
