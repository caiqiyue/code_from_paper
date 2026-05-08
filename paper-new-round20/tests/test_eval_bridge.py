import sys
from pathlib import Path
import unittest

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from paper_new_selector.eval_bridge import _build_thesis_eval_config, prepare_eval_runtime


class EvalBridgeTests(unittest.TestCase):
    def test_eval_runtime_uses_real_pretext_small_eval_contract(self):
        runtime = prepare_eval_runtime("paper-new/configs/single_node_jobs_selector.yaml")
        self.assertTrue(runtime["enabled"])
        self.assertEqual(runtime["mode"], "pretext_small")
        self.assertEqual(runtime["small_eval_mode"], "gpt2")

    def test_formal_eval_config_preserves_none_limits_for_jobs_base(self):
        thesis_config = _build_thesis_eval_config("configs/experiments/single_node_formal/ns_c1_jobs_base.yaml")
        self.assertIsNone(thesis_config.data.get("train_limit"))
        self.assertIsNone(thesis_config.data.get("eval_limit"))
        self.assertIsNone(thesis_config.data.get("initialization_limit"))

    def test_all_formal_configs_build_eval_config_without_none_cast_failures(self):
        configs = [
            "configs/experiments/single_node_formal/ns_c1_jobs_base.yaml",
            "configs/experiments/single_node_formal/ns_c2_congressional_base.yaml",
            "configs/experiments/single_node_formal/ns_c3_forums_base.yaml",
            "configs/experiments/single_node_formal/ns_c4_microblog_base.yaml",
            "configs/experiments/single_node_formal/ns_c5_jobs_eps05.yaml",
            "configs/experiments/single_node_formal/ns_c6_jobs_eps758.yaml",
            "configs/experiments/single_node_formal/ns_c7_jobs_no_privacy.yaml",
            "configs/experiments/single_node_formal/ns_c8_jobs_seed123.yaml",
            "configs/experiments/single_node_formal/ns_c9_jobs_seed456.yaml",
        ]
        for config_path in configs:
            with self.subTest(config_path=config_path):
                thesis_config = _build_thesis_eval_config(config_path)
                self.assertEqual(thesis_config.data.get("train_limit"), None)
                self.assertEqual(thesis_config.data.get("eval_limit"), None)
                self.assertEqual(thesis_config.data.get("initialization_limit"), None)

    def test_screening_eval_config_preserves_dataset_limits_into_pretext_raw(self):
        from thesis_platform.evaluation.downstream_eval import _build_pretext_raw

        thesis_config = _build_thesis_eval_config("configs/experiments/single_node_screening/ns_s_jobs_screening.yaml")
        self.assertEqual(thesis_config.data.get("train_limit"), 256)
        self.assertEqual(thesis_config.data.get("eval_limit"), 256)
        self.assertEqual(thesis_config.data.get("initialization_limit"), 1024)

        raw = _build_pretext_raw(
            thesis_config,
            output_dir=thesis_config.resolve_path("paper-new/outputs/test_eval_bridge"),
            enable_large_eval=False,
            enable_small_eval=True,
        )
        self.assertEqual(raw["data"].get("train_limit"), 256)
        self.assertEqual(raw["data"].get("eval_limit"), 256)
        self.assertEqual(raw["data"].get("initialization_limit"), 1024)


if __name__ == "__main__":
    unittest.main()
