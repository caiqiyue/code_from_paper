from __future__ import annotations

import unittest
from pathlib import Path

from paper_new_selector.thesis_bridge import load_yaml_config


class DiagnosisConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.diag_root = cls.root / "configs" / "experiments" / "single_node_diagnosis"

    def test_genericity_off_variant_loads(self):
        config = load_yaml_config(self.diag_root / "_variant_genericity_off.yaml")
        self.assertEqual(config["meta"]["stage"], "single_node_diagnosis")
        self.assertEqual(config["meta"]["diagnosis_variant"], "genericity_off")
        self.assertEqual(config["selector"]["lambda_generic"], 0.0)

    def test_all_diagnosis_configs_resolve(self):
        names = [
            "ns_d1_jobs.yaml",
            "ns_d1_congressional.yaml",
            "ns_d1_forums.yaml",
            "ns_d1_microblog.yaml",
            "ns_d2_jobs.yaml",
            "ns_d2_congressional.yaml",
            "ns_d2_forums.yaml",
            "ns_d2_microblog.yaml",
            "ns_d3_jobs.yaml",
            "ns_d3_congressional.yaml",
            "ns_d3_forums.yaml",
            "ns_d3_microblog.yaml",
        ]
        for name in names:
            config = load_yaml_config(self.diag_root / name)
            self.assertEqual(config["data"]["train_limit"], 256)
            self.assertEqual(config["data"]["eval_limit"], 256)
            self.assertEqual(config["data"]["initialization_limit"], 1024)
            self.assertEqual(config["bootstrap"]["num_prompts"], 100)
            self.assertEqual(config["eval"]["small_epochs"], 6)
            self.assertTrue(str(config["paths"]["output_root"]).startswith("paper-new-xiugai/outputs/"))

    def test_d1_genericity_off_overrides_lambda_generic(self):
        config = load_yaml_config(self.diag_root / "ns_d1_forums.yaml")
        self.assertEqual(config["selector"]["lambda_generic"], 0.0)
        self.assertEqual(config["selector"]["lambda_redundancy"], 0.25)

    def test_d2_redundancy_up_overrides_lambda_redundancy(self):
        config = load_yaml_config(self.diag_root / "ns_d2_jobs.yaml")
        self.assertEqual(config["selector"]["lambda_redundancy"], 0.45)
        self.assertEqual(config["selector"]["lambda_generic"], 0.35)

    def test_d3_support_softened_overrides_support_shape(self):
        config = load_yaml_config(self.diag_root / "ns_d3_microblog.yaml")
        self.assertEqual(config["selector"]["top_q"], 2)
        self.assertEqual(config["selector"]["rank_weights"], [1.0, 0.4])
        self.assertEqual(config["selector"]["density_lambda"], 0.35)
        self.assertEqual(config["selector"]["novelty_lambda"], 0.45)
        self.assertEqual(config["selector"]["length_lambda"], 0.20)


if __name__ == "__main__":
    unittest.main()
