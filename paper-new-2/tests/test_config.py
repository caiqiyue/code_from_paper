import unittest
from pathlib import Path

from paper_new_stage2_selector.thesis_bridge import load_yaml_config


class PaperNew2ConfigTests(unittest.TestCase):
    def test_jobs_screening_config_defines_stage2_seed_aware_contract(self):
        config_path = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "single_node_screening" / "sas_s_jobs_screening.yaml"
        self.assertTrue(config_path.exists())
        config = load_yaml_config(config_path)
        self.assertEqual(config["pipeline"]["stage1_mode"], "pretext_stage1_passthrough")
        self.assertEqual(config["pipeline"]["stage2_mode"], "pretext_bootstrap_seed_aware_selector")
        self.assertEqual(config["selector"]["target_count_mode"], "match_baseline_clean_count")
        self.assertEqual(config["selector"]["consistency_metric"], "max_seed_cosine")
        self.assertEqual(config["bootstrap"]["generator_backend"], "vllm")

    def test_config_matrix_exists_for_screening_and_formal(self):
        root = Path(__file__).resolve().parents[1]
        expected = [
            root / "configs" / "experiments" / "single_node_screening" / "sas_s_jobs_screening.yaml",
            root / "configs" / "experiments" / "single_node_screening" / "sas_s_congressional_screening.yaml",
            root / "configs" / "experiments" / "single_node_screening" / "sas_s_forums_screening.yaml",
            root / "configs" / "experiments" / "single_node_screening" / "sas_s_microblog_screening.yaml",
            root / "configs" / "experiments" / "single_node_formal" / "sas_c1_jobs_base.yaml",
        ]
        for path in expected:
            self.assertTrue(path.exists(), path)


if __name__ == "__main__":
    unittest.main()
