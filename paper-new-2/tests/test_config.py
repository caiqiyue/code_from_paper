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
        self.assertEqual(config["selector"]["target_count_mode"], "match_eval_clean_count")
        self.assertEqual(config["selector"]["consistency_metric"], "max_seed_cosine")
        self.assertEqual(config["bootstrap"]["generator_backend"], "vllm")

    def test_screening_base_matches_pretext_and_paper_new_comparison_budget(self):
        config_path = Path(__file__).resolve().parents[1] / "configs" / "base" / "stage2_seed_aware_base.yaml"
        config = load_yaml_config(config_path)
        self.assertEqual(config["data"]["train_limit"], 256)
        self.assertEqual(config["data"]["eval_limit"], 256)
        self.assertEqual(config["data"]["initialization_limit"], 1024)
        self.assertEqual(config["bootstrap"]["num_prompts"], 100)
        self.assertEqual(config["stage1"]["rounds"], 6)
        self.assertEqual(config["stage1"]["lookahead"], 2)
        self.assertEqual(config["stage1"]["multiplier"], 2)
        self.assertEqual(config["stage1"]["batch_size"], 16)
        self.assertEqual(config["stage1"]["embed_batch_size"], 32)
        self.assertEqual(config["eval"]["small_epochs"], 6)
        self.assertEqual(config["eval"]["small_batch_size"], 8)
        self.assertEqual(config["eval"]["small_eval_batch_size"], 2)
        self.assertEqual(config["eval"]["small_grad_accum_steps"], 4)
        self.assertEqual(config["eval"]["max_samples_per_client"], 16)

    def test_formal_base_matches_pretext_and_paper_new_formal_budget(self):
        config_path = Path(__file__).resolve().parents[1] / "configs" / "base" / "stage2_seed_aware_formal_base.yaml"
        config = load_yaml_config(config_path)
        self.assertIsNone(config["data"]["train_limit"])
        self.assertIsNone(config["data"]["eval_limit"])
        self.assertIsNone(config["data"]["initialization_limit"])
        self.assertEqual(config["stage1"]["rounds"], 25)
        self.assertEqual(config["stage1"]["lookahead"], 4)
        self.assertEqual(config["stage1"]["multiplier"], 4)
        self.assertEqual(config["stage1"]["batch_size"], 64)
        self.assertEqual(config["stage1"]["embed_batch_size"], 128)
        self.assertEqual(config["bootstrap"]["num_prompts"], 1500)
        self.assertEqual(config["eval"]["small_epochs"], 20)
        self.assertEqual(config["eval"]["small_grad_accum_steps"], 8)

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
