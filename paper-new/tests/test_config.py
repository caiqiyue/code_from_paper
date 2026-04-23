import unittest
from pathlib import Path

import yaml

from paper_new_selector.thesis_bridge import resolve_config_path


class PaperNewSelectorConfigTests(unittest.TestCase):
    def test_config_fully_defines_algorithm_contract(self):
        config_path = resolve_config_path("paper-new/configs/single_node_jobs_selector.yaml")
        self.assertTrue(config_path.exists())
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["pipeline"]["stage1_mode"], "selector_seed_search")
        self.assertEqual(config["pipeline"]["stage2_mode"], "pretext_bootstrap")
        self.assertEqual(config["paths"]["datasets_root"], "thesis_platform/datasets")
        self.assertEqual(config["paths"]["models_root"], "thesis_platform/open_model")
        self.assertEqual(config["generator"]["backend"], "thesis_pretext_prompt")
        self.assertEqual(config["generator"]["candidate_count"], 8)
        self.assertEqual(config["generator"]["max_prompt_chars"], 192)
        self.assertEqual(config["generator"]["max_exemplar_chars"], 220)
        self.assertEqual(config["selector"]["rank_weights"], [1.0, 0.6, 0.3, 0.15])
        self.assertEqual(config["selector"]["private_knn_k"], 8)
        self.assertEqual(config["selector"]["reference_top_k"], 4)
        self.assertEqual(config["selector"]["density_lambda"], 0.50)
        self.assertEqual(config["selector"]["novelty_lambda"], 0.30)
        self.assertEqual(config["selector"]["length_lambda"], 0.20)
        self.assertEqual(config["embedding"]["model_path"], "thesis_platform/open_model/all_minilm_l6_v2")
        self.assertEqual(config["llm"]["generator"]["engine"], "transformers")
        self.assertEqual(config["llm"]["generator"]["model_name_or_path"], "thesis_platform/open_model/distilgpt2")
        self.assertEqual(config["bootstrap"]["generator_backend"], "huggingface")
        self.assertTrue(config["eval"]["enabled"])
        self.assertEqual(config["eval"]["mode"], "pretext_small")

    def test_config_path_resolution_works_from_project_root_and_paper_new_root(self):
        resolved_prefixed = resolve_config_path("paper-new/configs/single_node_jobs_selector.yaml")
        resolved_local = resolve_config_path("configs/single_node_jobs_selector.yaml")
        self.assertEqual(resolved_prefixed, resolved_local)
        self.assertEqual(resolved_local.name, "single_node_jobs_selector.yaml")


if __name__ == "__main__":
    unittest.main()
