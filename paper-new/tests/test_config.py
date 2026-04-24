import unittest

from paper_new_selector.thesis_bridge import load_yaml_config, resolve_config_path


class PaperNewSelectorConfigTests(unittest.TestCase):
    def test_config_fully_defines_algorithm_contract(self):
        config_path = resolve_config_path("paper-new/configs/single_node_jobs_selector.yaml")
        self.assertTrue(config_path.exists())
        config = load_yaml_config(config_path)
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
        self.assertEqual(config["llm"]["generator"]["engine"], "vllm")
        self.assertEqual(config["llm"]["generator"]["model_name_or_path"], "thesis_platform/open_model/llama_2_7b_hf")
        self.assertEqual(config["llm"]["generator"]["max_model_len"], 512)
        self.assertEqual(config["llm"]["generator"]["gpu_memory_utilization"], 0.55)
        self.assertEqual(config["llm"]["generator"]["startup_required_free_gb"], 26)
        self.assertEqual(config["llm"]["generator"]["tensor_parallel_size"], 1)
        self.assertTrue(config["llm"]["generator"]["enforce_eager"])
        self.assertEqual(config["bootstrap"]["generator_backend"], "vllm")
        self.assertTrue(config["eval"]["enabled"])
        self.assertEqual(config["eval"]["mode"], "pretext_small")

    def test_config_path_resolution_works_from_project_root_and_paper_new_root(self):
        resolved_prefixed = resolve_config_path("paper-new/configs/single_node_jobs_selector.yaml")
        resolved_local = resolve_config_path("configs/single_node_jobs_selector.yaml")
        self.assertEqual(resolved_prefixed, resolved_local)
        self.assertEqual(resolved_local.name, "single_node_jobs_selector.yaml")

    def test_formal_config_supports_inherits_and_jobs_base_contract(self):
        config = load_yaml_config("paper-new/configs/experiments/single_node_formal/ns_c1_jobs_base.yaml")
        self.assertEqual(config["meta"]["experiment_id"], "ns_c1_jobs_base")
        self.assertEqual(config["meta"]["stage"], "single_node_formal")
        self.assertEqual(config["data"]["dataset_name"], "jobs")
        self.assertEqual(config["bootstrap"]["num_prompts"], 1500)
        self.assertEqual(config["bootstrap"]["generator_backend"], "vllm")
        self.assertEqual(config["bootstrap"]["startup_required_free_gb"], 26)
        self.assertEqual(config["privacy"]["epsilon"], 1.29)
        self.assertEqual(config["stage1"]["sigma"], 11.3)

    def test_formal_config_uses_vllm_for_stage1_and_stage2(self):
        config = load_yaml_config("paper-new/configs/experiments/single_node_formal/ns_c1_jobs_base.yaml")
        self.assertEqual(config["llm"]["generator"]["engine"], "vllm")
        self.assertEqual(config["bootstrap"]["generator_backend"], "vllm")

    def test_smoke_config_uses_real_local_vllm_models_only(self):
        config = load_yaml_config("paper-new/configs/single_node_jobs_selector.yaml")
        self.assertEqual(config["llm"]["generator"]["engine"], "vllm")
        self.assertEqual(config["llm"]["generator"]["model_name_or_path"], "thesis_platform/open_model/llama_2_7b_hf")
        self.assertEqual(config["bootstrap"]["generator_backend"], "vllm")

    def test_vllm_a6000_smoke_config_matches_strict_vllm_contract_with_tiny_scale(self):
        config = load_yaml_config("paper-new/configs/experiments/single_node_formal/ns_test_vllm_a6000.yaml")
        self.assertEqual(config["meta"]["experiment_id"], "ns_test_vllm_a6000")
        self.assertEqual(config["llm"]["generator"]["engine"], "vllm")
        self.assertEqual(config["bootstrap"]["generator_backend"], "vllm")
        self.assertEqual(config["generator"]["candidate_count"], 8)
        self.assertEqual(config["generator"]["generated_per_round"], 2)
        self.assertEqual(config["llm"]["generator"]["max_new_tokens"], 48)
        self.assertEqual(config["bootstrap"]["num_prompts"], 4)
        self.assertEqual(config["eval"]["small_epochs"], 1)
        self.assertEqual(config["data"]["train_limit"], 64)
        self.assertEqual(config["data"]["eval_limit"], 32)
        self.assertEqual(config["data"]["initialization_limit"], 128)

    def test_formal_privacy_variants_and_seed_variants_override_base_values(self):
        eps05 = load_yaml_config("paper-new/configs/experiments/single_node_formal/ns_c5_jobs_eps05.yaml")
        no_privacy = load_yaml_config("paper-new/configs/experiments/single_node_formal/ns_c7_jobs_no_privacy.yaml")
        seed456 = load_yaml_config("paper-new/configs/experiments/single_node_formal/ns_c9_jobs_seed456.yaml")
        self.assertEqual(eps05["privacy"]["epsilon"], 0.5)
        self.assertEqual(eps05["stage1"]["sigma"], 160.0)
        self.assertFalse(no_privacy["privacy"]["enabled"])
        self.assertTrue(no_privacy["stage1"]["privacy_disabled"])
        self.assertEqual(no_privacy["stage1"]["sigma"], 0.0)
        self.assertEqual(seed456["meta"]["seed"], 456)
        self.assertEqual(seed456["paths"]["output_root"], "paper-new/outputs/ns_c9_jobs_seed456")


if __name__ == "__main__":
    unittest.main()
