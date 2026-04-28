import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from paper_new_selector.pretext_bridge import prepare_bootstrap_runtime


def _pretext_bootstrap_patch():
    pretext_platform = ModuleType("pretext_platform")
    algorithms = ModuleType("pretext_platform.algorithms")
    bootstrap = ModuleType("pretext_platform.algorithms.bootstrap")

    def build_bootstrap_prompts(seed_texts, *, num_prompts, seed):
        del seed
        return list(seed_texts)[:num_prompts]

    def generate_bootstrapped_samples_vllm(prompt_list, model_path, bootstrap_cfg):
        del model_path, bootstrap_cfg
        return list(prompt_list)

    bootstrap.build_bootstrap_prompts = build_bootstrap_prompts
    bootstrap.generate_bootstrapped_samples_vllm = generate_bootstrapped_samples_vllm
    algorithms.bootstrap = bootstrap
    pretext_platform.algorithms = algorithms

    return patch.dict(
        "sys.modules",
        {
            "pretext_platform": pretext_platform,
            "pretext_platform.algorithms": algorithms,
            "pretext_platform.algorithms.bootstrap": bootstrap,
        },
    )


class PretextBridgeTests(unittest.TestCase):
    def test_forums_max_tokens_override_applies_to_bootstrap_runtime(self):
        config = {
            "data": {"dataset_name": "forums"},
            "paths": {"models_root": "models"},
            "selector": {"_forums_max_tokens": 83},
            "bootstrap": {
                "num_prompts": 100,
                "generator_backend": "vllm",
                "generator_model": "llama2_7b",
                "max_tokens": 85,
            },
        }

        with patch(
            "paper_new_selector.pretext_bridge.load_yaml_config",
            return_value=config,
        ), patch(
            "paper_new_selector.pretext_bridge.resolve_repo_root",
            return_value=Path("/tmp/repo"),
        ), patch(
            "paper_new_selector.pretext_bridge._ensure_pretext_importable",
        ), patch(
            "paper_new_selector.pretext_bridge.resolve_bootstrap_model_path",
            return_value=Path("/tmp/repo/models/llama_2_7b_hf"),
        ), _pretext_bootstrap_patch():
            runtime = prepare_bootstrap_runtime("dummy.yaml")

        self.assertEqual(runtime["bootstrap_cfg"]["max_tokens"], 83)


if __name__ == "__main__":
    unittest.main()
