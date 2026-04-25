import tempfile
import unittest
from pathlib import Path

import yaml

from paper_new_selector.generator_bridge import build_candidate_generator
from paper_new_selector.thesis_bridge import load_yaml_config


class GeneratorBridgeTests(unittest.TestCase):
    def _deep_merge(self, base: dict, override: dict) -> dict:
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _write_temp_config(self, base_config: str, overrides: dict) -> Path:
        config = load_yaml_config(base_config)
        config = self._deep_merge(config, overrides)
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return path

    def test_generator_bridge_uses_one_fixed_generator_source(self):
        generator = build_candidate_generator("configs/single_node_jobs_selector.yaml")
        self.assertEqual(generator.contract["backend"], "thesis_pretext_prompt")
        self.assertIn("pretext_prompt_generator", generator.contract["source"])
        self.assertEqual(generator.contract["max_prompt_chars"], 192)
        self.assertEqual(generator.contract["max_exemplar_chars"], 220)

    def test_stage1_generator_rejects_non_vllm_engine_for_real_configs(self):
        config_path = self._write_temp_config(
            "configs/single_node_jobs_selector.yaml",
            {"llm": {"generator": {"engine": "transformers"}}},
        )
        with self.assertRaisesRegex(ValueError, "Stage 1 candidate generation requires llm.generator.engine='vllm'"):
            build_candidate_generator(str(config_path))

    def test_stage1_generator_contract_reports_vllm_runtime_fields(self):
        handle = build_candidate_generator("configs/single_node_jobs_selector.yaml")
        self.assertEqual(handle.contract["llm_backend"], "vllm")
        self.assertEqual(handle.contract["max_model_len"], 512)
        self.assertEqual(handle.contract["gpu_memory_utilization"], 0.35)
        self.assertEqual(handle.contract["tensor_parallel_size"], 1)
        self.assertTrue(handle.contract["enforce_eager"])
        self.assertIsNotNone(handle.shared_session)
        self.assertEqual(handle.shared_session.to_dict()["llm_backend"], "vllm")

    def test_stage1_generator_runtime_uses_llm_generator_sampling_values_as_single_source(self):
        config_path = self._write_temp_config(
            "configs/single_node_jobs_selector.yaml",
            {
                "generator": {
                    "temperature": 0.1,
                    "max_new_tokens": 12,
                },
                "llm": {
                    "generator": {
                        "temperature": 0.9,
                        "max_new_tokens": 55,
                    }
                },
            },
        )
        handle = build_candidate_generator(str(config_path))
        self.assertEqual(handle.generator.temperature, 0.9)
        self.assertEqual(handle.generator.max_new_tokens, 55)

    def test_generator_handle_exposes_shared_vllm_backend_methods(self):
        handle = build_candidate_generator("configs/single_node_jobs_selector.yaml")
        self.assertTrue(hasattr(handle.text_backend, "ensure_session"))
        self.assertTrue(hasattr(handle.text_backend, "generate_batch"))


if __name__ == "__main__":
    unittest.main()
