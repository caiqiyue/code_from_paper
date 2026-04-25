import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import yaml

from paper_new_selector.pretext_bridge import (
    _patch_vllm_network_host_ip,
    generate_with_shared_vllm_session,
    prepare_bootstrap_runtime,
    resolve_bootstrap_model_path,
)
from paper_new_selector.thesis_bridge import load_yaml_config, resolve_dataset_paths


class BridgeTests(unittest.TestCase):
    def _deep_merge(self, base: dict, override: dict) -> dict:
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _write_temp_config(self, base_config: str, overrides: dict) -> Path:
        config = self._deep_merge(load_yaml_config(base_config), overrides)
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return path

    def test_thesis_bridge_resolves_existing_dataset_roots(self):
        train_path, eval_path, init_path = resolve_dataset_paths("configs/single_node_jobs_selector.yaml")
        self.assertIn("thesis_platform/datasets", train_path.as_posix())
        self.assertIn("thesis_platform/datasets", eval_path.as_posix())
        self.assertIn("thesis_platform/datasets", init_path.as_posix())

    def test_pretext_bridge_uses_existing_open_model_root(self):
        model_path = resolve_bootstrap_model_path("thesis_platform/open_model", "llama2_7b")
        self.assertIn("thesis_platform/open_model", model_path.as_posix())

    def test_pretext_bridge_prepares_real_bootstrap_call_contract(self):
        runtime = prepare_bootstrap_runtime("configs/single_node_jobs_selector.yaml")
        self.assertTrue(callable(runtime["build_bootstrap_prompts"]))
        self.assertTrue(callable(runtime["generate_bootstrapped_samples"]))
        self.assertEqual(runtime["bootstrap_cfg"]["generator_backend"], "vllm")
        self.assertEqual(runtime["bootstrap_cfg"]["max_model_len"], 512)
        self.assertEqual(runtime["bootstrap_cfg"]["gpu_memory_utilization"], 0.35)
        self.assertEqual(runtime["bootstrap_cfg"]["tensor_parallel_size"], 1)
        self.assertEqual(runtime["bootstrap_cfg"]["temperature"], 1.0)
        self.assertEqual(runtime["bootstrap_cfg"]["top_p"], 1.0)
        self.assertEqual(runtime["bootstrap_cfg"]["max_tokens"], 32)
        self.assertTrue(runtime["bootstrap_cfg"]["enforce_eager"])
        self.assertEqual(runtime["bootstrap_cfg"]["startup_required_free_gb"], 2)

    def test_pretext_bridge_rejects_non_vllm_backend(self):
        config_path = self._write_temp_config(
            "configs/single_node_jobs_selector.yaml",
            {"bootstrap": {"generator_backend": "huggingface"}},
        )
        with self.assertRaisesRegex(ValueError, "bootstrap.generator_backend='vllm'"):
            prepare_bootstrap_runtime(str(config_path))

    def test_pretext_bridge_wraps_stage2_generation_with_vllm_host_ip_patch(self):
        fake_bootstrap_module = ModuleType("pretext_platform.algorithms.bootstrap")
        patch_calls = []

        def fake_build_bootstrap_prompts(seed_texts, *, num_prompts, seed):
            return [f"{len(seed_texts)}:{num_prompts}:{seed}"]

        def fake_generate(prompt_list, model_path, bootstrap_cfg):
            return [f"{prompt_list[0]}::{model_path.name}::{bootstrap_cfg['generator_backend']}"]

        fake_bootstrap_module.build_bootstrap_prompts = fake_build_bootstrap_prompts
        fake_bootstrap_module.generate_bootstrapped_samples_vllm = fake_generate
        config_path = self._write_temp_config("configs/single_node_jobs_selector.yaml", {})

        with patch.dict(
            "sys.modules",
            {"pretext_platform.algorithms.bootstrap": fake_bootstrap_module},
        ), patch(
            "paper_new_selector.pretext_bridge._ensure_pretext_importable",
            return_value=None,
        ), patch(
            "paper_new_selector.pretext_bridge._patch_vllm_network_host_ip",
            side_effect=lambda: patch_calls.append("patched"),
        ):
            runtime = prepare_bootstrap_runtime(str(config_path))
            outputs = runtime["generate_bootstrapped_samples"](["prompt"], Path("demo-model"), {"generator_backend": "vllm"})

        self.assertEqual(patch_calls, ["patched"])
        self.assertEqual(outputs, ["prompt::demo-model::vllm"])

    def test_shared_stage2_generation_uses_existing_backend_batch_generate(self):
        class _FakeBackend:
            def __init__(self):
                self.calls = []

            def generate_batch(self, prompts, *, max_new_tokens, temperature=None):
                self.calls.append((list(prompts), max_new_tokens, temperature))
                return ["alpha", "beta"]

        backend = _FakeBackend()
        outputs = generate_with_shared_vllm_session(
            ["p1", "p2"],
            shared_session={"backend": backend},
            bootstrap_cfg={"max_tokens": 85, "temperature": 1.0},
        )
        self.assertEqual(outputs, ["alpha", "beta"])
        self.assertEqual(backend.calls, [(["p1", "p2"], 85, 1.0)])

    def test_pretext_bridge_defaults_vllm_host_to_loopback_without_env_override(self):
        fake_vllm = ModuleType("vllm")
        fake_utils = ModuleType("vllm.utils")
        fake_utils.get_ip = lambda: "dns.google"
        fake_engine = ModuleType("vllm.engine")
        fake_llm_engine = ModuleType("vllm.engine.llm_engine")
        fake_llm_engine.get_ip = lambda: "dns.google"
        fake_vllm.utils = fake_utils
        fake_vllm.engine = fake_engine
        fake_engine.llm_engine = fake_llm_engine

        with patch.dict(
            "sys.modules",
            {
                "vllm": fake_vllm,
                "vllm.utils": fake_utils,
                "vllm.engine": fake_engine,
                "vllm.engine.llm_engine": fake_llm_engine,
            },
        ), patch.dict("os.environ", {}, clear=True):
            host_ip = _patch_vllm_network_host_ip()

        self.assertEqual(host_ip, "127.0.0.1")
        self.assertEqual(fake_utils.get_ip(), "127.0.0.1")
        self.assertEqual(fake_llm_engine.get_ip(), "127.0.0.1")

    def test_pretext_bridge_honors_explicit_vllm_host_override(self):
        fake_vllm = ModuleType("vllm")
        fake_utils = ModuleType("vllm.utils")
        fake_utils.get_ip = lambda: "dns.google"
        fake_engine = ModuleType("vllm.engine")
        fake_llm_engine = ModuleType("vllm.engine.llm_engine")
        fake_llm_engine.get_ip = lambda: "dns.google"
        fake_vllm.utils = fake_utils
        fake_vllm.engine = fake_engine
        fake_engine.llm_engine = fake_llm_engine

        with patch.dict(
            "sys.modules",
            {
                "vllm": fake_vllm,
                "vllm.utils": fake_utils,
                "vllm.engine": fake_engine,
                "vllm.engine.llm_engine": fake_llm_engine,
            },
        ), patch.dict("os.environ", {"HOST_IP": "10.0.0.8"}, clear=True):
            host_ip = _patch_vllm_network_host_ip()

        self.assertEqual(host_ip, "10.0.0.8")
        self.assertEqual(fake_utils.get_ip(), "10.0.0.8")
        self.assertEqual(fake_llm_engine.get_ip(), "10.0.0.8")


if __name__ == "__main__":
    unittest.main()
