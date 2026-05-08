import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from paper_new_selector.pipeline import run_pipeline
from paper_new_selector.run_selector_single_node import main
from paper_new_selector.thesis_bridge import load_yaml_config


class PipelineSmokeTests(unittest.TestCase):
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

    def test_pipeline_returns_stage1_stage2_boundary_and_generator_contract(self):
        summary = run_pipeline("configs/single_node_jobs_selector.yaml", validate_only=True)
        self.assertIn("stage1", summary)
        self.assertIn("stage2", summary)
        self.assertIn("boundary_state", summary["stage1"])
        self.assertIn("generator_contract", summary)
        self.assertIn("eval", summary)
        self.assertTrue(summary["eval"]["enabled"])
        self.assertEqual(summary["eval"]["mode"], "pretext_small")

    def test_validate_only_reports_vllm_contract_for_both_generation_stages(self):
        summary = run_pipeline("configs/single_node_jobs_selector.yaml", validate_only=True)
        self.assertEqual(summary["generator_contract"]["llm_backend"], "vllm")
        self.assertEqual(summary["stage2"]["bootstrap_cfg"]["generator_backend"], "vllm")
        self.assertEqual(summary["stage1"]["shared_session"]["llm_backend"], "vllm")

    def test_cli_wraps_invalid_non_vllm_config_in_clear_system_exit(self):
        config_path = self._write_temp_config(
            "configs/single_node_jobs_selector.yaml",
            {"llm": {"generator": {"engine": "transformers"}}},
        )
        stdout = io.StringIO()
        with patch("sys.argv", ["run_selector_single_node", "--config", str(config_path)]):
            with patch("sys.stdout", stdout):
                with self.assertRaisesRegex(SystemExit, "strict vLLM generation"):
                    main()

    def test_pipeline_releases_runtime_memory_between_stage2_and_eval(self):
        with patch(
            "paper_new_selector.pipeline.run_stage1_with_runtime",
            return_value=(
                {
                    "generator_contract": {"llm_backend": "vllm"},
                    "selected_texts": ["seed one", "seed two", "seed three"],
                },
                {
                    "generator_handle": type("Handle", (), {"text_backend": object()})(),
                    "shared_session": {"backend": type("B", (), {"generate_batch": lambda self, prompts, **_: [f'generated::{p}' for p in prompts]})()},
                    "embedder": object(),
                },
            ),
        ), patch(
            "paper_new_selector.pipeline.prepare_bootstrap_runtime",
            return_value={
                "bootstrap_cfg": {"num_prompts": 4, "generator_backend": "vllm"},
                "model_path": "local-llama",
                "build_bootstrap_prompts": lambda seed_texts, *, num_prompts, seed: [f"{seed}:{num_prompts}:{len(seed_texts)}"],
                "generate_bootstrapped_samples": lambda prompt_list, _model_path, _bootstrap_cfg: (_ for _ in ()).throw(AssertionError("fresh stage2 path should not run")),
                "generate_with_shared_session": lambda prompt_list, *, shared_session, bootstrap_cfg: shared_session["backend"].generate_batch(
                    prompt_list,
                    max_new_tokens=bootstrap_cfg.get("max_tokens", 85),
                    temperature=bootstrap_cfg.get("temperature", 1.0),
                ),
            },
        ), patch(
            "paper_new_selector.pipeline.prepare_eval_runtime",
            return_value={"enabled": True, "mode": "pretext_small"},
        ), patch(
            "paper_new_selector.pipeline.run_eval",
            return_value={"status": "completed", "kind": "pretext_small_eval"},
        ), patch(
            "paper_new_selector.pipeline.release_runtime_memory",
        ) as release_runtime:
            summary = run_pipeline("configs/single_node_jobs_selector.yaml", validate_only=False)

        self.assertEqual(summary["eval"]["status"], "completed")
        release_runtime.assert_called()
        self.assertEqual(summary["stage2"]["generated_count"], 1)

    def test_pipeline_runs_eval_in_subprocess_after_runtime_release(self):
        fake_eval_summary = {"status": "completed", "kind": "pretext_small_eval"}

        class _Completed:
            def __init__(self, stdout: str):
                self.stdout = stdout
                self.returncode = 0

        with patch(
            "paper_new_selector.pipeline.run_stage1_with_runtime",
            return_value=(
                {
                    "generator_contract": {"llm_backend": "vllm"},
                    "selected_texts": ["seed one", "seed two", "seed three"],
                },
                {
                    "generator_handle": type("Handle", (), {"text_backend": object()})(),
                    "shared_session": {"backend": type("B", (), {"generate_batch": lambda self, prompts, **_: [f'generated::{p}' for p in prompts]})()},
                    "embedder": object(),
                },
            ),
        ), patch(
            "paper_new_selector.pipeline.prepare_bootstrap_runtime",
            return_value={
                "bootstrap_cfg": {"num_prompts": 4, "generator_backend": "vllm"},
                "model_path": "local-llama",
                "build_bootstrap_prompts": lambda seed_texts, *, num_prompts, seed: [f"{seed}:{num_prompts}:{len(seed_texts)}"],
                "generate_bootstrapped_samples": lambda prompt_list, _model_path, _bootstrap_cfg: (_ for _ in ()).throw(AssertionError("fresh stage2 path should not run")),
                "generate_with_shared_session": lambda prompt_list, *, shared_session, bootstrap_cfg: shared_session["backend"].generate_batch(
                    prompt_list,
                    max_new_tokens=bootstrap_cfg.get("max_tokens", 85),
                    temperature=bootstrap_cfg.get("temperature", 1.0),
                ),
            },
        ), patch(
            "paper_new_selector.pipeline.prepare_eval_runtime",
            return_value={"enabled": True, "mode": "pretext_small"},
        ), patch(
            "paper_new_selector.pipeline.run_eval",
            side_effect=AssertionError("eval should run in a subprocess"),
        ), patch(
            "paper_new_selector.pipeline.release_runtime_memory",
        ) as release_runtime, patch(
            "subprocess.run",
            return_value=_Completed(json.dumps(fake_eval_summary)),
        ) as subprocess_run:
            summary = run_pipeline("configs/single_node_jobs_selector.yaml", validate_only=False)

        self.assertEqual(summary["eval"]["status"], "completed")
        release_runtime.assert_called()
        subprocess_run.assert_called_once()

    def test_pipeline_writes_budget_calibration_artifact_for_hierarchical_mode(self):
        with patch(
            "paper_new_selector.pipeline.run_stage1_with_runtime",
            return_value=(
                {
                    "generator_contract": {"llm_backend": "vllm"},
                    "selected_texts": ["seed one"],
                    "seed_budget": {
                        "mode": "hierarchical_shape_routing",
                        "resolved_seed_top_k": 22,
                        "regime": "broad_tail",
                    },
                },
                {
                    "generator_handle": type("Handle", (), {"text_backend": object()})(),
                    "shared_session": {"backend": type("B", (), {"generate_batch": lambda self, prompts, **_: [f'generated::{p}' for p in prompts]})()},
                    "embedder": object(),
                },
            ),
        ), patch(
            "paper_new_selector.pipeline.prepare_bootstrap_runtime",
            return_value={
                "bootstrap_cfg": {"num_prompts": 1, "generator_backend": "vllm"},
                "model_path": "local-llama",
                "build_bootstrap_prompts": lambda seed_texts, *, num_prompts, seed: ["prompt"],
                "generate_bootstrapped_samples": lambda prompt_list, _model_path, _bootstrap_cfg: prompt_list,
                "generate_with_shared_session": lambda prompt_list, *, shared_session, bootstrap_cfg: ["generated::prompt"],
            },
        ), patch(
            "paper_new_selector.pipeline.prepare_eval_runtime",
            return_value={"enabled": False, "mode": "pretext_small"},
        ), patch(
            "paper_new_selector.pipeline.write_json",
        ) as write_json, patch(
            "paper_new_selector.pipeline.release_runtime_memory",
        ):
            run_pipeline("configs/single_node_jobs_selector.yaml", validate_only=False)

        written_names = [call.args[0].name for call in write_json.call_args_list]
        self.assertIn("stage1_summary.json", written_names)
        self.assertIn("stage1_budget_calibration.json", written_names)

    def test_pipeline_falls_back_to_standalone_stage2_generation_without_shared_session(self):
        with patch(
            "paper_new_selector.pipeline.run_stage1_with_runtime",
            return_value=(
                {
                    "generator_contract": {"llm_backend": "none"},
                    "selected_texts": ["seed one", "seed two"],
                    "skip_bootstrap": False,
                    "seed_budget": {"mode": "fixed_public_seed_budget"},
                },
                {
                    "generator_handle": None,
                    "shared_session": None,
                    "embedder": None,
                },
            ),
        ), patch(
            "paper_new_selector.pipeline.prepare_bootstrap_runtime",
            return_value={
                "bootstrap_cfg": {"num_prompts": 2, "generator_backend": "vllm"},
                "model_path": "local-llama",
                "build_bootstrap_prompts": lambda seed_texts, *, num_prompts, seed: ["prompt-a", "prompt-b"],
                "generate_bootstrapped_samples": lambda prompt_list, _model_path, _bootstrap_cfg: [f"fresh::{p}" for p in prompt_list],
                "generate_with_shared_session": lambda prompt_list, *, shared_session, bootstrap_cfg: (_ for _ in ()).throw(
                    AssertionError("shared session path should not run")
                ),
            },
        ), patch(
            "paper_new_selector.pipeline.prepare_eval_runtime",
            return_value={"enabled": False, "mode": "pretext_small"},
        ), patch(
            "paper_new_selector.pipeline.release_runtime_memory",
        ):
            summary = run_pipeline("configs/single_node_jobs_selector.yaml", validate_only=False)

        self.assertEqual(summary["stage2"]["generation_path"], "standalone_bootstrap")
        self.assertEqual(summary["stage2"]["generated_count"], 2)
        self.assertEqual(summary["stage2"]["synthetic_outputs"], ["fresh::prompt-a", "fresh::prompt-b"])


if __name__ == "__main__":
    unittest.main()
