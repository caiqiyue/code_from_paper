import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paper_new_selector.pipeline import run_pipeline


class PipelineWriteTests(unittest.TestCase):
    def test_run_pipeline_writes_stage1_summary_and_budget_calibration_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            config = {
                "pipeline": {"stage1_mode": "selector_seed_search", "stage2_mode": "bootstrap"},
                "meta": {"seed": 42},
            }
            stage1_summary = {
                "selected_texts": ["seed alpha"],
                "generator_contract": {"llm_backend": "vllm"},
                "seed_budget": {
                    "resolved_seed_top_k": 19,
                    "mode": "self_calibrated",
                },
            }
            stage1_runtime = {
                "generator_handle": None,
                "shared_session": None,
                "embedder": None,
            }
            bootstrap_runtime = {
                "bootstrap_cfg": {"num_prompts": 1},
                "model_path": "fake-model",
                "build_bootstrap_prompts": lambda selected_texts, num_prompts, seed: selected_texts,
                "generate_bootstrapped_samples": lambda *args, **kwargs: [],
                "generate_with_shared_session": lambda *args, **kwargs: ["synthetic alpha"],
            }
            eval_runtime = {"enabled": False}

            with patch("paper_new_selector.pipeline.load_yaml_config", return_value=config), patch(
                "paper_new_selector.pipeline.run_stage1_with_runtime",
                return_value=(stage1_summary, stage1_runtime),
            ), patch(
                "paper_new_selector.pipeline.prepare_bootstrap_runtime",
                return_value=bootstrap_runtime,
            ), patch(
                "paper_new_selector.pipeline.prepare_eval_runtime",
                return_value=eval_runtime,
            ), patch(
                "paper_new_selector.pipeline.resolve_output_root",
                return_value=output_root,
            ), patch(
                "paper_new_selector.pipeline.release_runtime_memory",
            ):
                run_pipeline("dummy.yaml", validate_only=False)

            self.assertTrue((output_root / "stage1_summary.json").exists())
            self.assertTrue((output_root / "stage1_budget_calibration.json").exists())

    def test_run_pipeline_skips_budget_calibration_file_for_non_self_calibrated_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            config = {
                "pipeline": {"stage1_mode": "selector_seed_search", "stage2_mode": "bootstrap"},
                "meta": {"seed": 42},
            }
            stage1_summary = {
                "selected_texts": ["seed alpha"],
                "generator_contract": {"llm_backend": "vllm"},
                "seed_budget": {
                    "resolved_seed_top_k": 19,
                    "mode": "length_family",
                },
            }
            stage1_runtime = {
                "generator_handle": None,
                "shared_session": None,
                "embedder": None,
            }
            bootstrap_runtime = {
                "bootstrap_cfg": {"num_prompts": 1},
                "model_path": "fake-model",
                "build_bootstrap_prompts": lambda selected_texts, num_prompts, seed: selected_texts,
                "generate_bootstrapped_samples": lambda *args, **kwargs: [],
                "generate_with_shared_session": lambda *args, **kwargs: ["synthetic alpha"],
            }
            eval_runtime = {"enabled": False}

            with patch("paper_new_selector.pipeline.load_yaml_config", return_value=config), patch(
                "paper_new_selector.pipeline.run_stage1_with_runtime",
                return_value=(stage1_summary, stage1_runtime),
            ), patch(
                "paper_new_selector.pipeline.prepare_bootstrap_runtime",
                return_value=bootstrap_runtime,
            ), patch(
                "paper_new_selector.pipeline.prepare_eval_runtime",
                return_value=eval_runtime,
            ), patch(
                "paper_new_selector.pipeline.resolve_output_root",
                return_value=output_root,
            ), patch(
                "paper_new_selector.pipeline.release_runtime_memory",
            ):
                run_pipeline("dummy.yaml", validate_only=False)

            self.assertTrue((output_root / "stage1_summary.json").exists())
            self.assertFalse((output_root / "stage1_budget_calibration.json").exists())


if __name__ == "__main__":
    unittest.main()
