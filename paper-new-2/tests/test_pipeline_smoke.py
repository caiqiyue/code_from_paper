import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paper_new_stage2_selector.pipeline import run_pipeline


class PipelineSmokeTests(unittest.TestCase):
    def test_validate_only_reports_stage2_selector_contract(self):
        summary = run_pipeline(
            "paper-new-2/configs/experiments/single_node_screening/sas_s_jobs_screening.yaml",
            validate_only=True,
        )
        self.assertEqual(summary["stage1_mode"], "pretext_stage1_passthrough")
        self.assertEqual(summary["stage2_mode"], "pretext_bootstrap_seed_aware_selector")
        self.assertEqual(summary["stage2"]["selector"]["target_count_mode"], "match_eval_clean_count")

    def test_pipeline_inserts_selector_between_bootstrap_and_eval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "paper_new_stage2_selector.pipeline.run_pretext_stage1",
                return_value={"stage1_dir": Path("stage1"), "seed_texts": ["seed-a", "seed-b", "seed-c"]},
            ), patch(
                "paper_new_stage2_selector.pipeline.load_yaml_config",
                return_value={
                    "meta": {"seed": 42},
                    "pipeline": {
                        "stage1_mode": "pretext_stage1_passthrough",
                        "stage2_mode": "pretext_bootstrap_seed_aware_selector",
                    },
                    "bootstrap": {"num_prompts": 2},
                    "selector": {
                        "target_count_mode": "match_eval_clean_count",
                        "consistency_threshold": 0.42,
                        "duplicate_threshold": 0.95,
                        "min_words": 4,
                        "prompt_echo_ngram": 8,
                        "unique_token_ratio_floor": 0.45,
                        "w_consistency": 1.0,
                        "w_template": 0.35,
                        "w_duplicate": 0.30,
                    },
                },
            ), patch(
                "paper_new_stage2_selector.pipeline.prepare_bootstrap_runtime",
                return_value={
                    "bootstrap_cfg": {"num_prompts": 2, "generator_backend": "vllm"},
                    "generate_bootstrapped_samples": lambda prompts, _model_path, _cfg: ["good synthetic sample text", "good synthetic sample text"],
                    "model_path": "unused",
                },
            ), patch(
                "paper_new_stage2_selector.pipeline.embed_records",
                return_value=(
                    [[1.0, 0.0], [0.999, 0.001]],
                    [[[1.0, 0.0], [0.8, 0.2], [0.7, 0.3]], [[1.0, 0.0], [0.8, 0.2], [0.7, 0.3]]],
                ),
            ), patch(
                "paper_new_stage2_selector.pipeline.resolve_output_root",
                return_value=Path(tmpdir),
            ), patch(
                "paper_new_stage2_selector.pipeline.run_eval_from_stage2_dir",
                return_value={"enabled": True, "best_top1": 0.3},
            ):
                summary = run_pipeline(
                    "paper-new-2/configs/experiments/single_node_screening/sas_s_jobs_screening.yaml",
                    validate_only=False,
                )
        self.assertEqual(summary["stage2"]["raw_generated_count"], 2)
        self.assertEqual(summary["stage2"]["selected_count"], 1)
        self.assertEqual(summary["eval"]["best_top1"], 0.3)


if __name__ == "__main__":
    unittest.main()
