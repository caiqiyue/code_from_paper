from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pretext_platform.core.config import ExperimentConfig
from pretext_platform.core.preflight import run_preflight
from pretext_platform.evaluation.glue_classification_eval import validate_local_glue_datasets
from pretext_platform.scripts.run_experiments import _write_runtime_config


def _touch_json(path: Path, payload: str = "[]") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _build_config(root: Path, *, stage1: bool = True, stage2: bool = True, eval_large: bool = False) -> ExperimentConfig:
    return ExperimentConfig.from_mapping(
        {
            "meta": {"experiment_id": "demo"},
            "paths": {
                "repo_root": ".",
                "output_root": "./outputs",
                "dataset_root": "./datasets",
                "model_root": "./models",
            },
            "data": {
                "dataset_name": "jobs",
                "train_path": "./datasets/jobs_train.json",
                "eval_path": "./datasets/jobs_eval.json",
                "initialization_path": "./datasets/init.json",
            },
            "stage1": {"enabled": stage1, "rounds": 2},
            "bootstrap": {"enabled": stage2},
            "eval_small": {"enabled": False},
            "eval_large": {"enabled": eval_large, "eval_mode": "peft_lora"},
        },
        base_dir=root,
    )


class PreflightTests(unittest.TestCase):
    def test_preflight_reports_missing_dependencies_models_and_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _build_config(Path(tmp_dir), stage1=True, stage2=True, eval_large=True)
            with patch("pretext_platform.core.preflight._module_available", return_value=False):
                report = run_preflight(config)

        self.assertFalse(report.ready)
        categories = {issue.category for issue in report.errors}
        self.assertIn("dependency", categories)
        self.assertIn("data", categories)
        self.assertIn("model", categories)

    def test_preflight_passes_when_required_inputs_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _touch_json(root / "datasets" / "jobs_train.json", '["train"]')
            _touch_json(root / "datasets" / "jobs_eval.json", '["eval"]')
            _touch_json(root / "datasets" / "init.json", '["seed text with enough words for initialization"]')
            for model_dir in ["all_minilm_l6_v2", "roberta_large", "llama_2_7b_hf", "distilgpt2"]:
                (root / "models" / model_dir).mkdir(parents=True, exist_ok=True)

            config = _build_config(root, stage1=True, stage2=True, eval_large=False)
            with patch("pretext_platform.core.preflight._module_available", return_value=True):
                report = run_preflight(config)

        self.assertTrue(report.ready)
        self.assertEqual(report.errors, [])

    def test_preflight_requires_stage2_artifact_for_eval_only_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _touch_json(root / "datasets" / "jobs_train.json", '["train"]')
            _touch_json(root / "datasets" / "jobs_eval.json", '["eval"]')
            _touch_json(root / "datasets" / "init.json", '["seed text with enough words for initialization"]')
            for model_dir in ["all_minilm_l6_v2", "roberta_large", "llama_2_7b_hf", "distilgpt2"]:
                (root / "models" / model_dir).mkdir(parents=True, exist_ok=True)

            config = _build_config(root, stage1=False, stage2=False, eval_large=True)
            with patch("pretext_platform.core.preflight._module_available", return_value=True):
                report = run_preflight(config)

        self.assertFalse(report.ready)
        self.assertTrue(any(issue.category == "artifact" for issue in report.errors))

    def test_glue_validation_accepts_local_rotten_tomatoes_raw_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for task in ["sst2", "qqp", "qnli"]:
                formatted = root / f"glue_{task}" / "formatted"
                (formatted / "train").mkdir(parents=True, exist_ok=True)
                (formatted / "validation").mkdir(parents=True, exist_ok=True)
                (formatted / "dataset_dict.json").write_text("{}", encoding="utf-8")
            imdb = root / "imdb" / "formatted"
            imdb.mkdir(parents=True, exist_ok=True)
            (imdb / "train_len256.jsonl").write_text("{}", encoding="utf-8")
            (imdb / "validation_len256.jsonl").write_text("{}", encoding="utf-8")
            rotten = root / "rotten_tomatoes" / "raw"
            (rotten / "train").mkdir(parents=True, exist_ok=True)
            (rotten / "validation").mkdir(parents=True, exist_ok=True)
            (rotten / "dataset_dict.json").write_text("{}", encoding="utf-8")

            validation = validate_local_glue_datasets(root)

        self.assertTrue(validation["all_available"])
        self.assertTrue(validation["all_required_available"])
        self.assertEqual(validation["missing"], [])
        self.assertEqual(validation["fallback_only"], [])

    def test_glue_validation_accepts_local_rotten_tomatoes_dataset_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for task in ["sst2", "qqp", "qnli"]:
                formatted = root / f"glue_{task}" / "formatted"
                (formatted / "train").mkdir(parents=True, exist_ok=True)
                (formatted / "validation").mkdir(parents=True, exist_ok=True)
                (formatted / "dataset_dict.json").write_text("{}", encoding="utf-8")
            imdb = root / "imdb" / "formatted"
            imdb.mkdir(parents=True, exist_ok=True)
            (imdb / "train_len256.jsonl").write_text("{}", encoding="utf-8")
            (imdb / "validation_len256.jsonl").write_text("{}", encoding="utf-8")
            rotten_raw = root / "rotten_tomatoes" / "raw"
            (rotten_raw / "train").mkdir(parents=True, exist_ok=True)
            (rotten_raw / "validation").mkdir(parents=True, exist_ok=True)
            (rotten_raw / "dataset_dict.json").write_text("{}", encoding="utf-8")

            validation = validate_local_glue_datasets(root)

        self.assertTrue(validation["all_required_available"])
        self.assertTrue(validation["all_available"])
        self.assertEqual(validation["missing"], [])
        self.assertEqual(validation["fallback_only"], [])

    def test_runtime_config_reanchors_repo_root_for_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "demo.json"
            config_path.write_text(
                json.dumps(
                    {
                        "meta": {"experiment_id": "demo"},
                        "paths": {
                            "repo_root": ".",
                            "dataset_root": "./datasets",
                            "model_root": "./models",
                            "output_root": "./outputs",
                        },
                    }
                ),
                encoding="utf-8",
            )

            runtime_path, _ = _write_runtime_config(str(config_path), output_root=root / "other_outputs")
            try:
                from pretext_platform.core.config import load_experiment_config

                runtime_config = load_experiment_config(runtime_path)
                self.assertEqual(runtime_config.repo_root(), root.resolve())
                self.assertEqual(runtime_config.dataset_root(), (root / "datasets").resolve())
                self.assertEqual(runtime_config.model_root(), (root / "models").resolve())
                self.assertEqual(runtime_config.output_root().name, "other_outputs")
                self.assertEqual(runtime_config.output_root().parent.name, root.name)
            finally:
                runtime_path.unlink(missing_ok=True)
