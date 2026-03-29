from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from thesis_platform.core.config import load_experiment_config
from thesis_platform.core.privacy import PrivacyLedger, PrivacyPolicy
from thesis_platform.evaluation.downstream_eval import (
    DownstreamEvalManager,
    resolve_large_eval_mode,
    resolve_small_eval_mode,
)
from thesis_platform.models.backends import build_text_backend


class PrivacyRuntimeTests(unittest.TestCase):
    """Unit tests for the v3 privacy runtime contract."""

    def test_privacy_ledger_disabled_is_a_noop(self) -> None:
        policy = PrivacyPolicy.from_config({"enabled": False})
        ledger = PrivacyLedger(policy=policy)

        round_summary = ledger.record_round(round_id=0, sample_count=8, critique_count=2, upload_token_count=30)

        self.assertFalse(round_summary["privacy_enabled"])
        self.assertEqual(round_summary["privacy_mode"], "disabled")
        self.assertEqual(round_summary["privacy_spent"], 0.0)
        self.assertEqual(ledger.summary()["spent_total"], 0.0)

    def test_privacy_ledger_accumulates_budget_usage(self) -> None:
        policy = PrivacyPolicy.from_config(
            {
                "enabled": True,
                "epsilon": 1.0,
                "delta": 1e-5,
                "sample_cost": 0.01,
                "critique_cost": 0.02,
                "upload_token_cost": 0.001,
            }
        )
        ledger = PrivacyLedger(policy=policy)

        first = ledger.record_round(round_id=0, sample_count=10, critique_count=3, upload_token_count=20)
        second = ledger.record_round(round_id=1, sample_count=5, critique_count=1, upload_token_count=10)

        self.assertAlmostEqual(first["privacy_spent"], 0.18)
        self.assertAlmostEqual(second["privacy_spent"], 0.08)
        self.assertAlmostEqual(second["privacy_spent_cumulative"], 0.26)
        self.assertAlmostEqual(second["privacy_budget_left"], 0.74)
        self.assertEqual(ledger.summary()["round_count"], 2)


class DownstreamEvalManagerTests(unittest.TestCase):
    """Unit tests for the downstream evaluation manager runtime contract."""

    def test_auto_mode_resolution_is_platform_aware(self) -> None:
        downstream_cfg = {
            "large_eval_mode": "auto",
            "windows_large_eval_mode": "full_finetune",
            "linux_large_eval_mode": "peft_lora",
            "small_eval_mode": "auto",
            "windows_small_eval_mode": "gpt2",
            "linux_small_eval_mode": "distilgpt2",
            "c4_checkpoint_path": "",
        }
        self.assertEqual(resolve_large_eval_mode(downstream_cfg, platform_name="win32"), "full_finetune")
        self.assertEqual(resolve_large_eval_mode(downstream_cfg, platform_name="linux"), "peft_lora")
        self.assertEqual(resolve_small_eval_mode(downstream_cfg, platform_name="win32"), "gpt2")
        self.assertEqual(resolve_small_eval_mode(downstream_cfg, platform_name="linux"), "gpt2")

    def test_disabled_stages_produce_disabled_overall_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            config_path = tmp_root / "downstream_disabled.yaml"
            config_path.write_text(
                f"""
meta:
  experiment_id: downstream_disabled
paths:
  repo_root: "{tmp_root.as_posix()}"
  output_root: "{(tmp_root / 'out').as_posix()}"
  cache_root: "{(tmp_root / 'cache').as_posix()}"
downstream_eval:
  enabled: true
  kind: pretext_large_eval
  run_large_eval: false
  run_small_eval: false
""".strip(),
                encoding="utf-8",
            )

            config = load_experiment_config(config_path)
            summary = DownstreamEvalManager(
                config,
                experiment_id="downstream_disabled",
                output_dir=tmp_root / "out" / "downstream_eval",
            ).run(["synthetic one"])

            self.assertFalse(summary["enabled"])
            self.assertEqual(summary["status"], "disabled")

    def test_windows_large_eval_auto_mode_runs_windows_compatible_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            train_path = tmp_root / "train.json"
            eval_path = tmp_root / "eval.json"
            init_path = tmp_root / "init.json"
            llama32_dir = tmp_root / "thesis_platform" / "open_model" / "llama_3_2_3b_instruct"
            llama32_dir.mkdir(parents=True, exist_ok=True)
            train_path.write_text(json.dumps({"0": ["sample alpha text"], "1": ["sample beta text"]}), encoding="utf-8")
            eval_path.write_text(json.dumps(["eval gamma text"]), encoding="utf-8")
            init_path.write_text(json.dumps(["seed delta epsilon zeta eta theta iota kappa lambda"]), encoding="utf-8")

            config_path = tmp_root / "downstream_windows_large.yaml"
            config_path.write_text(
                f"""
meta:
  experiment_id: downstream_windows_large
paths:
  repo_root: "{tmp_root.as_posix()}"
  output_root: "{(tmp_root / 'out').as_posix()}"
  cache_root: "{(tmp_root / 'cache').as_posix()}"
data:
  dataset_name: tmp
  train_path: "{train_path.as_posix()}"
  eval_path: "{eval_path.as_posix()}"
  initialization_path: "{init_path.as_posix()}"
  max_samples_per_client: 2
  initialization_min_words: 3
downstream_eval:
  enabled: true
  kind: pretext_large_eval
  run_large_eval: true
  run_small_eval: false
  large_eval_mode: auto
  windows_large_eval_mode: full_finetune
  linux_large_eval_mode: peft_lora
  llama_3_2_3b_instruct_path: thesis_platform/open_model/llama_3_2_3b_instruct
""".strip(),
                encoding="utf-8",
            )

            config = load_experiment_config(config_path)
            fake_eval_summary = {
                "stage_name": "eval_large",
                "metrics": {"best_top1": 0.13, "model": "Llama-3.2-3B-Instruct"},
                "artifacts": {"stats_dir": str(tmp_root / "fake_stats")},
            }
            with patch("thesis_platform.evaluation.downstream_eval.sys.platform", "win32"), patch(
                "thesis_platform.evaluation.downstream_eval.run_pretext_large_eval",
                return_value=fake_eval_summary,
            ):
                summary = DownstreamEvalManager(
                    config,
                    experiment_id="downstream_windows_large",
                    output_dir=tmp_root / "out" / "downstream_eval",
                ).run(["synthetic one", "synthetic two"])

            self.assertEqual(summary["resolved_modes"]["large_eval_mode"], "full_finetune")
            self.assertEqual(summary["stages"]["large_eval"]["status"], "completed")
            self.assertEqual(summary["stages"]["large_eval"]["metrics"]["model"], "Llama-3.2-3B-Instruct")

    def test_small_eval_reports_missing_checkpoint_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            train_path = tmp_root / "train.json"
            eval_path = tmp_root / "eval.json"
            init_path = tmp_root / "init.json"
            distilgpt2_dir = tmp_root / "thesis_platform" / "open_model" / "distilgpt2"
            distilgpt2_dir.mkdir(parents=True, exist_ok=True)
            train_path.write_text(json.dumps({"0": ["sample alpha text"], "1": ["sample beta text"]}), encoding="utf-8")
            eval_path.write_text(json.dumps(["eval gamma text"]), encoding="utf-8")
            init_path.write_text(json.dumps(["seed delta epsilon zeta eta theta iota kappa lambda"]), encoding="utf-8")

            config_path = tmp_root / "downstream_small.yaml"
            config_path.write_text(
                f"""
meta:
  experiment_id: downstream_small
paths:
  repo_root: "{tmp_root.as_posix()}"
  output_root: "{(tmp_root / 'out').as_posix()}"
  cache_root: "{(tmp_root / 'cache').as_posix()}"
data:
  dataset_name: tmp
  train_path: "{train_path.as_posix()}"
  eval_path: "{eval_path.as_posix()}"
  initialization_path: "{init_path.as_posix()}"
  max_samples_per_client: 2
  initialization_min_words: 3
downstream_eval:
  enabled: true
  kind: pretext_large_eval
  run_large_eval: false
  run_small_eval: true
  export_filename: llama7b_text_syn.json
  small_eval_mode: distilgpt2
  distilgpt2_path: thesis_platform/open_model/distilgpt2
  c4_checkpoint_path: thesis_platform/open_model/c4_checkpoint.pth
""".strip(),
                encoding="utf-8",
            )

            config = load_experiment_config(config_path)
            summary = DownstreamEvalManager(
                config,
                experiment_id="downstream_small",
                output_dir=tmp_root / "out" / "downstream_eval",
            ).run(["synthetic one", "synthetic two"])

            small_stage = summary["stages"]["small_eval"]
            self.assertEqual(small_stage["status"], "blocked_missing_asset")
            self.assertTrue(any(item["label"] == "c4_checkpoint_path" for item in small_stage["missing_assets"]))
            self.assertTrue((tmp_root / "out" / "downstream_eval" / "pretext_small_eval_summary.json").exists())


class BackendRuntimeTests(unittest.TestCase):
    """Unit tests for backend configuration edge cases seen in real experiments."""

    def test_build_text_backend_accepts_string_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            model_dir = tmp_root / "models" / "demo"
            model_dir.mkdir(parents=True, exist_ok=True)

            with patch("thesis_platform.models.backends.TransformersTextBackend") as backend_cls:
                build_text_backend(
                    {
                        "engine": "transformers",
                        "model_name_or_path": "models/demo",
                        "use_fast": False,
                    },
                    repo_root=str(tmp_root),
                )

            kwargs = backend_cls.call_args.kwargs
            self.assertEqual(kwargs["model_path"], model_dir.resolve())
            self.assertIs(kwargs["use_fast"], False)


if __name__ == "__main__":
    unittest.main()
