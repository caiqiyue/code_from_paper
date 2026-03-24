from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from thesis_platform.core.config import load_experiment_config
from thesis_platform.core.privacy import PrivacyLedger, PrivacyPolicy
from thesis_platform.evaluation.downstream_eval import DownstreamEvalManager


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


if __name__ == "__main__":
    unittest.main()
