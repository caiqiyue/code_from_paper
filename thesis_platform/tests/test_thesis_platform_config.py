from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from thesis_platform.core.config import load_experiment_config
from thesis_platform.core.preflight import _cross_domain_eval_runtime_cfg


class ConfigTests(unittest.TestCase):
    """Validate experiment config loading and path resolution behavior."""

    def test_load_real_smoke_config(self) -> None:
        """Verify a shipped smoke config resolves into a usable ExperimentConfig."""

        config = load_experiment_config("thesis_platform/configs/experiments/smoke/smoke_pretext_hist_congressional.yaml")
        self.assertEqual(config.meta["experiment_id"], "smoke_pretext_hist_congressional")
        self.assertTrue(config.repo_root().exists())
        self.assertTrue(str(config.output_root()).endswith("outputs\\thesis_platform") or str(config.output_root()).endswith("outputs/thesis_platform"))

    def test_load_v3_jobs_config_exposes_new_sections(self) -> None:
        """Verify the v3 Jobs config exposes prototype, routing, privacy, and downstream_eval."""

        config = load_experiment_config("thesis_platform/configs/experiments/v3/jobs_real_datainf_v3.yaml")
        self.assertEqual(config.prototype["name"], "minilm_mean")
        self.assertTrue(config.routing["enabled"])
        self.assertTrue(config.privacy["enabled"])
        self.assertEqual(config.privacy["epsilon"], 1.29)
        self.assertTrue(config.downstream_eval["enabled"])
        self.assertFalse(config.downstream_eval["run_large_eval"])
        self.assertFalse(config.downstream_eval["run_small_eval"])
        self.assertEqual(config.downstream_eval["large_eval_mode"], "auto")
        self.assertEqual(config.downstream_eval["windows_large_eval_mode"], "full_finetune")
        self.assertEqual(config.downstream_eval["linux_large_eval_mode"], "peft_lora")
        self.assertEqual(config.scorer["name"], "datainf_real")

    def test_load_v3_jobs_large_eval_variant_enables_large_eval(self) -> None:
        """Verify the large-eval variant keeps the same algorithm config and turns on final evaluation."""

        config = load_experiment_config(
            "thesis_platform/configs/experiments/v3/jobs_real_datainf_v3_large_eval.yaml"
        )
        self.assertEqual(config.meta["experiment_id"], "jobs_real_datainf_v3_large_eval")
        self.assertTrue(config.privacy["enabled"])
        self.assertEqual(config.scorer["name"], "datainf_real")
        self.assertTrue(config.downstream_eval["enabled"])
        self.assertTrue(config.downstream_eval["run_large_eval"])
        self.assertFalse(config.downstream_eval["run_small_eval"])

    def test_transfer_configs_run_cross_domain_small_eval(self) -> None:
        """Verify transfer configs do not silently fall back to large eval."""

        for config_path in (
            "thesis_platform/configs/experiments/linux/jobs_to_forums_transfer.yaml",
            "thesis_platform/configs/experiments/linux/jobs_to_microblog_transfer.yaml",
            "thesis_platform/configs/experiments/linux/jobs_to_congressional_transfer.yaml",
            "thesis_platform/configs/experiments/linux/forums_to_jobs_transfer.yaml",
            "thesis_platform/configs/experiments/linux/microblog_to_jobs_transfer.yaml",
            "thesis_platform/configs/experiments/linux/congressional_to_jobs_transfer.yaml",
        ):
            with self.subTest(config_path=config_path):
                config = load_experiment_config(config_path)
                eval_cfg = _cross_domain_eval_runtime_cfg(config)
                self.assertTrue(eval_cfg["enabled"])
                self.assertFalse(eval_cfg["run_large_eval"])
                self.assertTrue(eval_cfg["run_small_eval"])
                self.assertEqual(eval_cfg["small_eval_mode"], "gpt2")

    def test_resolve_path_normalizes_windows_style_relative_paths(self) -> None:
        """Verify backslash-separated relative paths resolve correctly across platforms."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            config_path = tmp_root / "path_normalize.yaml"
            config_path.write_text(
                """
meta:
  experiment_id: path_normalize
paths:
  repo_root: .
  output_root: outputs\\thesis_platform
  cache_root: thesis_platform\\workspace\\cache
""".strip(),
                encoding="utf-8",
            )

            config = load_experiment_config(config_path)
            self.assertEqual(
                config.output_root(),
                (tmp_root / "outputs" / "thesis_platform").resolve(),
            )
            self.assertEqual(
                config.resolve_path("thesis_platform\\datasets\\demo.json"),
                (tmp_root / "thesis_platform" / "datasets" / "demo.json").resolve(),
            )


if __name__ == "__main__":
    unittest.main()
