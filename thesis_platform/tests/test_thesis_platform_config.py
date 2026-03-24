from __future__ import annotations

import unittest

from thesis_platform.core.config import load_experiment_config


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
        self.assertTrue(config.downstream_eval["run_large_eval"])
        self.assertFalse(config.downstream_eval["run_small_eval"])
        self.assertEqual(config.scorer["name"], "datainf_real")


if __name__ == "__main__":
    unittest.main()
