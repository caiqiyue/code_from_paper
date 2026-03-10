from __future__ import annotations

import unittest
from pathlib import Path

from thesis_platform.core.config import load_experiment_config


class ConfigTests(unittest.TestCase):
    def test_load_real_smoke_config(self) -> None:
        config = load_experiment_config("thesis_platform/configs/experiments/smoke/smoke_pretext_hist_congressional.yaml")
        self.assertEqual(config.meta["experiment_id"], "smoke_pretext_hist_congressional")
        self.assertTrue(config.repo_root().exists())
        self.assertTrue(str(config.output_root()).endswith("outputs\\thesis_platform") or str(config.output_root()).endswith("outputs/thesis_platform"))


if __name__ == "__main__":
    unittest.main()
