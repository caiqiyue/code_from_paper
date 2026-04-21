from __future__ import annotations

from pathlib import Path
import unittest

from pretext_platform.core.config import load_experiment_config


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = REPO_ROOT / "configs" / "experiments"
SAFE_SMALL_EVAL_MODES = {"gpt2", "distilgpt2"}


class EvalConfigCoverageTests(unittest.TestCase):
    def test_all_enabled_eval_small_configs_use_verified_modes(self) -> None:
        enabled_configs: list[tuple[Path, str]] = []
        for config_path in sorted(EXPERIMENTS_ROOT.rglob("*.yaml")):
            config = load_experiment_config(config_path)
            if not bool(config.eval_small.get("enabled", False)):
                continue
            mode = str(config.eval_small.get("eval_mode", "gpt2")).strip().lower()
            enabled_configs.append((config_path.relative_to(REPO_ROOT), mode))

        self.assertTrue(enabled_configs, "Expected at least one config with eval_small enabled.")
        unexpected = [(path, mode) for path, mode in enabled_configs if mode not in SAFE_SMALL_EVAL_MODES]
        self.assertEqual([], unexpected)

    def test_formal_pretext_configs_route_manual_small_eval_to_verified_modes(self) -> None:
        formal_roots = (
            EXPERIMENTS_ROOT / "single_node_formal",
            EXPERIMENTS_ROOT / "federated_formal",
        )
        formal_configs: list[tuple[Path, str]] = []
        for root in formal_roots:
            for config_path in sorted(root.rglob("*.yaml")):
                config = load_experiment_config(config_path)
                mode = str(config.eval_small.get("eval_mode", "gpt2")).strip().lower()
                formal_configs.append((config_path.relative_to(REPO_ROOT), mode))

        self.assertTrue(formal_configs, "Expected formal pre-text configs to be present.")
        unexpected = [(path, mode) for path, mode in formal_configs if mode not in SAFE_SMALL_EVAL_MODES]
        self.assertEqual([], unexpected)


if __name__ == "__main__":
    unittest.main()
