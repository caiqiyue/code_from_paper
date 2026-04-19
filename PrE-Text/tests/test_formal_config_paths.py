from __future__ import annotations

from pathlib import Path
import unittest

from pretext_platform.core.config import load_experiment_config


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent


class FormalConfigPathTests(unittest.TestCase):
    def test_single_node_formal_config_resolves_workspace_paths(self) -> None:
        config = load_experiment_config(
            REPO_ROOT / "configs" / "experiments" / "single_node_formal" / "sp_c1_jobs_base.yaml"
        )

        self.assertEqual(config.repo_root(), REPO_ROOT)
        self.assertEqual(config.output_root(), REPO_ROOT / "outputs" / "pretext_platform")
        self.assertEqual(
            config.resolve_path(config.data.get("train_path")),
            WORKSPACE_ROOT / "thesis_platform" / "datasets" / "pretext_jobs" / "formatted" / "jobs_train.json",
        )
        self.assertEqual(
            config.resolve_path(config.data.get("initialization_path")),
            WORKSPACE_ROOT
            / "thesis_platform"
            / "datasets"
            / "pretext_initialization_c4_en"
            / "formatted"
            / "initialization.json",
        )

    def test_federated_formal_config_resolves_workspace_paths(self) -> None:
        config = load_experiment_config(
            REPO_ROOT / "configs" / "experiments" / "federated_formal" / "fp_c1_jobs_base.yaml"
        )

        self.assertEqual(config.repo_root(), REPO_ROOT)
        self.assertEqual(config.output_root(), REPO_ROOT / "outputs" / "pretext_platform")
        self.assertEqual(
            config.resolve_path(config.data.get("train_path")),
            WORKSPACE_ROOT / "thesis_platform" / "datasets" / "pretext_jobs" / "formatted" / "jobs_train.json",
        )
        self.assertEqual(
            config.resolve_path(config.data.get("eval_path")),
            WORKSPACE_ROOT / "thesis_platform" / "datasets" / "pretext_jobs" / "formatted" / "jobs_eval.json",
        )


if __name__ == "__main__":
    unittest.main()
