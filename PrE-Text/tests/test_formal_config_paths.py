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

    def test_formal_configs_pin_runtime_device_to_cuda1(self) -> None:
        for config_path in (
            REPO_ROOT / "configs" / "experiments" / "single_node_formal" / "_base_pretext_formal.yaml",
            REPO_ROOT / "configs" / "experiments" / "federated_formal" / "_base_federated_formal.yaml",
        ):
            with self.subTest(config_path=config_path):
                config = load_experiment_config(config_path)
                self.assertEqual(config.runtime["device"], "cuda:1")


if __name__ == "__main__":
    unittest.main()
