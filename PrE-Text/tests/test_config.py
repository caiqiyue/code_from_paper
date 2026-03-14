from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pretext_platform.core.config import ExperimentConfig, load_experiment_config
from pretext_platform.core.models import resolve_model_paths


class ConfigTests(unittest.TestCase):
    """Exercise the config loader and path resolution helpers."""

    def test_yaml_inheritance_and_path_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "base.yaml").write_text(
                """
paths:
  repo_root: .
  output_root: ./out
models:
  minilm_path: ./models/minilm
stage1:
  sigma: 2.31
""".strip(),
                encoding="utf-8",
            )
            (root / "experiment.yaml").write_text(
                """
inherits:
  - ./base.yaml
meta:
  experiment_id: demo
paths:
  dataset_root: ./datasets
  model_root: ./model_root
models:
  roberta_large_path: ./models/roberta
""".strip(),
                encoding="utf-8",
            )
            config = load_experiment_config(root / "experiment.yaml")
            self.assertEqual(config.experiment_id(), "demo")
            self.assertEqual(config.output_root(), (root / "out").resolve())
            self.assertEqual(config.dataset_root(), (root / "datasets").resolve())
            model_paths = resolve_model_paths(config)
            self.assertEqual(model_paths.minilm, (root / "models" / "minilm").resolve())
            self.assertEqual(model_paths.roberta_large, (root / "models" / "roberta").resolve())

    def test_from_mapping_supports_in_memory_configs(self) -> None:
        config = ExperimentConfig.from_mapping(
            {
                "meta": {"experiment_id": "memory_demo"},
                "paths": {"repo_root": ".", "output_root": "./out"},
            },
            base_dir=Path.cwd(),
        )
        self.assertEqual(config.experiment_id(), "memory_demo")
        self.assertTrue(str(config.output_root()).endswith("out"))
