from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from thesis_platform.core.config import load_experiment_config
from thesis_platform.core.experiment_runner import ExperimentRunner


class PipelineTests(unittest.TestCase):
    """Exercise the full single-round pipeline against a temporary dataset."""

    def test_single_round_pipeline_outputs_expected_files(self) -> None:
        """Verify one round produces every mandatory artifact in the output folder."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            train_path = tmp_root / "train.json"
            seed_path = tmp_root / "seed.json"
            train_path.write_text(json.dumps({"0": ["alpha beta gamma"], "1": ["beta gamma delta"], "2": ["gamma delta epsilon"]}), encoding="utf-8")
            seed_path.write_text(json.dumps(["seed alpha beta", "seed gamma delta"]), encoding="utf-8")

            config_path = tmp_root / "experiment.yaml"
            config_path.write_text(
                """
meta:
  experiment_id: tmp_smoke
  seed: 7
paths:
  repo_root: .
  output_root: ./out
  cache_root: ./cache
data:
  dataset_name: tmp
  task_type: instruction_tuning
  train_path: ./train.json
  public_seed_path: ./seed.json
  num_clients: 2
  max_samples_per_client: 2
  validation_ratio: 0.5
federation:
  rounds: 1
  top_k_bad: 1
generator:
  name: pretext_seed
  generated_per_round: 4
  mask: 0.2
  t_steps: 1
scorer:
  name: pretext_hist
retriever:
  name: knn
  top_k: 1
critic:
  name: fedtextgrad_qwen
  compress_to_n_rules: 1
  redact_enable: true
aggregator:
  name: uid
""".strip(),
                encoding="utf-8",
            )

            config = load_experiment_config(config_path)
            summary = ExperimentRunner(config).run()
            exp_dir = tmp_root / "out" / "tmp_smoke"
            round_dir = exp_dir / "round_000"
            self.assertEqual(summary["experiment_id"], "tmp_smoke")
            self.assertTrue((exp_dir / "metrics_summary.json").exists())
            self.assertTrue((exp_dir / "resolved_config.json").exists())
            self.assertTrue((exp_dir / "privacy_ledger.json").exists())
            self.assertTrue((exp_dir / "artifact_manifest.json").exists())
            self.assertTrue((exp_dir / "config.yaml").exists())
            self.assertTrue((round_dir / "generated_samples.jsonl").exists())
            self.assertTrue((round_dir / "scored_samples.jsonl").exists())
            self.assertTrue((round_dir / "selected_bad_samples.jsonl").exists())
            self.assertTrue((round_dir / "retrieved_pairs.jsonl").exists())
            self.assertTrue((round_dir / "client_critiques.jsonl").exists())
            self.assertTrue((round_dir / "round_metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
