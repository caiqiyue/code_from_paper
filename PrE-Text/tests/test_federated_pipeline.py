from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pretext_platform.core.config import ExperimentConfig, load_experiment_config
from pretext_platform.core.pipeline import run_pipeline
from pretext_platform.core.types import ModelPaths, StageSummary


class FederatedConfigAndPipelineTests(unittest.TestCase):
    """Validate the federated PrE-Text config surface and orchestration shell."""

    def test_tiny_federated_config_loads_from_repo_assets(self) -> None:
        config = load_experiment_config(
            Path(__file__).resolve().parents[1] / "configs" / "experiments" / "federated" / "fpt_jobs_tiny.yaml"
        )

        self.assertEqual(config.execution["mode"], "federated_pretext")
        self.assertEqual(config.federation["rounds"], 2)
        self.assertEqual(config.experiment_id(), "fpt_jobs_tiny")

    def test_config_exposes_execution_and_federation_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "fpt_demo.yaml"
            config_path.write_text(
                """
meta:
  experiment_id: fpt_demo
paths:
  repo_root: .
  output_root: ./out
execution:
  mode: federated_pretext
federation:
  rounds: 2
  num_clients: 3
  validation_ratio: 0.25
  max_samples_per_client: 4
  partition_strategy: preserve_buckets
""".strip(),
                encoding="utf-8",
            )

            config = load_experiment_config(config_path)

            self.assertEqual(config.execution["mode"], "federated_pretext")
            self.assertEqual(config.federation["rounds"], 2)
            self.assertEqual(config.federation["num_clients"], 3)

    def test_pipeline_dispatches_to_federated_runner_when_mode_requests_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = ExperimentConfig.from_mapping(
                {
                    "meta": {"experiment_id": "fpt_dispatch"},
                    "paths": {"repo_root": ".", "output_root": "./out"},
                    "execution": {"mode": "federated_pretext"},
                    "federation": {"rounds": 1, "num_clients": 2},
                },
                base_dir=root,
            )
            fake_summary = {
                "experiment_id": "fpt_dispatch",
                "status": "SUCCESS",
                "experiment_dir": str(root / "out" / "fpt_dispatch"),
            }

            with patch("pretext_platform.core.pipeline.run_federated_pipeline", return_value=fake_summary) as mocked:
                summary = run_pipeline(config)

            mocked.assert_called_once_with(config)
            self.assertEqual(summary["experiment_id"], "fpt_dispatch")

    def test_federated_runner_writes_round_artifacts_and_uses_last_round_output(self) -> None:
        from pretext_platform.core.federated_runner import FederatedPretextRunner

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = ExperimentConfig.from_mapping(
                {
                    "meta": {"experiment_id": "fpt_roundtrip"},
                    "paths": {"repo_root": ".", "output_root": "./out"},
                    "execution": {"mode": "federated_pretext"},
                    "federation": {
                        "rounds": 2,
                        "num_clients": 2,
                        "validation_ratio": 0.25,
                        "max_samples_per_client": 4,
                        "partition_strategy": "preserve_buckets",
                    },
                    "data": {"dataset_name": "jobs"},
                    "stage1": {"enabled": True},
                    "bootstrap": {"enabled": True},
                    "eval_small": {"enabled": False},
                    "eval_large": {"enabled": False},
                },
                base_dir=root,
            )

            partition = {
                "client_000": {
                    "train_texts": ["client0 text a", "client0 text b"],
                    "eval_texts": ["client0 eval"],
                },
                "client_001": {
                    "train_texts": ["client1 text a", "client1 text b"],
                    "eval_texts": ["client1 eval"],
                },
            }

            def fake_stage1_runner(*, client_id: str, round_id: int, output_dir: Path, **_: object) -> tuple[StageSummary, list[str]]:
                surviving_texts = [f"{client_id}_round_{round_id}_survivor_a", f"{client_id}_round_{round_id}_survivor_b"]
                surviving_path = output_dir / f"surviving_text_it{round_id}.json"
                surviving_path.write_text(json.dumps(surviving_texts, ensure_ascii=False), encoding="utf-8")
                summary = StageSummary(
                    stage_name="stage1",
                    output_dir=output_dir,
                    artifacts={"surviving_files": [str(surviving_path)]},
                    metrics={"epsilon": 1.29, "delta": 1e-6, "rounds": 1, "surviving_count": len(surviving_texts)},
                )
                return summary, surviving_texts

            def fake_bootstrap_runner(*, merged_surviving_texts: list[str], round_id: int, server_output_dir: Path, **_: object) -> StageSummary:
                bootstrap_path = server_output_dir / "bootstrap_inputs.json"
                bootstrap_path.write_text(json.dumps(merged_surviving_texts, ensure_ascii=False), encoding="utf-8")
                return StageSummary(
                    stage_name="bootstrap",
                    output_dir=server_output_dir,
                    artifacts={"bootstrap_inputs_path": str(bootstrap_path)},
                    metrics={"seed_text_count": len(merged_surviving_texts), "round_id": round_id},
                )

            def fake_stage2_runner(*, merged_surviving_texts: list[str], round_id: int, server_output_dir: Path, **_: object) -> StageSummary:
                output_path = server_output_dir / "llama7b_text_syn.json"
                generated = [f"global_round_{round_id}_sample_{idx}" for idx, _ in enumerate(merged_surviving_texts)]
                output_path.write_text(json.dumps(generated, ensure_ascii=False), encoding="utf-8")
                return StageSummary(
                    stage_name="stage2",
                    output_dir=server_output_dir,
                    artifacts={"synthetic_corpus_path": str(output_path)},
                    metrics={"generated_count": len(generated), "round_id": round_id},
                )

            runner = FederatedPretextRunner(
                config,
                partition_fn=lambda _: partition,
                stage1_runner=fake_stage1_runner,
                bootstrap_runner=fake_bootstrap_runner,
                stage2_runner=fake_stage2_runner,
            )

            summary = runner.run()
            experiment_dir = root / "out" / "fpt_roundtrip"

            self.assertEqual(summary["status"], "SUCCESS")
            self.assertEqual(summary["completed_rounds"], 2)
            self.assertTrue((experiment_dir / "metrics_summary.json").exists())
            self.assertTrue((experiment_dir / "privacy_ledger.json").exists())
            self.assertTrue((experiment_dir / "round_000" / "client_000" / "stage1_summary.json").exists())
            self.assertTrue((experiment_dir / "round_000" / "server_stage2" / "llama7b_text_syn.json").exists())
            self.assertTrue((experiment_dir / "stage2" / "llama7b_text_syn.json").exists())
            self.assertTrue((experiment_dir / "stage2_summary.json").exists())
            self.assertTrue(summary["final_synthetic_corpus_path"].endswith("round_001/server_stage2/llama7b_text_syn.json"))

    def test_federated_runner_releases_gpu_between_client_stage1_and_server_stage2(self) -> None:
        from pretext_platform.core.federated_runner import FederatedPretextRunner

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = ExperimentConfig.from_mapping(
                {
                    "meta": {"experiment_id": "fpt_cleanup"},
                    "paths": {"repo_root": ".", "output_root": "./out"},
                    "execution": {"mode": "federated_pretext"},
                    "federation": {"rounds": 1, "num_clients": 2},
                    "stage1": {"enabled": True},
                    "bootstrap": {"enabled": True},
                },
                base_dir=root,
            )
            partition = {
                "client_000": {"train_texts": ["client0"], "eval_texts": []},
                "client_001": {"train_texts": ["client1"], "eval_texts": []},
            }

            def fake_stage1_runner(*, client_id: str, round_id: int, output_dir: Path, **_: object) -> tuple[StageSummary, list[str]]:
                return (
                    StageSummary("stage1", output_dir, artifacts={}, metrics={"round_id": round_id}),
                    [f"{client_id} survivor a", f"{client_id} survivor b"],
                )

            def fake_bootstrap_runner(*, server_output_dir: Path, round_id: int, **_: object) -> StageSummary:
                return StageSummary("bootstrap", server_output_dir, artifacts={}, metrics={"round_id": round_id})

            def fake_stage2_runner(*, server_output_dir: Path, round_id: int, **_: object) -> StageSummary:
                output_path = server_output_dir / "llama7b_text_syn.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(["synthetic"], ensure_ascii=False), encoding="utf-8")
                return StageSummary(
                    "stage2",
                    server_output_dir,
                    artifacts={"synthetic_corpus_path": str(output_path)},
                    metrics={"round_id": round_id, "generated_count": 1},
                )

            runner = FederatedPretextRunner(
                config,
                partition_fn=lambda _: partition,
                stage1_runner=fake_stage1_runner,
                bootstrap_runner=fake_bootstrap_runner,
                stage2_runner=fake_stage2_runner,
            )

            with patch("pretext_platform.core.federated_runner.release_gpu_memory", create=True) as release:
                runner.run()

            self.assertGreaterEqual(release.call_count, 3)

    def test_default_stage1_runner_uses_configured_initialization_pool(self) -> None:
        from pretext_platform.core.federated_runner import _default_stage1_runner

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            init_path = root / "initialization.json"
            init_path.write_text(
                json.dumps(
                    [
                        "this initialization sentence has enough words for stage one filtering",
                        "another initialization sentence with enough words for the test harness",
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            config = ExperimentConfig.from_mapping(
                {
                    "meta": {"experiment_id": "fpt_stage1_real", "seed": 42},
                    "paths": {"repo_root": ".", "output_root": "./out"},
                    "data": {
                        "dataset_name": "jobs",
                        "initialization_path": str(init_path),
                        "initialization_min_words": 2,
                    },
                    "stage1": {"rounds": 1},
                },
                base_dir=root,
            )

            expected_init = json.loads(init_path.read_text(encoding="utf-8"))
            surviving_path = root / "out" / "client_000" / "surviving_text_it0.json"
            surviving_path.parent.mkdir(parents=True, exist_ok=True)
            surviving_path.write_text(json.dumps(["survivor"]), encoding="utf-8")
            model_paths = ModelPaths(
                minilm=root / "minilm",
                roberta_large=root / "roberta",
                llama2_7b=root / "llama",
                distilgpt2=root / "distilgpt2",
            )

            def fake_stage1(config_arg, dataset_bundle, model_paths_arg, output_dir):
                self.assertEqual(dataset_bundle.initialization_texts, expected_init)
                self.assertEqual(model_paths_arg, model_paths)
                self.assertEqual(output_dir, root / "out" / "client_000")
                return StageSummary(
                    stage_name="stage1",
                    output_dir=output_dir,
                    artifacts={"surviving_files": [str(surviving_path)]},
                    metrics={"epsilon": 1.0, "delta": 1e-6, "rounds": 1},
                )

            with patch("pretext_platform.core.federated_runner.resolve_model_paths", return_value=model_paths), patch(
                "pretext_platform.core.federated_runner.run_private_evolution_stage",
                side_effect=fake_stage1,
            ):
                summary, surviving = _default_stage1_runner(
                    config=config,
                    client_id="client_000",
                    round_id=0,
                    client_partition={"train_texts": ["private text"], "eval_texts": []},
                    output_dir=root / "out" / "client_000",
                )

            self.assertEqual(surviving, ["survivor"])
            self.assertEqual(summary.metrics["surviving_count"], 1)

    def test_partition_builder_reuses_thesis_platform_partition_samples(self) -> None:
        from pretext_platform.core.federated_partition import build_federated_client_partitions

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = ExperimentConfig.from_mapping(
                {
                    "meta": {"experiment_id": "fpt_partition", "seed": 11},
                    "paths": {"repo_root": ".", "output_root": "./out"},
                    "data": {
                        "dataset_name": "jobs",
                        "sample_format": "raw_text",
                        "task_type": "instruction_tuning",
                        "train_path": "./train.json",
                        "eval_path": "./eval.json",
                        "max_samples_per_client": 4,
                    },
                    "federation": {
                        "num_clients": 2,
                        "max_samples_per_client": 4,
                        "validation_ratio": 0.25,
                        "partition_strategy": "preserve_buckets",
                    },
                },
                base_dir=root,
            )
            loaded_samples = [
                SimpleNamespace(
                    sample_id="s0",
                    client_id="raw",
                    round_id=0,
                    source="real",
                    dataset_name="jobs",
                    task_type="instruction_tuning",
                    text="client0 text",
                    rendered_text=lambda: "client0 text",
                    meta={"bucket_id": "0"},
                ),
                SimpleNamespace(
                    sample_id="s1",
                    client_id="raw",
                    round_id=0,
                    source="real",
                    dataset_name="jobs",
                    task_type="instruction_tuning",
                    text="client1 text",
                    rendered_text=lambda: "client1 text",
                    meta={"bucket_id": "1"},
                ),
            ]
            returned_partitions = [
                {"train": [loaded_samples[0]], "validation": [], "all": [loaded_samples[0]]},
                {"train": [loaded_samples[1]], "validation": [], "all": [loaded_samples[1]]},
            ]

            with patch("pretext_platform.core.federated_partition.load_samples", return_value=loaded_samples), patch(
                "pretext_platform.core.federated_partition.partition_samples",
                return_value=returned_partitions,
            ) as mocked_partition:
                partitions = build_federated_client_partitions(config)

            mocked_partition.assert_called_once()
            self.assertEqual(partitions["client_000"]["train_texts"], ["client0 text"])
            self.assertEqual(partitions["client_001"]["train_texts"], ["client1 text"])
            self.assertEqual(mocked_partition.call_args.kwargs["strategy"], "preserve_buckets")

    def test_default_bootstrap_runner_builds_prompt_artifact_from_merged_survivors(self) -> None:
        from pretext_platform.core.federated_runner import _default_bootstrap_runner

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = ExperimentConfig.from_mapping(
                {
                    "meta": {"experiment_id": "fpt_bootstrap", "seed": 13},
                    "paths": {"repo_root": ".", "output_root": "./out"},
                    "bootstrap": {"num_prompts": 2},
                },
                base_dir=root,
            )
            with patch(
                "pretext_platform.core.federated_runner.build_bootstrap_prompts",
                return_value=["prompt one", "prompt two"],
            ) as mocked_prompts:
                summary = _default_bootstrap_runner(
                    config=config,
                    merged_surviving_texts=[
                        "seed text one for prompt building",
                        "seed text two for prompt building",
                        "seed text three for prompt building",
                    ],
                    round_id=0,
                    server_output_dir=root / "out" / "server_stage2",
                )

            mocked_prompts.assert_called_once()
            prompt_path = root / "out" / "server_stage2" / "bootstrap_prompts.json"
            self.assertTrue(prompt_path.exists())
            self.assertEqual(summary.metrics["prompt_count"], 2)
            self.assertEqual(summary.artifacts["bootstrap_prompt_path"], str(prompt_path))

    def test_default_bootstrap_runner_rejects_too_few_survivors(self) -> None:
        from pretext_platform.core.federated_runner import _default_bootstrap_runner

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = ExperimentConfig.from_mapping(
                {
                    "meta": {"experiment_id": "fpt_bootstrap_small", "seed": 13},
                    "paths": {"repo_root": ".", "output_root": "./out"},
                    "bootstrap": {"num_prompts": 2},
                },
                base_dir=root,
            )

            with self.assertRaisesRegex(ValueError, "at least 3 surviving texts"):
                _default_bootstrap_runner(
                    config=config,
                    merged_surviving_texts=["only one", "only two"],
                    round_id=0,
                    server_output_dir=root / "out" / "server_stage2",
                )

    def test_default_stage2_runner_uses_bootstrap_generation_instead_of_echoing_inputs(self) -> None:
        from pretext_platform.core.federated_runner import _default_stage2_runner

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = ExperimentConfig.from_mapping(
                {
                    "meta": {"experiment_id": "fpt_stage2_real", "seed": 7},
                    "paths": {"repo_root": ".", "output_root": "./out"},
                    "models": {"distilgpt2_path": "./distilgpt2"},
                    "bootstrap": {
                        "num_prompts": 2,
                        "generator_model": "distilgpt2",
                        "generator_backend": "delegated",
                        "max_tokens": 8,
                    },
                },
                base_dir=root,
            )
            model_paths = ModelPaths(
                minilm=root / "minilm",
                roberta_large=root / "roberta",
                llama2_7b=root / "llama",
                distilgpt2=root / "distilgpt2",
            )

            with patch("pretext_platform.core.federated_runner.resolve_model_paths", return_value=model_paths), patch(
                "pretext_platform.core.federated_runner.generate_bootstrapped_samples",
                return_value=["synthetic one", "synthetic two"],
            ) as mocked_generate:
                summary = _default_stage2_runner(
                    config=config,
                    merged_surviving_texts=[
                        "seed text one for prompt building",
                        "seed text two for prompt building",
                        "seed text three for prompt building",
                    ],
                    round_id=0,
                    server_output_dir=root / "out" / "server_stage2",
                )

            mocked_generate.assert_called_once()
            self.assertEqual(summary.metrics["generated_count"], 2)
            generated_path = root / "out" / "server_stage2" / "llama7b_text_syn.json"
            self.assertEqual(
                json.loads(generated_path.read_text(encoding="utf-8")),
                ["synthetic one", "synthetic two"],
            )

    def test_default_stage2_runner_supports_huggingface_backend_alias(self) -> None:
        from pretext_platform.core.federated_runner import _default_stage2_runner

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = ExperimentConfig.from_mapping(
                {
                    "meta": {"experiment_id": "fpt_stage2_hf", "seed": 7},
                    "paths": {"repo_root": ".", "output_root": "./out"},
                    "models": {"distilgpt2_path": "./distilgpt2"},
                    "bootstrap": {
                        "num_prompts": 2,
                        "generator_model": "distilgpt2",
                        "generator_backend": "huggingface",
                        "max_tokens": 8,
                    },
                },
                base_dir=root,
            )
            model_paths = ModelPaths(
                minilm=root / "minilm",
                roberta_large=root / "roberta",
                llama2_7b=root / "llama",
                distilgpt2=root / "distilgpt2",
            )

            with patch("pretext_platform.core.federated_runner.resolve_model_paths", return_value=model_paths), patch(
                "pretext_platform.core.federated_runner.generate_bootstrapped_samples_hf",
                return_value=["hf synthetic"],
            ) as mocked_generate_hf:
                summary = _default_stage2_runner(
                    config=config,
                    merged_surviving_texts=[
                        "seed text one for prompt building",
                        "seed text two for prompt building",
                        "seed text three for prompt building",
                    ],
                    round_id=0,
                    server_output_dir=root / "out" / "server_stage2",
                )

            mocked_generate_hf.assert_called_once()
            self.assertEqual(summary.metrics["generated_count"], 1)


if __name__ == "__main__":
    unittest.main()




