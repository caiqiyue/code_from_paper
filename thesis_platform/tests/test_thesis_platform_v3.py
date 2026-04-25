from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from thesis_platform.core.config import load_experiment_config
from thesis_platform.core.experiment_runner import ExperimentRunner
from thesis_platform.core.logging_utils import close_experiment_file_logger
from thesis_platform.core.preflight import validate_preflight
from thesis_platform.evaluation.downstream_eval import _build_pretext_raw


class V3PipelineTests(unittest.TestCase):
    """Validate the v3 real-platform extensions."""

    def test_preflight_reports_missing_jobs_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            config_path = tmp_root / "missing_jobs.yaml"
            config_path.write_text(
                """
meta:
  experiment_id: missing_jobs
paths:
  repo_root: .
  output_root: ./out
  cache_root: ./cache
data:
  dataset_name: jobs
  train_path: ./missing_train.json
  eval_path: ./missing_eval.json
  initialization_path: ./missing_init.json
generator:
  name: pretext_seed
scorer:
  name: datainf_real
  feature_model: ./missing_roberta
  allow_hashing_fallback: false
prototype:
  name: minilm_mean
  embedding_model: ./missing_minilm
  allow_hashing_fallback: false
routing:
  enabled: true
retriever:
  name: knn
  embedding_model: ./missing_minilm
  allow_hashing_fallback: false
critic:
  name: none
aggregator:
  name: none
downstream_eval:
  enabled: false
llm:
  client:
    engine: mock
    model_name_or_path: ""
  server:
    engine: mock
    model_name_or_path: ""
""".strip(),
                encoding="utf-8",
            )
            config = load_experiment_config(config_path)
            with patch("thesis_platform.core.preflight._module_available", return_value=True):
                with self.assertRaises(ValueError) as error:
                    validate_preflight(config)
            message = str(error.exception)
            self.assertIn("missing jobs train dataset", message)
            self.assertIn("missing prototype model", message)
            self.assertIn("missing scorer feature model", message)

    def test_preflight_rejects_preserve_buckets_when_dataset_has_no_bucket_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            train_path = tmp_root / "train.json"
            train_path.write_text(json.dumps(["alpha", "beta", "gamma"]), encoding="utf-8")
            config_path = tmp_root / "preserve_buckets_invalid.yaml"
            config_path.write_text(
                f"""
meta:
  experiment_id: preserve_buckets_invalid
paths:
  repo_root: "{tmp_root.as_posix()}"
  output_root: "{(tmp_root / 'out').as_posix()}"
  cache_root: "{(tmp_root / 'cache').as_posix()}"
data:
  dataset_name: tmp
  task_type: instruction_tuning
  sample_format: raw_text
  train_path: "{train_path.as_posix()}"
  num_clients: 2
  partition_strategy: preserve_buckets
generator:
  name: pretext_seed
scorer:
  name: none
retriever:
  name: none
critic:
  name: none
aggregator:
  name: none
downstream_eval:
  enabled: false
llm:
  client:
    engine: mock
    model_name_or_path: ""
  server:
    engine: mock
    model_name_or_path: ""
""".strip(),
                encoding="utf-8",
            )

            config = load_experiment_config(config_path)
            with self.assertRaises(ValueError) as error:
                validate_preflight(config)

            self.assertIn("partition_strategy='preserve_buckets'", str(error.exception))

    def test_preflight_accepts_congressional_month_buckets_recovered_from_raw_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            dataset_root = tmp_root / "congressional"
            formatted_dir = dataset_root / "formatted"
            raw_dir = dataset_root / "raw"
            formatted_dir.mkdir(parents=True, exist_ok=True)
            raw_dir.mkdir(parents=True, exist_ok=True)

            monthly_records = []
            train_texts = []
            for month_index in range(1, 4):
                text = " ".join([f"speech{month_index}"] * 24)
                train_texts.append(text)
                monthly_records.append(
                    {
                        "url": f"https://api.openparliament.ca/debates/2026/{month_index}/9/example-{month_index}/",
                        "date_str": f"2026-0{month_index}-09",
                        "title": "Sample Debate",
                        "speaker": f"Hon. Speaker {month_index}",
                        "data": text,
                        "chamber": "House of Commons",
                        "country": "CA",
                    }
                )

            (formatted_dir / "congressional_train.json").write_text(
                json.dumps(train_texts),
                encoding="utf-8",
            )
            (raw_dir / "congressional_data_2026-01.json").write_text(
                json.dumps(monthly_records),
                encoding="utf-8",
            )

            config_path = tmp_root / "preserve_buckets_congressional.yaml"
            config_path.write_text(
                f"""
meta:
  experiment_id: preserve_buckets_congressional
paths:
  repo_root: "{tmp_root.as_posix()}"
  output_root: "{(tmp_root / 'out').as_posix()}"
  cache_root: "{(tmp_root / 'cache').as_posix()}"
data:
  dataset_name: congressional
  task_type: instruction_tuning
  sample_format: raw_text
  train_path: "{(formatted_dir / 'congressional_train.json').as_posix()}"
  num_clients: 3
  partition_strategy: preserve_buckets
generator:
  name: pretext_seed
scorer:
  name: none
retriever:
  name: none
critic:
  name: none
aggregator:
  name: none
downstream_eval:
  enabled: false
llm:
  client:
    engine: mock
    model_name_or_path: ""
  server:
    engine: mock
    model_name_or_path: ""
""".strip(),
                encoding="utf-8",
            )

            config = load_experiment_config(config_path)
            validate_preflight(config)

    def test_selector_method_config_sets_federation_selector(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            repo_root = Path(__file__).resolve().parents[1]
            random_selector_path = repo_root / "configs" / "methods" / "selectors" / "random.yaml"
            config_path = tmp_root / "selector_random.yaml"
            config_path.write_text(
                f"""
inherits:
  - "{random_selector_path.as_posix()}"
federation:
  top_k_bad: 2
""".strip(),
                encoding="utf-8",
            )

            config = load_experiment_config(config_path)
            self.assertEqual(config.federation["selector"], "random")
            self.assertEqual(config.federation["top_k_bad"], 2)

    def test_preflight_reports_missing_cross_domain_target_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            train_path = tmp_root / "train.json"
            train_path.write_text(json.dumps(["alpha", "beta"]), encoding="utf-8")
            config_path = tmp_root / "missing_cross_domain.yaml"
            config_path.write_text(
                f"""
meta:
  experiment_id: missing_cross_domain
paths:
  repo_root: "{tmp_root.as_posix()}"
  output_root: "{(tmp_root / 'out').as_posix()}"
  cache_root: "{(tmp_root / 'cache').as_posix()}"
data:
  dataset_name: tmp
  task_type: instruction_tuning
  sample_format: raw_text
  train_path: "{train_path.as_posix()}"
generator:
  name: pretext_seed
scorer:
  name: none
retriever:
  name: none
critic:
  name: none
aggregator:
  name: none
cross_domain_eval:
  enabled: true
  target_dataset: forums
  target_train_path: thesis_platform/datasets/pretext_forums/formatted/missing_train.json
downstream_eval:
  enabled: false
llm:
  client:
    engine: mock
    model_name_or_path: ""
  server:
    engine: mock
    model_name_or_path: ""
""".strip(),
                encoding="utf-8",
            )
            config = load_experiment_config(config_path)
            with patch("thesis_platform.core.preflight._module_available", return_value=True):
                with self.assertRaises(ValueError) as error:
                    validate_preflight(config)

            self.assertIn("missing cross-domain forums train dataset", str(error.exception))

    def test_preflight_reports_missing_local_lora_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            train_path = tmp_root / "train.json"
            train_path.write_text(json.dumps(["alpha", "beta"]), encoding="utf-8")
            config_path = tmp_root / "missing_lora_model.yaml"
            config_path.write_text(
                f"""
meta:
  experiment_id: missing_lora_model
paths:
  repo_root: "{tmp_root.as_posix()}"
  output_root: "{(tmp_root / 'out').as_posix()}"
  cache_root: "{(tmp_root / 'cache').as_posix()}"
data:
  dataset_name: tmp
  task_type: instruction_tuning
  sample_format: raw_text
  train_path: "{train_path.as_posix()}"
generator:
  name: pretext_seed
scorer:
  name: datainf_lora
  use_real_gradients: true
  model_name: thesis_platform/open_model/qwen_2_0_5b_instruct
retriever:
  name: none
critic:
  name: none
aggregator:
  name: none
downstream_eval:
  enabled: false
llm:
  client:
    engine: mock
    model_name_or_path: ""
  server:
    engine: mock
    model_name_or_path: ""
""".strip(),
                encoding="utf-8",
            )
            config = load_experiment_config(config_path)
            with patch("thesis_platform.core.preflight._module_available", return_value=True):
                with self.assertRaises(ValueError) as error:
                    validate_preflight(config)

            self.assertIn("missing scorer LoRA base model", str(error.exception))

    def test_v3_pipeline_writes_routing_and_downstream_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            train_path = tmp_root / "train.json"
            seed_path = tmp_root / "seed.json"
            eval_path = tmp_root / "eval.json"
            (tmp_root / "thesis_platform" / "open_model" / "llama_2_7b_hf").mkdir(parents=True, exist_ok=True)
            (tmp_root / "thesis_platform" / "open_model").mkdir(parents=True, exist_ok=True)
            train_path.write_text(
                json.dumps(
                    {
                        "0": ["job interview notes", "resume screening checklist"],
                        "1": ["software hiring rubric", "candidate scorecard"],
                    }
                ),
                encoding="utf-8",
            )
            seed_path.write_text(json.dumps(["seed hiring memo", "seed recruiter prompt", "seed role summary"]), encoding="utf-8")
            eval_path.write_text(json.dumps(["job posting eval text", "candidate pipeline eval text"]), encoding="utf-8")

            config_path = tmp_root / "v3_tmp.yaml"
            config_path.write_text(
                f"""
meta:
  experiment_id: tmp_v3
  seed: 7
paths:
  repo_root: "{tmp_root.as_posix()}"
  output_root: "{(tmp_root / 'out').as_posix()}"
  cache_root: "{(tmp_root / 'cache').as_posix()}"
llm:
  client:
    engine: mock
    model_name_or_path: ""
  server:
    engine: mock
    model_name_or_path: ""
data:
  dataset_name: tmp
  task_type: instruction_tuning
  sample_format: raw_text
  partition_strategy: preserve_buckets
  train_path: "{train_path.as_posix()}"
  eval_path: "{eval_path.as_posix()}"
  public_seed_path: "{seed_path.as_posix()}"
  initialization_path: "{seed_path.as_posix()}"
  max_public_seed_samples: 3
  num_clients: 2
  max_samples_per_client: 2
  validation_ratio: 0.5
federation:
  rounds: 2
  top_k_bad: 1
generator:
  name: pretext_prompt_llm
  initial_prompt: "Generate hiring-domain text."
  generated_per_round: 4
  exemplars_per_prompt: 1
scorer:
  name: datainf_real
  feature_model: ""
  allow_hashing_fallback: true
retriever:
  name: knn
  top_k: 1
  embedding_model: missing-model
  allow_hashing_fallback: true
critic:
  name: fedtextgrad_llm
  compress_to_n_rules: 2
  redact_enable: true
aggregator:
  name: dbscan_attn_tsgdm
  max_rules: 2
  embedding_model: missing-model
  allow_hashing_fallback: true
  cluster_eps: 0.5
  cluster_min_samples: 1
  momentum_beta: 0.6
prototype:
  name: minilm_mean
routing:
  enabled: true
  personalized_mix_ratio: 0.5
  cluster_eps: 0.5
  cluster_min_samples: 1
downstream_eval:
  enabled: true
  kind: pretext_large_eval
  export_filename: llama7b_text_syn.json
  windows_large_eval_mode: peft_lora
  linux_large_eval_mode: peft_lora
  guard_windows_llama2_large_eval: false
""".strip(),
                encoding="utf-8",
            )

            config = load_experiment_config(config_path)
            fake_eval_summary = {
                "stage_name": "eval_large",
                "metrics": {"best_top1": 0.42},
                "artifacts": {"stats_dir": str(tmp_root / "fake_stats")},
            }
            with patch("thesis_platform.core.experiment_runner.validate_preflight", return_value=None), patch(
                "thesis_platform.evaluation.downstream_eval.run_pretext_large_eval",
                return_value=fake_eval_summary,
            ):
                summary = ExperimentRunner(config).run()

            exp_dir = Path(summary["experiment_dir"])
            round_1 = exp_dir / "round_001"
            with (round_1 / "round_metrics.json").open("r", encoding="utf-8") as handle:
                round_metrics = json.load(handle)
            with (round_1 / "routing_summary.json").open("r", encoding="utf-8") as handle:
                routing_summary = json.load(handle)
            with (exp_dir / "artifact_manifest.json").open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertTrue((round_1 / "client_prototypes.jsonl").exists())
            self.assertTrue((round_1 / "cluster_assignments.json").exists())
            self.assertTrue((round_1 / "cluster_prompts.json").exists())
            self.assertTrue((round_1 / "routing_summary.json").exists())
            self.assertTrue((exp_dir / "downstream_eval" / "stage2" / "llama7b_text_syn.json").exists())
            self.assertTrue((exp_dir / "downstream_eval" / "downstream_eval_summary.json").exists())
            self.assertTrue((exp_dir / "downstream_eval" / "pretext_large_eval_summary.json").exists())
            self.assertTrue((exp_dir / "downstream_eval" / "pretext_small_eval_summary.json").exists())
            self.assertTrue((exp_dir / "privacy_ledger.json").exists())
            self.assertEqual(round_metrics["artifact_type"], "round_metrics")
            self.assertEqual(routing_summary["artifact_type"], "routing_summary")
            self.assertEqual(round_metrics["schema_version"], "thesis_platform.runtime.v1")
            self.assertEqual(manifest["artifact_type"], "experiment_manifest")
            self.assertEqual(len(manifest["rounds"]), 2)
            self.assertEqual(summary["downstream_eval"]["metrics"]["best_top1"], 0.42)

    def test_cross_domain_eval_uses_target_downstream_manager(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            source_train = tmp_root / "source_train.json"
            target_train = tmp_root / "target_train.json"
            source_train.write_text(json.dumps(["source one", "source two"]), encoding="utf-8")
            target_train.write_text(json.dumps(["target one", "target two"]), encoding="utf-8")
            config_path = tmp_root / "cross_domain.yaml"
            config_path.write_text(
                f"""
meta:
  experiment_id: cross_domain_demo
paths:
  repo_root: "{tmp_root.as_posix()}"
  output_root: "{(tmp_root / 'out').as_posix()}"
  cache_root: "{(tmp_root / 'cache').as_posix()}"
data:
  dataset_name: jobs
  task_type: instruction_tuning
  sample_format: raw_text
  train_path: "{source_train.as_posix()}"
generator:
  name: pretext_seed
scorer:
  name: none
retriever:
  name: none
critic:
  name: none
aggregator:
  name: none
cross_domain_eval:
  enabled: true
  target_dataset: forums
  target_train_path: "{target_train.as_posix()}"
downstream_eval:
  enabled: false
llm:
  client:
    engine: mock
    model_name_or_path: ""
  server:
    engine: mock
    model_name_or_path: ""
""".strip(),
                encoding="utf-8",
            )
            config = load_experiment_config(config_path)
            runner = ExperimentRunner(config)
            fake_transfer_summary = {
                "status": "completed",
                "message": "",
                "metrics": {"best_top1": 0.61},
                "synthetic_corpus_path": str(runner.experiment_dir / "cross_domain_eval" / "forums" / "downstream_eval" / "stage2" / "llama7b_text_syn.json"),
                "stage2_dir": str(runner.experiment_dir / "cross_domain_eval" / "forums" / "downstream_eval" / "stage2"),
            }
            with patch.object(ExperimentRunner, "_write_run_state", return_value=None), patch(
                "thesis_platform.core.experiment_runner.DownstreamEvalManager.run",
                return_value=fake_transfer_summary,
            ):
                summary = runner._run_cross_domain_eval(
                    synthetic_texts=["synthetic example"],
                    cross_domain_cfg=config.cross_domain_eval,
                    current_round=2,
                    rounds_total=3,
                    checkpoint_path=None,
                )
            close_experiment_file_logger()

            cross_domain_root = runner.experiment_dir / "cross_domain_eval" / "forums"
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["metrics"]["best_top1"], 0.61)
            self.assertEqual(summary["target_dataset"], "forums")
            self.assertTrue((cross_domain_root / "cross_domain_summary.json").exists())
            self.assertTrue((cross_domain_root / "cross_domain_results.json").exists())
            self.assertTrue((cross_domain_root / "synthetic_corpus.json").exists())

    def test_run_pretext_glue_eval_writes_task_summary(self) -> None:
        from thesis_platform.evaluation.downstream_eval import run_pretext_glue_eval

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            corpus_dir = tmp_root / "stage2"
            corpus_dir.mkdir(parents=True, exist_ok=True)
            (corpus_dir / "llama7b_text_syn.json").write_text(json.dumps(["synthetic text"]), encoding="utf-8")
            config_path = tmp_root / "glue_eval.yaml"
            config_path.write_text(
                f"""
meta:
  experiment_id: glue_eval_demo
paths:
  repo_root: "{tmp_root.as_posix()}"
  output_root: "{(tmp_root / 'out').as_posix()}"
  cache_root: "{(tmp_root / 'cache').as_posix()}"
data:
  dataset_name: jobs
  task_type: instruction_tuning
  sample_format: raw_text
  train_path: "{(tmp_root / 'train.json').as_posix()}"
downstream_eval:
  enabled: false
llm:
  client:
    engine: mock
    model_name_or_path: ""
  server:
    engine: mock
    model_name_or_path: ""
""".strip(),
                encoding="utf-8",
            )
            (tmp_root / "train.json").write_text(json.dumps(["alpha"]), encoding="utf-8")
            config = load_experiment_config(config_path)
            fake_stage_summary = {
                "stage_name": "glue_eval_sst2",
                "output_dir": str(tmp_root / "glue_eval"),
                "artifacts": {"stats_dir": str(tmp_root / "glue_eval" / "glue_sst2_eval")},
                "metrics": {"best_accuracy": 0.88, "correct": 44, "total": 50},
                "message": "",
                "skipped": False,
            }
            pretext_root = Path(__file__).resolve().parents[2] / "PrE-Text"
            sys.path.insert(0, str(pretext_root))
            with patch("pretext_platform.core.models.resolve_model_paths", return_value=object()), patch(
                "pretext_platform.evaluation.glue_classification_eval.run_glue_classification_eval",
                return_value=fake_stage_summary,
            ):
                summary = run_pretext_glue_eval(
                    config,
                    stage2_dir=corpus_dir,
                    output_dir=tmp_root / "glue_eval",
                    tasks=["sst2"],
                )
            sys.path.remove(str(pretext_root))

            self.assertEqual(summary["tasks"]["sst2"]["metrics"]["best_accuracy"], 0.88)
            self.assertTrue((tmp_root / "glue_eval" / "glue_sst2_summary.json").exists())
            self.assertTrue((tmp_root / "glue_eval" / "glue_summary.json").exists())

    def test_build_pretext_raw_preserves_dataset_limits(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        config = load_experiment_config(
            repo_root / "paper-new" / "configs" / "experiments" / "single_node_screening" / "ns_s_jobs_screening.yaml"
        )

        raw = _build_pretext_raw(
            config,
            output_dir=repo_root / "paper-new" / "outputs" / "test_thesis_platform_v3",
            enable_large_eval=False,
            enable_small_eval=True,
        )

        self.assertEqual(raw["data"].get("train_limit"), 256)
        self.assertEqual(raw["data"].get("eval_limit"), 256)
        self.assertEqual(raw["data"].get("initialization_limit"), 1024)


if __name__ == "__main__":
    unittest.main()
