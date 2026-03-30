from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from thesis_platform.core.config import load_experiment_config
from thesis_platform.core.context import ClientContext, ServerContext
from thesis_platform.core.experiment_runner import ExperimentRunner
from thesis_platform.core.logging_utils import close_experiment_file_logger
from thesis_platform.core.preflight import validate_preflight
from thesis_platform.core.privacy import PrivacyLedger, PrivacyPolicy
from thesis_platform.core.round_runner import RoundRunner
from thesis_platform.core.schemas import ScoredSample, Sample
from thesis_platform.evaluation.downstream_eval import DownstreamEvalManager


class _FakeGenerator:
    def __init__(self, *, fail_if_called: bool = False) -> None:
        self.fail_if_called = fail_if_called
        self.calls = 0

    def generate(self, round_ctx):
        self.calls += 1
        if self.fail_if_called:
            raise AssertionError("generation should not rerun during resume")
        return [
            Sample(
                sample_id=f"syn_{round_ctx.round_id}_{index}",
                client_id="server",
                round_id=round_ctx.round_id,
                source="synthetic",
                dataset_name="tmp",
                task_type="instruction_tuning",
                text=f"synthetic sample {index}",
            )
            for index in range(2)
        ]


class _FakeScorer:
    def __init__(self, *, fail_client_id: str | None = None) -> None:
        self.fail_client_id = fail_client_id
        self.calls: list[str] = []

    def score(self, samples, client_ctx):
        self.calls.append(client_ctx.client_id)
        if client_ctx.client_id == self.fail_client_id:
            raise RuntimeError(f"boom-{client_ctx.client_id}")
        return [
            ScoredSample.from_sample(
                sample,
                client_id=client_ctx.client_id,
                score=float(index + 1),
                score_name="fake",
            )
            for index, sample in enumerate(samples)
        ]


class _FakeRetriever:
    def retrieve(self, client_selected, client_ctx):
        del client_ctx
        return []


class _FakeCritic:
    def critique(self, paired_samples, client_ctx):
        del paired_samples, client_ctx
        return []


class _FakeAggregator:
    def aggregate(self, client_critiques, server_ctx):
        del client_critiques, server_ctx
        return None


class ResumeBehaviorTests(unittest.TestCase):
    def _build_client_contexts(self) -> list[ClientContext]:
        contexts: list[ClientContext] = []
        for index in range(2):
            sample = Sample(
                sample_id=f"real_{index}",
                client_id=f"client_{index}",
                round_id=0,
                source="real",
                dataset_name="tmp",
                task_type="instruction_tuning",
                text=f"real sample {index}",
            )
            contexts.append(
                ClientContext(
                    client_id=f"client_{index}",
                    train_samples=[sample],
                    validation_samples=[sample],
                    all_samples=[sample],
                    embedder=object(),
                    config={},
                )
            )
        return contexts

    def test_round_runner_resumes_from_partial_client_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            round_dir = Path(tmp_dir) / "round_000"
            server_ctx = ServerContext(
                experiment_id="resume_round",
                prompt_text="prompt",
                prompt_history=["prompt"],
                config={},
                output_dir=round_dir,
            )
            privacy_ledger = PrivacyLedger(policy=PrivacyPolicy.from_config({"enabled": False}))

            first_runner = RoundRunner(
                generator=_FakeGenerator(),
                scorer=_FakeScorer(fail_client_id="client_1"),
                retriever=_FakeRetriever(),
                critic=_FakeCritic(),
                aggregator=_FakeAggregator(),
            )
            with self.assertRaises(RuntimeError):
                first_runner.run_round(
                    round_id=0,
                    server_ctx=server_ctx,
                    client_contexts=self._build_client_contexts(),
                    public_seed_samples=[],
                    federation_cfg={"top_k_bad": 1},
                    output_dir=round_dir,
                    routing_cfg={"enabled": False},
                    privacy_ledger=privacy_ledger,
                )

            stage_state = json.loads((round_dir / "round_stage_state.json").read_text(encoding="utf-8"))
            self.assertEqual(stage_state["stage"], "client_analysis_in_progress")
            self.assertEqual(stage_state["completed_clients"], ["client_0"])
            self.assertTrue((round_dir / "round_privacy_ledger.json").exists())

            resume_generator = _FakeGenerator(fail_if_called=True)
            resume_scorer = _FakeScorer()
            resumed_runner = RoundRunner(
                generator=resume_generator,
                scorer=resume_scorer,
                retriever=_FakeRetriever(),
                critic=_FakeCritic(),
                aggregator=_FakeAggregator(),
            )
            resumed_artifacts = resumed_runner.run_round(
                round_id=0,
                server_ctx=ServerContext(
                    experiment_id="resume_round",
                    prompt_text="prompt",
                    prompt_history=["prompt"],
                    config={},
                    output_dir=round_dir,
                ),
                client_contexts=self._build_client_contexts(),
                public_seed_samples=[],
                federation_cfg={"top_k_bad": 1},
                output_dir=round_dir,
                routing_cfg={"enabled": False},
                privacy_ledger=PrivacyLedger(policy=PrivacyPolicy.from_config({"enabled": False})),
            )

            final_stage_state = json.loads((round_dir / "round_stage_state.json").read_text(encoding="utf-8"))
            self.assertEqual(resume_generator.calls, 0)
            self.assertEqual(resume_scorer.calls, ["client_1"])
            self.assertEqual(final_stage_state["stage"], "completed")
            self.assertEqual(len(resumed_artifacts.selected_bad_samples), 2)

    def test_resolve_resume_directory_rejects_multiple_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            config_path = tmp_root / "resume_conflict.yaml"
            config_path.write_text(
                f"""
meta:
  experiment_id: resume_conflict
paths:
  repo_root: "{tmp_root.as_posix()}"
  output_root: "{(tmp_root / 'out').as_posix()}"
  cache_root: "{(tmp_root / 'cache').as_posix()}"
data:
  dataset_name: tmp
  train_path: "{(tmp_root / 'train.json').as_posix()}"
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
llm:
  client:
    engine: mock
    model_name_or_path: ""
  server:
    engine: mock
    model_name_or_path: ""
downstream_eval:
  enabled: false
""".strip(),
                encoding="utf-8",
            )
            (tmp_root / "train.json").write_text(json.dumps(["alpha"]), encoding="utf-8")
            runner = ExperimentRunner(load_experiment_config(config_path))
            try:
                registry_dir = tmp_root / "out" / "run_registry" / "resume_conflict"
                registry_dir.mkdir(parents=True, exist_ok=True)
                for index in range(2):
                    experiment_dir = tmp_root / "out" / f"resume_conflict_20260101_00000{index}"
                    experiment_dir.mkdir(parents=True, exist_ok=True)
                    run_state_path = experiment_dir / "run_state.json"
                    run_state_path.write_text(
                        json.dumps(
                            {
                                "status": "failed",
                                "updated_at": f"2026-01-01T00:00:0{index}Z",
                                "pid": 0,
                                "hostname": "local-test",
                            }
                        ),
                        encoding="utf-8",
                    )
                    (registry_dir / f"resume_conflict_20260101_00000{index}.json").write_text(
                        json.dumps(
                            {
                                "experiment_dir": str(experiment_dir),
                                "run_state_path": str(run_state_path),
                                "updated_at": f"2026-01-01T00:00:0{index}Z",
                                "status": "failed",
                            }
                        ),
                        encoding="utf-8",
                    )

                with self.assertRaises(ValueError):
                    runner._resolve_resume_directory()
                chosen = runner._resolve_resume_directory(resume_dir=str(tmp_root / "out" / "resume_conflict_20260101_000000"))
                self.assertEqual(chosen, tmp_root / "out" / "resume_conflict_20260101_000000")
            finally:
                close_experiment_file_logger("thesis_platform")

    def test_downstream_eval_reuses_completed_stage_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            train_path = tmp_root / "train.json"
            eval_path = tmp_root / "eval.json"
            init_path = tmp_root / "init.json"
            train_path.write_text(json.dumps(["alpha beta"]), encoding="utf-8")
            eval_path.write_text(json.dumps(["eval beta"]), encoding="utf-8")
            init_path.write_text(json.dumps(["init gamma delta epsilon"]), encoding="utf-8")
            config_path = tmp_root / "downstream_resume.yaml"
            config_path.write_text(
                f"""
meta:
  experiment_id: downstream_resume
paths:
  repo_root: "{tmp_root.as_posix()}"
  output_root: "{(tmp_root / 'out').as_posix()}"
  cache_root: "{(tmp_root / 'cache').as_posix()}"
data:
  dataset_name: tmp
  train_path: "{train_path.as_posix()}"
  eval_path: "{eval_path.as_posix()}"
  initialization_path: "{init_path.as_posix()}"
downstream_eval:
  enabled: true
  kind: pretext_large_eval
  run_large_eval: true
  run_small_eval: false
""".strip(),
                encoding="utf-8",
            )
            config = load_experiment_config(config_path)
            output_dir = tmp_root / "out" / "downstream_eval"
            output_dir.mkdir(parents=True, exist_ok=True)
            completed_payload = {
                "schema_version": "thesis_platform.runtime.v1",
                "artifact_type": "downstream_eval_large_eval_summary",
                "experiment_id": "downstream_resume",
                "stage_key": "large_eval",
                "stage_name": "eval_large",
                "enabled": True,
                "status": "completed",
                "message": "reused",
                "output_dir": str(output_dir / "pretext_large_eval"),
                "metrics": {"best_top1": 0.77},
                "artifacts": {},
                "missing_assets": [],
            }
            (output_dir / "pretext_large_eval_summary.json").write_text(json.dumps(completed_payload), encoding="utf-8")
            with patch("thesis_platform.evaluation.downstream_eval.run_pretext_large_eval", side_effect=AssertionError("should not rerun")):
                summary = DownstreamEvalManager(
                    config,
                    experiment_id="downstream_resume",
                    output_dir=output_dir,
                ).run(["synthetic one"])
            self.assertEqual(summary["stages"]["large_eval"]["metrics"]["best_top1"], 0.77)


class PreflightCoverageTests(unittest.TestCase):
    def test_preflight_reports_uid_llm_missing_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            train_path = tmp_root / "train.json"
            train_path.write_text(json.dumps(["alpha"]), encoding="utf-8")
            config_path = tmp_root / "uid_llm_missing.yaml"
            config_path.write_text(
                f"""
meta:
  experiment_id: uid_llm_missing
paths:
  repo_root: "{tmp_root.as_posix()}"
  output_root: "{(tmp_root / 'out').as_posix()}"
  cache_root: "{(tmp_root / 'cache').as_posix()}"
data:
  dataset_name: tmp
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
  name: uid_llm
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
            availability = {"numpy": False, "sklearn": False}
            with patch("thesis_platform.core.preflight._module_available", side_effect=lambda name: availability.get(name, True)):
                with self.assertRaises(ValueError) as error:
                    validate_preflight(config)
            message = str(error.exception)
            self.assertIn("missing Python package: numpy", message)
            self.assertIn("missing Python package: scikit-learn", message)

    def test_preflight_reports_missing_paper_scorer_base_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            train_path = tmp_root / "train.json"
            train_path.write_text(json.dumps(["alpha"]), encoding="utf-8")
            config_path = tmp_root / "paper_missing_model.yaml"
            config_path.write_text(
                f"""
meta:
  experiment_id: paper_missing_model
paths:
  repo_root: "{tmp_root.as_posix()}"
  output_root: "{(tmp_root / 'out').as_posix()}"
  cache_root: "{(tmp_root / 'cache').as_posix()}"
data:
  dataset_name: tmp
  train_path: "{train_path.as_posix()}"
generator:
  name: pretext_seed
scorer:
  name: datainf_paper
  use_real_gradients: true
  model_name: thesis_platform/open_model/missing_paper_model
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
            self.assertIn("missing scorer base model", str(error.exception))


if __name__ == "__main__":
    unittest.main()




