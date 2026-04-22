from __future__ import annotations

import random
import sys
import tempfile
import types
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from thesis_platform.core.config import load_experiment_config
from thesis_platform.core.context import ServerContext
from thesis_platform.core.schemas import Critique, PairedSample, PromptUpdate, Sample, ScoredSample
from thesis_platform.core.single_node_runner import SingleNodeRunner


class _ConfigStub:
    def __init__(self, output_root: Path) -> None:
        self.raw = {}
        self.meta = {"experiment_id": "single_node_stage_a_test", "seed": 17}
        self.data = {"dataset_name": "demo", "task_type": "instruction_tuning"}
        self.generator = {"initial_prompt": "Base prompt"}
        self.retriever = {}
        self.critic = {}
        self.scorer = {"name": "primary"}
        self.aggregator = {"name": "dbscan_attn_tsgdm"}
        self.stage_a = {
            "generated_count": 4,
            "select_top_k": 2,
            "max_iterations": 1,
            "convergence_threshold": -1.0,
            "max_probe_samples": 8,
            "failure_equal_epsilon": 1e-9,
            "failure_margin_threshold": 0.25,
            "random_fallback_seed": 7,
        }
        self.stage_b = {"generated_count": 4}
        self._output_root = output_root

    def output_root(self) -> Path:
        return self._output_root

    def resolve_path(self, value):
        return Path(value) if value else None

    def repo_root(self) -> Path:
        return self._output_root


class _StaticScorer:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def score(self, samples, client_ctx):
        del client_ctx
        return [
            ScoredSample.from_sample(
                sample,
                client_id="single_node_client",
                score=self._scores[index],
                score_name="test",
            )
            for index, sample in enumerate(samples)
        ]


class _TrackingAggregator:
    def __init__(self, *, rules: list[str] | None = None) -> None:
        self.called = False
        self.last_critiques = None
        self.last_server_ctx = None
        self._rules = rules or []

    def aggregate(self, critiques, server_ctx):
        self.called = True
        self.last_critiques = critiques
        self.last_server_ctx = server_ctx
        if not self._rules:
            return None
        return PromptUpdate(
            update_id="update_0",
            round_id=0,
            rules=list(self._rules),
            summary="summary",
            prompt_text="unused",
        )


class _SingleNodeStageATests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.output_root = Path(self.tmp_dir.name)
        self.config = _ConfigStub(self.output_root)
        self.seed_samples = [
            Sample("seed_0", "client", 0, "seed", "demo", "instruction_tuning", "seed alpha"),
            Sample("seed_1", "client", 0, "seed", "demo", "instruction_tuning", "seed beta"),
        ]
        self.generated_samples = [
            Sample(f"syn_{idx}", "server", 0, "synthetic", "demo", "instruction_tuning", f"synthetic {idx}")
            for idx in range(4)
        ]

    def _make_runner(self, scorer, aggregator):
        runner = SingleNodeRunner(
            generator=SimpleNamespace(),
            scorer=scorer,
            retriever=SimpleNamespace(),
            critic=SimpleNamespace(),
            aggregator=aggregator,
            config=self.config,
        )
        selected_ids: list[str] = []

        def _retrieve(samples, client_ctx):
            del client_ctx
            selected_ids[:] = [sample.sample_id for sample in samples]
            return [
                PairedSample(
                    pair_id=f"pair_{sample.sample_id}",
                    client_id="single_node_client",
                    round_id=0,
                    bad_sample=sample,
                    real_samples=self.seed_samples[:1],
                )
                for sample in samples
            ]

        runner._load_seed_corpus = lambda train_path: list(self.seed_samples)  # type: ignore[method-assign]
        runner._generate_with_prompt = lambda seed_corpus, total_count, prompt_text: list(self.generated_samples)  # type: ignore[method-assign]
        runner._build_client_context = lambda train_samples, all_samples, **kwargs: SimpleNamespace(  # type: ignore[method-assign]
            embedder=None,
            text_backend=None,
            train_samples=train_samples,
            all_samples=all_samples,
            aggregation_memory={"entries": []},
            prototype_feedbacks=[],
            objective_type=kwargs.get("objective_type", "domain_probe"),
        )
        runner._retrieve_batched = _retrieve  # type: ignore[method-assign]
        runner._critique_batched = lambda paired_samples, client_ctx: [  # type: ignore[method-assign]
            Critique(
                critique_id=f"critique_{pair.bad_sample.sample_id}",
                client_id="single_node_client",
                round_id=0,
                bad_sample_id=pair.bad_sample.sample_id,
                real_sample_ids=[sample.sample_id for sample in pair.real_samples],
                rules=[f"Fix {pair.bad_sample.sample_id}"],
                text=f"Fix {pair.bad_sample.sample_id}",
            )
            for pair in paired_samples
        ]
        runner._build_server_context = lambda: ServerContext(  # type: ignore[method-assign]
            experiment_id="exp",
            prompt_text="Base prompt",
            prompt_history=["Base prompt"],
            config={},
            output_dir=self.output_root,
            text_backend=None,
            aggregation_memory={"entries": []},
        )
        return runner, selected_ids

    def test_load_seed_corpus_honors_train_limit_for_smoke_configs(self) -> None:
        train_path = self.output_root / "train.json"
        train_path.write_text(
            json.dumps([{"text": f"seed {idx}"} for idx in range(5)]),
            encoding="utf-8",
        )
        self.config.data["train_limit"] = 2
        runner = SingleNodeRunner(
            generator=SimpleNamespace(),
            scorer=SimpleNamespace(),
            retriever=SimpleNamespace(),
            critic=SimpleNamespace(),
            aggregator=SimpleNamespace(),
            config=self.config,
        )

        samples = runner._load_seed_corpus(train_path)

        self.assertEqual([sample.text for sample in samples], ["seed 0", "seed 1"])

    def test_stage_a_randomly_selects_samples_when_scores_have_no_signal(self) -> None:
        scorer = _StaticScorer([0.0, 0.0, 0.0, 0.0])
        aggregator = _TrackingAggregator()
        runner, selected_ids = self._make_runner(scorer, aggregator)
        fake_dbscan_core = types.SimpleNamespace(
            aggregate_dbscan_critiques=lambda **kwargs: (None, {"entries": []})
        )

        with patch.dict(sys.modules, {"thesis_platform.algorithms.aggregators.dbscan_core": fake_dbscan_core}):
            runner.run_stage_a(self.output_root)

        expected = random.Random(self.config.stage_a["random_fallback_seed"]).sample(
            [sample.sample_id for sample in self.generated_samples],
            self.config.stage_a["select_top_k"],
        )
        self.assertEqual(selected_ids, expected)

    def test_stage_a_uses_injected_aggregator_for_prompt_updates(self) -> None:
        scorer = _StaticScorer([0.9, 0.5, 0.2, 0.1])
        aggregator = _TrackingAggregator(rules=["Use more domain detail"])
        runner, _selected_ids = self._make_runner(scorer, aggregator)
        fake_dbscan_core = types.SimpleNamespace(
            aggregate_dbscan_critiques=lambda **kwargs: (None, {"entries": []})
        )

        with patch.dict(sys.modules, {"thesis_platform.algorithms.aggregators.dbscan_core": fake_dbscan_core}):
            result = runner.run_stage_a(self.output_root)

        self.assertTrue(aggregator.called)
        self.assertIn("Generated guidance", result["optimized_prompt"])

    def test_stage_a_uses_stage_a_scorer_override_when_present(self) -> None:
        self.config.stage_a["scorer"] = "alternate"
        primary_scorer = _StaticScorer([0.9, 0.8, 0.2, 0.1])
        alternate_scorer = _StaticScorer([0.1, 0.2, 0.8, 0.9])
        aggregator = _TrackingAggregator()
        runner, selected_ids = self._make_runner(primary_scorer, aggregator)
        fake_dbscan_core = types.SimpleNamespace(
            aggregate_dbscan_critiques=lambda **kwargs: (None, {"entries": []})
        )

        with patch("thesis_platform.core.single_node_runner.create", return_value=alternate_scorer, create=True), patch.dict(
            sys.modules,
            {"thesis_platform.algorithms.aggregators.dbscan_core": fake_dbscan_core},
        ):
            runner.run_stage_a(self.output_root)

        self.assertEqual(selected_ids, ["syn_3", "syn_2"])

    def test_stage_a_runs_all_five_iterations_even_when_scores_converge_early(self) -> None:
        self.config.stage_a["max_iterations"] = 5
        self.config.stage_a["convergence_threshold"] = 1.0
        scorer = _StaticScorer([0.9, 0.5, 0.2, 0.1])
        aggregator = _TrackingAggregator()
        runner, _selected_ids = self._make_runner(scorer, aggregator)

        result = runner.run_stage_a(self.output_root)

        self.assertEqual(result["iterations"], 5)
        for idx in range(5):
            selection_path = self.output_root / "stage_a" / f"iteration_{idx}" / "selection_summary.json"
            self.assertTrue(selection_path.exists())

    def test_stage_a_honors_yaml_max_iterations_value(self) -> None:
        self.config.stage_a["max_iterations"] = 3
        scorer = _StaticScorer([0.9, 0.5, 0.2, 0.1])
        aggregator = _TrackingAggregator()
        runner, _selected_ids = self._make_runner(scorer, aggregator)

        result = runner.run_stage_a(self.output_root)

        self.assertEqual(result["iterations"], 3)
        for idx in range(3):
            selection_path = self.output_root / "stage_a" / f"iteration_{idx}" / "selection_summary.json"
            self.assertTrue(selection_path.exists())

    def test_stage_a_allows_ira_for_raw_text_to_match_federated_configs(self) -> None:
        self.config.data["sample_format"] = "raw_text"
        self.config.stage_a["scorer"] = "ira"
        scorer = _StaticScorer([0.9, 0.5, 0.2, 0.1])
        aggregator = _TrackingAggregator()
        runner, _selected_ids = self._make_runner(scorer, aggregator)

        with patch("thesis_platform.core.single_node_runner.create", return_value=scorer, create=True):
            resolved_scorer, resolved_name = runner._resolve_stage_a_scorer()

        self.assertIs(resolved_scorer, scorer)
        self.assertEqual(resolved_name, "ira")

    def test_stage_a_rejects_domain_probe_aliases_for_raw_text_single_node(self) -> None:
        self.config.data["sample_format"] = "raw_text"
        self.config.stage_a["scorer"] = "datainf"
        scorer = _StaticScorer([0.9, 0.5, 0.2, 0.1])
        aggregator = _TrackingAggregator()
        runner, _selected_ids = self._make_runner(scorer, aggregator)

        with self.assertRaises(ValueError) as error:
            runner.run_stage_a(self.output_root)

        self.assertIn("datainf_real", str(error.exception))

    def test_stage_a_stale_cache_is_ignored_when_signature_changes(self) -> None:
        scorer = _StaticScorer([0.9, 0.5, 0.2, 0.1])
        aggregator = _TrackingAggregator()
        runner, _selected_ids = self._make_runner(scorer, aggregator)
        stage_a_dir = self.output_root / "stage_a"
        stage_a_dir.mkdir(parents=True, exist_ok=True)
        (stage_a_dir / "prompt_update.json").write_text(
            json.dumps(
                {
                    "final_prompt": "cached prompt",
                    "prompt_history": ["cached prompt"],
                    "iterations": 1,
                    "cache_signature": {"scorer": "different"},
                }
            ),
            encoding="utf-8",
        )
        called = {"generate": False}

        def _generate(seed_corpus, total_count, prompt_text):
            del seed_corpus, total_count, prompt_text
            called["generate"] = True
            return list(self.generated_samples)

        runner._generate_with_prompt = _generate  # type: ignore[method-assign]

        runner.run_stage_a(self.output_root)

        self.assertTrue(called["generate"])

    def test_stage_b_stale_cache_is_ignored_when_prompt_changes(self) -> None:
        scorer = _StaticScorer([0.9, 0.5, 0.2, 0.1])
        aggregator = _TrackingAggregator()
        runner, _selected_ids = self._make_runner(scorer, aggregator)
        stage_b_dir = self.output_root / "stage_b"
        stage_b_dir.mkdir(parents=True, exist_ok=True)
        (stage_b_dir / "llama7b_text_syn.json").write_text(json.dumps(["cached output"]), encoding="utf-8")
        (stage_b_dir / "stage_config.json").write_text(
            json.dumps({"optimized_prompt": "old prompt", "generated_count": 4}),
            encoding="utf-8",
        )
        called = {"generate": False}

        def _generate(seed_corpus, total_count, prompt_text):
            del seed_corpus, total_count, prompt_text
            called["generate"] = True
            return list(self.generated_samples)

        runner._load_seed_corpus = lambda train_path: list(self.seed_samples)  # type: ignore[method-assign]
        runner._generate_with_prompt = _generate  # type: ignore[method-assign]

        runner.run_stage_b(self.output_root, {"optimized_prompt": "new prompt"})

        self.assertTrue(called["generate"])


class _SingleNodeConfigTests(unittest.TestCase):
    def test_single_node_fine_config_exposes_stage_a_experiment_controls(self) -> None:
        config = load_experiment_config(
            "D:/学习记录/导师项目/研究/caiqiyue_file/thesis_platform/configs/experiments/fine/single_node_fine.yaml"
        )

        self.assertIn("scorer", config.stage_a)
        self.assertIn("failure_equal_epsilon", config.stage_a)
        self.assertIn("failure_margin_threshold", config.stage_a)
        self.assertIn("random_fallback_seed", config.stage_a)
        self.assertEqual(config.stage_a["scorer"], "datainf_real")


if __name__ == "__main__":
    unittest.main()
