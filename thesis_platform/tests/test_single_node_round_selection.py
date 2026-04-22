from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from thesis_platform.core.single_node_runner import SingleNodeRunner
from thesis_platform.evaluation.downstream_eval import enrich_small_eval_summary, rank_eval_summary


class _ConfigStub:
    def __init__(self, output_root: Path) -> None:
        self.raw = {}
        self.meta = {"experiment_id": "single_node_round_selection_test", "seed": 17}
        self.data = {"dataset_name": "demo", "task_type": "instruction_tuning"}
        self.generator = {"initial_prompt": "Base prompt"}
        self.retriever = {}
        self.critic = {}
        self.scorer = {"name": "primary"}
        self.aggregator = {"name": "dbscan_attn_tsgdm"}
        self.stage_a = {
            "generated_count": 4,
            "select_top_k": 2,
            "max_iterations": 5,
            "convergence_threshold": 0.1,
            "max_probe_samples": 8,
            "failure_equal_epsilon": 1e-9,
            "failure_margin_threshold": 0.25,
            "random_fallback_seed": 7,
        }
        self.stage_b = {"generated_count": 4}
        self.downstream_eval = {"run_small_eval": True}
        self.runtime = {"device": "cpu"}
        self._output_root = output_root

    def output_root(self) -> Path:
        return self._output_root

    def resolve_path(self, value):
        return Path(value) if value else None

    def repo_root(self) -> Path:
        return self._output_root


class SingleNodeRoundSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.tmp_root = Path(self.tmp_dir.name)

    def _make_runner(self) -> SingleNodeRunner:
        config = _ConfigStub(self.tmp_root)
        return SingleNodeRunner(
            generator=SimpleNamespace(),
            scorer=SimpleNamespace(),
            retriever=SimpleNamespace(),
            critic=SimpleNamespace(),
            aggregator=SimpleNamespace(),
            config=config,
        )

    def test_enrich_small_eval_summary_reads_best_stats_topk_metrics(self) -> None:
        stats_dir = self.tmp_root / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        (stats_dir / "best_stats.json").write_text(
            json.dumps({"1": 0.31, "3": 0.32, "5": 0.33, "10": 0.34}),
            encoding="utf-8",
        )
        summary = {
            "stage_name": "eval_small",
            "metrics": {"best_top1": 0.31},
            "artifacts": {"stats_dir": str(stats_dir)},
        }

        enriched = enrich_small_eval_summary(summary)

        self.assertEqual(enriched["metrics"]["best_top1"], 0.31)
        self.assertEqual(enriched["metrics"]["best_top3"], 0.32)
        self.assertEqual(enriched["metrics"]["best_top5"], 0.33)
        self.assertEqual(enriched["metrics"]["best_top10"], 0.34)

    def test_rank_eval_summary_prefers_top3_when_top1_ties(self) -> None:
        first = {"metrics": {"best_top1": 0.30, "best_top3": 0.40, "best_top5": 0.50, "best_top10": 0.60}}
        second = {"metrics": {"best_top1": 0.30, "best_top3": 0.41, "best_top5": 0.49, "best_top10": 0.61}}

        self.assertGreater(rank_eval_summary(second), rank_eval_summary(first))

    def test_run_evaluates_candidate_rounds_sequentially_and_keeps_all_outputs(self) -> None:
        runner = self._make_runner()
        stage_a_result = {
            "optimized_prompt": "final prompt",
            "prompt_history": ["base", "prompt_1", "prompt_2", "prompt_3", "prompt_4", "prompt_5"],
            "round_prompts": ["prompt_1", "prompt_2", "prompt_3", "prompt_4", "prompt_5"],
            "iterations": 5,
        }
        call_order: list[str] = []

        def fake_stage_a(_output_dir):
            return dict(stage_a_result)

        def fake_stage_b(output_dir, stage_a_payload):
            call_order.append(f"stage_b:{output_dir.name}")
            self.assertEqual(stage_a_payload["optimized_prompt"], stage_a_result["round_prompts"][int(output_dir.name.split("_")[-1])])
            return [f"text-for-{output_dir.name}"]

        def fake_eval(output_dir, synthetic_texts):
            call_order.append(f"eval:{output_dir.name}")
            round_index = int(output_dir.name.split("_")[-1])
            self.assertEqual(synthetic_texts, [f"text-for-{output_dir.name}"])
            return {
                "stage_name": "eval_small",
                "metrics": {
                    "best_top1": 0.30,
                    "best_top3": 0.40 + round_index / 1000.0,
                    "best_top5": 0.50,
                    "best_top10": 0.60,
                },
            }

        runner.run_stage_a = fake_stage_a  # type: ignore[method-assign]
        runner.run_stage_b = fake_stage_b  # type: ignore[method-assign]
        runner.run_evaluation = fake_eval  # type: ignore[method-assign]

        summary = runner.run()

        self.assertEqual(
            call_order,
            [
                "stage_b:round_000",
                "eval:round_000",
                "stage_b:round_001",
                "eval:round_001",
                "stage_b:round_002",
                "eval:round_002",
                "stage_b:round_003",
                "eval:round_003",
                "stage_b:round_004",
                "eval:round_004",
            ],
        )
        self.assertEqual(len(summary["evaluation_rounds"]), 5)
        self.assertEqual(summary["best_round_index"], 4)
        self.assertEqual(summary["evaluation"]["metrics"]["best_top3"], 0.404)


if __name__ == "__main__":
    unittest.main()
