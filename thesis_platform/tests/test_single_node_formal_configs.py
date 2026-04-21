from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from thesis_platform.core.config import load_experiment_config
from thesis_platform.core.pipeline import run_pipeline


class SingleNodeFormalConfigTests(unittest.TestCase):
    def test_run_pipeline_dispatches_single_node_configs_through_single_node_runner(self) -> None:
        config = SimpleNamespace(
            execution={"mode": "single_node"},
            meta={"experiment_id": "single_node_dispatch"},
            repo_root=lambda: Path("."),
            generator={},
            scorer={"name": "datainf_real"},
            retriever={},
            critic={},
            aggregator={"name": "dbscan_attn_tsgdm"},
        )
        fake_runner = SimpleNamespace(run=lambda: {"status": "single_node"})

        with patch("thesis_platform.core.pipeline.load_experiment_config", return_value=config), patch(
            "thesis_platform.core.pipeline._build_single_node_runner",
            return_value=fake_runner,
        ) as build_runner, patch("thesis_platform.core.pipeline.ExperimentRunner") as federated_runner:
            result = run_pipeline("dummy.yaml")

        self.assertEqual(result, {"status": "single_node"})
        build_runner.assert_called_once_with(config)
        federated_runner.assert_not_called()

    def test_single_node_formal_jobs_config_uses_real_scorer_and_unified_entry_mode(self) -> None:
        config = load_experiment_config(
            "D:/学习记录/导师项目/研究/caiqiyue_file/thesis_platform/configs/experiments/single_node_formal/sn_c1_jobs_base.yaml"
        )

        self.assertEqual(config.execution["mode"], "single_node")
        self.assertEqual(config.scorer["name"], "datainf_real")
        self.assertEqual(config.stage_a["scorer"], "datainf_real")
        self.assertEqual(config.aggregator["name"], "dbscan_attn_tsgdm")
        self.assertEqual(config.stage_b["generated_count"], 1500)

    def test_single_node_formal_server_generation_uses_vllm_memory_budget(self) -> None:
        config_root = Path(__file__).resolve().parents[1]
        config = load_experiment_config(
            config_root / "configs" / "experiments" / "single_node_formal" / "sn_c1_jobs_base.yaml"
        )

        self.assertEqual(config.llm["client"]["engine"], "transformers")
        self.assertEqual(config.llm["server"]["engine"], "vllm")
        self.assertEqual(config.llm["server"]["model_name_or_path"], "thesis_platform/open_model/llama_2_7b_hf")
        self.assertEqual(config.llm["server"]["max_model_len"], 512)
        self.assertAlmostEqual(float(config.llm["server"]["gpu_memory_utilization"]), 0.55)
        self.assertEqual(config.llm["server"]["startup_required_free_gb"], 28)
        self.assertEqual(config.llm["server"]["tensor_parallel_size"], 1)

    def test_single_node_vllm_a6000_smoke_config_matches_sn_c1_with_tiny_scale(self) -> None:
        config_root = Path(__file__).resolve().parents[1]
        formal = load_experiment_config(
            config_root / "configs" / "experiments" / "single_node_formal" / "sn_c1_jobs_base.yaml"
        )
        smoke = load_experiment_config(
            config_root / "configs" / "experiments" / "single_node_formal" / "sn_test_vllm_a6000.yaml"
        )

        self.assertEqual(smoke.execution["mode"], formal.execution["mode"])
        self.assertEqual(smoke.data["dataset_name"], formal.data["dataset_name"])
        self.assertEqual(smoke.data["train_path"], formal.data["train_path"])
        self.assertEqual(smoke.scorer["name"], formal.scorer["name"])
        self.assertEqual(smoke.critic["name"], formal.critic["name"])
        self.assertEqual(smoke.aggregator["name"], formal.aggregator["name"])

        self.assertEqual(smoke.generator["generated_per_round"], 2)
        self.assertEqual(smoke.stage_a["generated_count"], 4)
        self.assertEqual(smoke.stage_a["select_top_k"], 2)
        self.assertEqual(smoke.stage_a["max_iterations"], 1)
        self.assertEqual(smoke.stage_b["generated_count"], 4)
        self.assertEqual(smoke.data["train_limit"], 8)
        self.assertEqual(smoke.data["eval_limit"], 4)

        self.assertEqual(smoke.downstream_eval["run_small_eval"], formal.downstream_eval["run_small_eval"])
        self.assertEqual(smoke.downstream_eval["small_epochs"], 1)
        self.assertEqual(smoke.downstream_eval["small_batch_size"], 1)
        self.assertEqual(smoke.downstream_eval["small_eval_batch_size"], 1)

        for key in (
            "engine",
            "model_name_or_path",
            "max_model_len",
            "gpu_memory_utilization",
            "tensor_parallel_size",
        ):
            self.assertEqual(smoke.llm["server"][key], formal.llm["server"][key])
        self.assertEqual(formal.llm["server"]["startup_required_free_gb"], 28)
        self.assertEqual(smoke.llm["server"]["startup_required_free_gb"], 25)

    def test_single_node_formal_gradmm_and_ira_configs_share_federated_method_files(self) -> None:
        gradmm_config = load_experiment_config(
            "D:/学习记录/导师项目/研究/caiqiyue_file/thesis_platform/configs/experiments/single_node_formal/sn_a1_jobs_gradmm.yaml"
        )
        ira_config = load_experiment_config(
            "D:/学习记录/导师项目/研究/caiqiyue_file/thesis_platform/configs/experiments/single_node_formal/sn_a2_jobs_ira.yaml"
        )

        self.assertEqual(gradmm_config.scorer["name"], "gradmm_real")
        self.assertEqual(gradmm_config.stage_a["scorer"], "gradmm_real")
        self.assertEqual(ira_config.scorer["name"], "ira")
        self.assertEqual(ira_config.stage_a["scorer"], "ira")
        self.assertEqual(ira_config.data["sample_format"], "raw_text")


if __name__ == "__main__":
    unittest.main()
