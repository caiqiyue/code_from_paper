from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pretext_platform.algorithms.bootstrap import build_bootstrap_prompts, generate_bootstrapped_samples_vllm
from pretext_platform.core.config import ExperimentConfig, load_experiment_config
from pretext_platform.core.gpu_memory import BYTES_PER_GIB, ensure_vllm_startup_memory
from pretext_platform.core.pipeline import run_pipeline
from pretext_platform.core.run_state import PretextFailure
from pretext_platform.core.types import StageSummary


class VllmBootstrapMemoryGuardTests(unittest.TestCase):
    """Validate formal Stage 2 vLLM memory limits and failure artifacts."""

    def test_bootstrap_prompts_reuse_small_seed_pool_for_smoke_runs(self) -> None:
        prompts = build_bootstrap_prompts(
            ["survivor one", "survivor two"],
            num_prompts=2,
            seed=42,
        )

        self.assertEqual(len(prompts), 2)
        self.assertTrue(all("Original Text Sample 3" in prompt for prompt in prompts))
        self.assertTrue(all("survivor one" in prompt or "survivor two" in prompt for prompt in prompts))

    def test_startup_precheck_rejects_when_free_memory_is_below_threshold(self) -> None:
        class FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def current_device():
                return 0

            @staticmethod
            def mem_get_info(_device_index):
                return int(1 * BYTES_PER_GIB), int(47.5 * BYTES_PER_GIB)

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = FakeCuda()

        with patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaises(PretextFailure) as context:
                ensure_vllm_startup_memory({"startup_required_free_gb": 2})

        self.assertEqual(
            context.exception.failure_code,
            "insufficient_free_gpu_memory_before_stage2",
        )
        self.assertEqual(context.exception.phase, "stage2_precheck")
        self.assertEqual(context.exception.details["required_free_gib"], 2.0)
        self.assertEqual(context.exception.details["observed_free_gib"], 1.0)

    def test_vllm_generation_prechecks_memory_and_passes_bounded_constructor_args(self) -> None:
        captured_llm_kwargs: dict[str, object] = {}
        captured_sampling_kwargs: dict[str, object] = {}

        class FakeLLM:
            def __init__(self, **kwargs):
                captured_llm_kwargs.update(kwargs)

            def generate(self, prompt_list, sampling_params):
                del sampling_params
                return [
                    SimpleNamespace(outputs=[SimpleNamespace(text=f"generated:{idx}")])
                    for idx, _ in enumerate(prompt_list)
                ]

        class FakeSamplingParams:
            def __init__(self, **kwargs):
                captured_sampling_kwargs.update(kwargs)

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = FakeLLM
        fake_vllm.SamplingParams = FakeSamplingParams

        bootstrap_cfg = {
            "max_model_len": 512,
            "gpu_memory_utilization": 0.35,
            "startup_required_free_gb": 2,
            "temperature": 0.4,
            "top_p": 0.9,
            "max_tokens": 33,
        }
        with patch.dict(sys.modules, {"vllm": fake_vllm}), patch(
            "pretext_platform.algorithms.bootstrap.ensure_vllm_startup_memory",
            create=True,
        ) as precheck:
            outputs = generate_bootstrapped_samples_vllm(
                ["prompt a", "prompt b"],
                Path("local-llama"),
                bootstrap_cfg,
            )

        precheck.assert_called_once()
        self.assertEqual(outputs, ["generated:0", "generated:1"])
        self.assertEqual(captured_llm_kwargs["model"], "local-llama")
        self.assertEqual(captured_llm_kwargs["max_model_len"], 512)
        self.assertEqual(captured_llm_kwargs["tensor_parallel_size"], 1)
        self.assertEqual(captured_llm_kwargs["gpu_memory_utilization"], 0.35)
        self.assertEqual(captured_sampling_kwargs["max_tokens"], 33)

    def test_vllm_generation_can_disable_cuda_graph_capture_for_shared_gpu_runs(self) -> None:
        captured_llm_kwargs: dict[str, object] = {}

        class FakeLLM:
            def __init__(self, **kwargs):
                captured_llm_kwargs.update(kwargs)

            def generate(self, prompt_list, sampling_params):
                del prompt_list, sampling_params
                return [SimpleNamespace(outputs=[SimpleNamespace(text="generated:0")])]

        class FakeSamplingParams:
            def __init__(self, **_kwargs):
                pass

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = FakeLLM
        fake_vllm.SamplingParams = FakeSamplingParams

        with patch.dict(sys.modules, {"vllm": fake_vllm}), patch(
            "pretext_platform.algorithms.bootstrap.ensure_vllm_startup_memory",
            return_value={"observed_free_gib": 30.0, "required_free_gib": 25.0},
        ):
            generate_bootstrapped_samples_vllm(
                ["prompt a"],
                Path("local-llama"),
                {
                    "max_model_len": 512,
                    "gpu_memory_utilization": 0.35,
                    "enforce_eager": True,
                },
            )

        self.assertTrue(captured_llm_kwargs["enforce_eager"])

    def test_vllm_runtime_oom_after_precheck_is_classified(self) -> None:
        class FakeLLM:
            def __init__(self, **_kwargs):
                pass

            def generate(self, _prompt_list, _sampling_params):
                raise RuntimeError("CUDA out of memory while allocating KV cache")

        class FakeSamplingParams:
            def __init__(self, **_kwargs):
                pass

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = FakeLLM
        fake_vllm.SamplingParams = FakeSamplingParams

        with patch.dict(sys.modules, {"vllm": fake_vllm}), patch(
            "pretext_platform.algorithms.bootstrap.ensure_vllm_startup_memory",
            return_value={"observed_free_gib": 29.0, "required_free_gib": 2.0},
        ):
            with self.assertRaises(PretextFailure) as context:
                generate_bootstrapped_samples_vllm(
                    ["prompt a"],
                    Path("local-llama"),
                    {"max_model_len": 512, "gpu_memory_utilization": 0.35},
                )

        self.assertEqual(context.exception.failure_code, "stage2_runtime_gpu_oom")
        self.assertEqual(context.exception.phase, "stage2")

    def test_formal_vllm_configs_define_shared_a6000_memory_budget(self) -> None:
        root = Path(__file__).resolve().parents[1]
        expected_budgets = {
            root / "configs" / "experiments" / "single_node_formal" / "sp_c1_jobs_base.yaml": 2,
            root / "configs" / "experiments" / "federated_formal" / "fp_c1_jobs_base.yaml": 2,
        }

        for config_path, expected_budget in expected_budgets.items():
            with self.subTest(config_path=config_path.name):
                config = load_experiment_config(config_path)
                self.assertEqual(config.bootstrap.get("generator_backend"), "vllm")
                self.assertEqual(config.bootstrap.get("max_model_len"), 512)
                self.assertEqual(config.bootstrap.get("startup_required_free_gb"), expected_budget)
                self.assertAlmostEqual(float(config.bootstrap.get("gpu_memory_utilization")), 0.35)
                self.assertTrue(config.bootstrap.get("enforce_eager"))

    def test_single_node_a6000_smoke_config_matches_sp_c1_with_tiny_scale(self) -> None:
        root = Path(__file__).resolve().parents[1]
        formal = load_experiment_config(
            root / "configs" / "experiments" / "single_node_formal" / "sp_c1_jobs_base.yaml"
        )
        smoke = load_experiment_config(
            root / "configs" / "experiments" / "single_node_formal" / "sp_test_vllm_a6000.yaml"
        )

        self.assertEqual(smoke.data["dataset_name"], formal.data["dataset_name"])
        self.assertEqual(smoke.data["train_path"], formal.data["train_path"])
        self.assertEqual(smoke.data["eval_path"], formal.data["eval_path"])
        self.assertEqual(smoke.bootstrap["generator_backend"], formal.bootstrap["generator_backend"])
        self.assertEqual(smoke.bootstrap["generator_model"], formal.bootstrap["generator_model"])

        self.assertEqual(smoke.data["max_samples_per_client"], 4)
        self.assertEqual(smoke.data["train_limit"], 8)
        self.assertEqual(smoke.data["eval_limit"], 4)
        self.assertEqual(smoke.data["initialization_limit"], 32)
        self.assertEqual(smoke.stage1["rounds"], 1)
        self.assertEqual(smoke.stage1["batch_size"], 4)
        self.assertEqual(smoke.stage1["embed_batch_size"], 8)
        self.assertEqual(smoke.bootstrap["num_prompts"], 4)
        self.assertEqual(smoke.bootstrap["max_tokens"], 32)

        for key in (
            "max_model_len",
            "gpu_memory_utilization",
            "tensor_parallel_size",
            "enforce_eager",
        ):
            self.assertEqual(smoke.bootstrap[key], formal.bootstrap[key])
        self.assertEqual(formal.bootstrap["startup_required_free_gb"], 2)
        self.assertEqual(smoke.bootstrap["startup_required_free_gb"], 2)

    def test_pipeline_writes_failure_artifacts_for_stage2_startup_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = ExperimentConfig.from_mapping(
                {
                    "meta": {"experiment_id": "oom_guard_demo"},
                    "paths": {"repo_root": ".", "output_root": "./out"},
                    "stage1": {"enabled": True},
                    "bootstrap": {"enabled": True},
                    "eval_small": {"enabled": False},
                    "eval_large": {"enabled": False},
                },
                base_dir=root,
            )
            stage1_summary = StageSummary(
                "stage1",
                root / "out" / "oom_guard_demo" / "stage1",
                {"generated_files": []},
                {"rounds": 1},
            )
            failure = PretextFailure(
                "insufficient_free_gpu_memory_before_stage2",
                "free GPU memory 1.0 GiB is below required 2.0 GiB",
                phase="stage2_precheck",
                details={"observed_free_gib": 1.0, "required_free_gib": 2.0},
            )
            with patch("pretext_platform.core.pipeline.run_stage1", return_value=stage1_summary), patch(
                "pretext_platform.core.pipeline.run_bootstrap",
                side_effect=failure,
            ):
                with self.assertRaises(PretextFailure):
                    run_pipeline(config)

            experiment_dir = root / "out" / "oom_guard_demo"
            run_state = json.loads((experiment_dir / "run_state.json").read_text(encoding="utf-8"))
            failure_summary = json.loads((experiment_dir / "failure_summary.json").read_text(encoding="utf-8"))
            metrics_summary = json.loads((experiment_dir / "metrics_summary.json").read_text(encoding="utf-8"))

            self.assertEqual(run_state["status"], "failed")
            self.assertEqual(run_state["phase"], "stage2_precheck")
            self.assertEqual(run_state["last_error"]["failure_code"], "insufficient_free_gpu_memory_before_stage2")
            self.assertEqual(failure_summary["failure_code"], "insufficient_free_gpu_memory_before_stage2")
            self.assertEqual(metrics_summary["status"], "failed")
            self.assertEqual(metrics_summary["failure_code"], "insufficient_free_gpu_memory_before_stage2")


if __name__ == "__main__":
    unittest.main()
