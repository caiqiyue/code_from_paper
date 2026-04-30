import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from paper_new_selector.stage1_runner import run_stage1, run_stage1_with_runtime


class _FakeSample:
    def __init__(self, text: str) -> None:
        self._text = text

    def render_text(self) -> str:
        return self._text


class _FakeGenerator:
    def generate(self, _round_ctx):
        return [_FakeSample("candidate alpha text"), _FakeSample("candidate beta text")]


class _FakeTextBackend:
    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        self.released = True

    def ensure_session(self):
        return ("fake-llm", "fake-sampling")

    def generate_batch(self, prompts, *, max_new_tokens, temperature=None):
        del max_new_tokens, temperature
        return [f"generated::{prompt}" for prompt in prompts]


class _FakeEmbedder:
    def __init__(self) -> None:
        self.released = False

    def embed_texts(self, texts):
        return [[float(index + 1), 0.0] for index, _ in enumerate(texts)]

    def release(self) -> None:
        self.released = True


def _round_context_patch():
    thesis_platform = ModuleType("thesis_platform")
    core = ModuleType("thesis_platform.core")
    context = ModuleType("thesis_platform.core.context")

    class RoundContext:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    context.RoundContext = RoundContext
    core.context = context
    thesis_platform.core = core
    return patch.dict(
        "sys.modules",
        {
            "thesis_platform": thesis_platform,
            "thesis_platform.core": core,
            "thesis_platform.core.context": context,
        },
    )


class Stage1RunnerReleaseTests(unittest.TestCase):
    def test_run_stage1_releases_generator_backend_and_embedder_after_completion(self):
        fake_backend = _FakeTextBackend()
        fake_embedder = _FakeEmbedder()
        config = {
            "pipeline": {"stage1_mode": "selector_seed_search"},
            "generator": {
                "initial_prompt": "prompt",
                "candidate_count": 2,
                "max_rounds": 1,
                "exemplars_per_prompt": 1,
            },
            "meta": {"seed": 42},
            "selector": {
                "private_knn_k": 1,
                "density_lambda": 0.0,
                "novelty_lambda": 0.0,
                "length_lambda": 0.0,
                "length_floor": 1,
                "length_ceiling": 100,
                "rank_weights": [1.0],
                "top_q": 1,
                "reference_top_k": 1,
                "lambda_generic": 0.2,
                "lambda_redundancy": 0.3,
                "seed_top_k": 1,
                "hard_negative_top_k": 1,
            },
            "privacy": {"enabled": False, "delta": 1e-5},
            "stage1": {"sigma": 0.0, "delta": 1e-5},
        }
        sample_bundle = {
            "train_samples": [_FakeSample("private alpha"), _FakeSample("private beta")],
            "eval_samples": [_FakeSample("eval alpha")],
            "init_samples": [_FakeSample("seed alpha"), _FakeSample("seed beta")],
        }
        decision = SimpleNamespace(
            selected_indices=[0],
            hard_negative_indices=[1],
            hard_negative_reason={1: "boundary_negative"},
            accept_scores=[0.9, 0.2],
            to_dict=lambda: {"selected_indices": [0], "hard_negative_indices": [1]},
        )

        with _round_context_patch(), patch("paper_new_selector.stage1_runner.load_yaml_config", return_value=config), patch(
            "paper_new_selector.stage1_runner.load_text_samples",
            return_value=sample_bundle,
        ), patch(
            "paper_new_selector.stage1_runner.build_candidate_generator",
            return_value=SimpleNamespace(
                generator=_FakeGenerator(),
                text_backend=fake_backend,
                contract={"llm_backend": "vllm"},
            ),
        ), patch(
            "paper_new_selector.stage1_runner.build_embedder_from_config",
            return_value=fake_embedder,
        ), patch(
            "paper_new_selector.stage1_runner.build_private_importance_weights",
            return_value=[1.0, 1.0],
        ), patch(
            "paper_new_selector.stage1_runner.compute_private_support",
            return_value=[0.9, 0.2],
        ), patch(
            "paper_new_selector.stage1_runner.apply_gaussian_privacy_noise",
            side_effect=lambda scores, **_: scores,
        ), patch(
            "paper_new_selector.stage1_runner.compute_genericity_penalties",
            return_value=[0.1, 0.3],
        ), patch(
            "paper_new_selector.stage1_runner.greedy_select_candidates",
            return_value=decision,
        ), patch(
            "paper_new_selector.stage1_runner.build_boundary_state",
            return_value={"negative_pattern_stats": {"count": 1}},
        ):
            summary = run_stage1("dummy.yaml", validate_only=False)

        self.assertEqual(summary["selected_texts"], ["candidate alpha text"])
        self.assertTrue(fake_backend.released)
        self.assertTrue(fake_embedder.released)

    def test_run_stage1_with_runtime_keeps_generator_backend_and_embedder_loaded(self):
        fake_backend = _FakeTextBackend()
        fake_embedder = _FakeEmbedder()
        config = {
            "pipeline": {"stage1_mode": "selector_seed_search"},
            "generator": {
                "initial_prompt": "prompt",
                "candidate_count": 2,
                "max_rounds": 1,
                "exemplars_per_prompt": 1,
            },
            "meta": {"seed": 42},
            "selector": {
                "private_knn_k": 1,
                "density_lambda": 0.0,
                "novelty_lambda": 0.0,
                "length_lambda": 0.0,
                "length_floor": 1,
                "length_ceiling": 100,
                "rank_weights": [1.0],
                "top_q": 1,
                "reference_top_k": 1,
                "lambda_generic": 0.2,
                "lambda_redundancy": 0.3,
                "seed_top_k": 1,
                "hard_negative_top_k": 1,
            },
            "privacy": {"enabled": False, "delta": 1e-5},
            "stage1": {"sigma": 0.0, "delta": 1e-5},
        }
        sample_bundle = {
            "train_samples": [_FakeSample("private alpha"), _FakeSample("private beta")],
            "eval_samples": [_FakeSample("eval alpha")],
            "init_samples": [_FakeSample("seed alpha"), _FakeSample("seed beta")],
        }
        decision = SimpleNamespace(
            selected_indices=[0],
            hard_negative_indices=[1],
            hard_negative_reason={1: "boundary_negative"},
            accept_scores=[0.9, 0.2],
            to_dict=lambda: {"selected_indices": [0], "hard_negative_indices": [1]},
        )

        with _round_context_patch(), patch("paper_new_selector.stage1_runner.load_yaml_config", return_value=config), patch(
            "paper_new_selector.stage1_runner.load_text_samples",
            return_value=sample_bundle,
        ), patch(
            "paper_new_selector.stage1_runner.build_candidate_generator",
            return_value=SimpleNamespace(
                generator=_FakeGenerator(),
                text_backend=fake_backend,
                contract={"llm_backend": "vllm"},
                shared_session=SimpleNamespace(to_dict=lambda: {"llm_backend": "vllm"}, backend=fake_backend),
            ),
        ), patch(
            "paper_new_selector.stage1_runner.build_embedder_from_config",
            return_value=fake_embedder,
        ), patch(
            "paper_new_selector.stage1_runner.build_private_importance_weights",
            return_value=[1.0, 1.0],
        ), patch(
            "paper_new_selector.stage1_runner.compute_private_support",
            return_value=[0.9, 0.2],
        ), patch(
            "paper_new_selector.stage1_runner.apply_gaussian_privacy_noise",
            side_effect=lambda scores, **_: scores,
        ), patch(
            "paper_new_selector.stage1_runner.compute_genericity_penalties",
            return_value=[0.1, 0.3],
        ), patch(
            "paper_new_selector.stage1_runner.greedy_select_candidates",
            return_value=decision,
        ), patch(
            "paper_new_selector.stage1_runner.build_boundary_state",
            return_value={"negative_pattern_stats": {"count": 1}},
        ), patch(
            "paper_new_selector.stage1_runner.release_runtime_memory",
        ) as release_runtime:
            summary, runtime = run_stage1_with_runtime("dummy.yaml", validate_only=False)

        self.assertEqual(summary["selected_texts"], ["candidate alpha text"])
        self.assertEqual(summary["shared_session"]["llm_backend"], "vllm")
        self.assertIs(runtime["embedder"], fake_embedder)
        self.assertIs(runtime["shared_session"].backend, fake_backend)
        self.assertFalse(fake_backend.released)
        self.assertFalse(fake_embedder.released)
        release_runtime.assert_not_called()

    def test_stage1_runner_passes_reference_rank_weights_to_genericity(self):
        fake_backend = _FakeTextBackend()
        fake_embedder = _FakeEmbedder()
        config = {
            "pipeline": {"stage1_mode": "selector_seed_search"},
            "generator": {
                "initial_prompt": "prompt",
                "candidate_count": 2,
                "max_rounds": 1,
                "exemplars_per_prompt": 1,
            },
            "meta": {"seed": 42},
            "selector": {
                "private_knn_k": 1,
                "density_lambda": 0.0,
                "novelty_lambda": 0.0,
                "length_lambda": 0.0,
                "length_floor": 1,
                "length_ceiling": 100,
                "rank_weights": [1.0],
                "top_q": 1,
                "reference_top_k": 6,
                "reference_rank_weights": [1.0, 0.8, 0.6, 0.4, 0.25, 0.1],
                "lambda_generic": 0.2,
                "lambda_redundancy": 0.3,
                "seed_top_k": 1,
                "hard_negative_top_k": 1,
            },
            "privacy": {"enabled": False, "delta": 1e-5},
            "stage1": {"sigma": 0.0, "delta": 1e-5},
        }
        sample_bundle = {
            "train_samples": [_FakeSample("private alpha"), _FakeSample("private beta")],
            "eval_samples": [_FakeSample("eval alpha")],
            "init_samples": [_FakeSample("seed alpha"), _FakeSample("seed beta")],
        }
        decision = SimpleNamespace(
            selected_indices=[0],
            hard_negative_indices=[1],
            hard_negative_reason={1: "boundary_negative"},
            accept_scores=[0.9, 0.2],
            to_dict=lambda: {"selected_indices": [0], "hard_negative_indices": [1]},
        )

        with _round_context_patch(), patch("paper_new_selector.stage1_runner.load_yaml_config", return_value=config), patch(
            "paper_new_selector.stage1_runner.load_text_samples",
            return_value=sample_bundle,
        ), patch(
            "paper_new_selector.stage1_runner.build_candidate_generator",
            return_value=SimpleNamespace(
                generator=_FakeGenerator(),
                text_backend=fake_backend,
                contract={"llm_backend": "vllm"},
            ),
        ), patch(
            "paper_new_selector.stage1_runner.build_embedder_from_config",
            return_value=fake_embedder,
        ), patch(
            "paper_new_selector.stage1_runner.build_private_importance_weights",
            return_value=[1.0, 1.0],
        ), patch(
            "paper_new_selector.stage1_runner.compute_private_support",
            return_value=[0.9, 0.2],
        ), patch(
            "paper_new_selector.stage1_runner.apply_gaussian_privacy_noise",
            side_effect=lambda scores, **_: scores,
        ), patch(
            "paper_new_selector.stage1_runner.compute_genericity_penalties",
            return_value=[0.1, 0.3],
        ) as genericity_mock, patch(
            "paper_new_selector.stage1_runner.greedy_select_candidates",
            return_value=decision,
        ), patch(
            "paper_new_selector.stage1_runner.build_boundary_state",
            return_value={"negative_pattern_stats": {"count": 1}},
        ):
            run_stage1("dummy.yaml", validate_only=False)

        self.assertEqual(
            genericity_mock.call_args.kwargs["reference_rank_weights"],
            [1.0, 0.8, 0.6, 0.4, 0.25, 0.1],
        )

    def test_stage1_runner_passes_genericity_gate_config_to_genericity(self):
        fake_backend = _FakeTextBackend()
        fake_embedder = _FakeEmbedder()
        config = {
            "pipeline": {"stage1_mode": "selector_seed_search"},
            "generator": {
                "initial_prompt": "prompt",
                "candidate_count": 2,
                "max_rounds": 1,
                "exemplars_per_prompt": 1,
            },
            "meta": {"seed": 42},
            "selector": {
                "private_knn_k": 1,
                "density_lambda": 0.0,
                "novelty_lambda": 0.0,
                "length_lambda": 0.0,
                "length_floor": 1,
                "length_ceiling": 100,
                "rank_weights": [1.0],
                "top_q": 1,
                "reference_top_k": 6,
                "reference_rank_weights": [1.0, 0.8, 0.6, 0.4, 0.25, 0.1],
                "genericity_gate_low": 0.78,
                "genericity_gate_high": 0.90,
                "genericity_gate_low_scale": 0.10,
                "genericity_gate_mid_scale": 0.45,
                "lambda_generic": 0.2,
                "lambda_redundancy": 0.3,
                "seed_top_k": 1,
                "hard_negative_top_k": 1,
            },
            "privacy": {"enabled": False, "delta": 1e-5},
            "stage1": {"sigma": 0.0, "delta": 1e-5},
        }
        sample_bundle = {
            "train_samples": [_FakeSample("private alpha"), _FakeSample("private beta")],
            "eval_samples": [_FakeSample("eval alpha")],
            "init_samples": [_FakeSample("seed alpha"), _FakeSample("seed beta")],
        }
        decision = SimpleNamespace(
            selected_indices=[0],
            hard_negative_indices=[1],
            hard_negative_reason={1: "boundary_negative"},
            accept_scores=[0.9, 0.2],
            to_dict=lambda: {"selected_indices": [0], "hard_negative_indices": [1]},
        )

        with _round_context_patch(), patch("paper_new_selector.stage1_runner.load_yaml_config", return_value=config), patch(
            "paper_new_selector.stage1_runner.load_text_samples",
            return_value=sample_bundle,
        ), patch(
            "paper_new_selector.stage1_runner.build_candidate_generator",
            return_value=SimpleNamespace(
                generator=_FakeGenerator(),
                text_backend=fake_backend,
                contract={"llm_backend": "vllm"},
            ),
        ), patch(
            "paper_new_selector.stage1_runner.build_embedder_from_config",
            return_value=fake_embedder,
        ), patch(
            "paper_new_selector.stage1_runner.build_private_importance_weights",
            return_value=[1.0, 1.0],
        ), patch(
            "paper_new_selector.stage1_runner.compute_private_support",
            return_value=[0.9, 0.2],
        ), patch(
            "paper_new_selector.stage1_runner.apply_gaussian_privacy_noise",
            side_effect=lambda scores, **_: scores,
        ), patch(
            "paper_new_selector.stage1_runner.compute_genericity_penalties",
            return_value=[0.1, 0.3],
        ) as genericity_mock, patch(
            "paper_new_selector.stage1_runner.greedy_select_candidates",
            return_value=decision,
        ), patch(
            "paper_new_selector.stage1_runner.build_boundary_state",
            return_value={"negative_pattern_stats": {"count": 1}},
        ):
            run_stage1("dummy.yaml", validate_only=False)

        self.assertTrue(genericity_mock.call_args.kwargs["apply_gate"])
        self.assertEqual(genericity_mock.call_args.kwargs["gate_low"], 0.78)
        self.assertEqual(genericity_mock.call_args.kwargs["gate_high"], 0.90)
        self.assertEqual(genericity_mock.call_args.kwargs["low_scale"], 0.10)
        self.assertEqual(genericity_mock.call_args.kwargs["mid_scale"], 0.45)

    def test_stage1_runner_applies_forums_overrides_before_scoring(self):
        fake_backend = _FakeTextBackend()
        fake_embedder = _FakeEmbedder()
        config = {
            "pipeline": {"stage1_mode": "selector_seed_search"},
            "generator": {
                "initial_prompt": "prompt",
                "candidate_count": 2,
                "max_rounds": 1,
                "exemplars_per_prompt": 1,
            },
            "meta": {"seed": 42},
            "data": {"dataset_name": "forums"},
            "selector": {
                "private_knn_k": 1,
                "density_lambda": 0.0,
                "novelty_lambda": 0.0,
                "length_lambda": 0.0,
                "length_floor": 1,
                "length_ceiling": 100,
                "rank_weights": [1.0],
                "top_q": 1,
                "reference_top_k": 1,
                "genericity_gate_low": 0.78,
                "genericity_gate_high": 0.90,
                "genericity_gate_low_scale": 0.10,
                "genericity_gate_mid_scale": 0.45,
                "lambda_generic": 0.35,
                "lambda_redundancy": 0.25,
                "seed_top_k": 1,
                "hard_negative_top_k": 1,
                "_forums_lambda_generic": 0.15,
                "_forums_lambda_redundancy": 0.10,
                "_forums_seed_top_k": 2,
                "_forums_gate_low": 0.85,
                "_forums_gate_high": 0.95,
                "_forums_low_scale": 0.20,
                "_forums_mid_scale": 0.60,
            },
            "privacy": {"enabled": False, "delta": 1e-5},
            "stage1": {"sigma": 0.0, "delta": 1e-5},
        }
        sample_bundle = {
            "train_samples": [_FakeSample("private alpha"), _FakeSample("private beta")],
            "eval_samples": [_FakeSample("eval alpha")],
            "init_samples": [_FakeSample("seed alpha"), _FakeSample("seed beta")],
        }
        decision = SimpleNamespace(
            selected_indices=[0, 1],
            hard_negative_indices=[],
            hard_negative_reason={},
            accept_scores=[0.9, 0.2],
            to_dict=lambda: {"selected_indices": [0, 1], "hard_negative_indices": []},
        )

        with _round_context_patch(), patch("paper_new_selector.stage1_runner.load_yaml_config", return_value=config), patch(
            "paper_new_selector.stage1_runner.load_text_samples",
            return_value=sample_bundle,
        ), patch(
            "paper_new_selector.stage1_runner.build_candidate_generator",
            return_value=SimpleNamespace(
                generator=_FakeGenerator(),
                text_backend=fake_backend,
                contract={"llm_backend": "vllm"},
            ),
        ), patch(
            "paper_new_selector.stage1_runner.build_embedder_from_config",
            return_value=fake_embedder,
        ), patch(
            "paper_new_selector.stage1_runner.build_private_importance_weights",
            return_value=[1.0, 1.0],
        ), patch(
            "paper_new_selector.stage1_runner.compute_private_support",
            return_value=[0.9, 0.2],
        ), patch(
            "paper_new_selector.stage1_runner.apply_gaussian_privacy_noise",
            side_effect=lambda scores, **_: scores,
        ), patch(
            "paper_new_selector.stage1_runner.compute_genericity_penalties",
            return_value=[0.1, 0.3],
        ) as genericity_mock, patch(
            "paper_new_selector.stage1_runner.greedy_select_candidates",
            return_value=decision,
        ) as greedy_mock, patch(
            "paper_new_selector.stage1_runner.build_boundary_state",
            return_value={"negative_pattern_stats": {"count": 0}},
        ):
            run_stage1("dummy.yaml", validate_only=False)

        genericity_kwargs = genericity_mock.call_args.kwargs
        self.assertEqual(genericity_kwargs["gate_low"], 0.85)
        self.assertEqual(genericity_kwargs["gate_high"], 0.95)
        self.assertEqual(genericity_kwargs["low_scale"], 0.20)
        self.assertEqual(genericity_kwargs["mid_scale"], 0.60)

        greedy_kwargs = greedy_mock.call_args.kwargs
        self.assertEqual(greedy_kwargs["lambda_generic"], 0.15)
        self.assertEqual(greedy_kwargs["lambda_redundancy"], 0.10)
        self.assertEqual(greedy_kwargs["seed_top_k"], 2)

    def test_stage1_runner_uses_resolved_adaptive_seed_budget(self):
        fake_backend = _FakeTextBackend()
        fake_embedder = _FakeEmbedder()
        config = {
            "pipeline": {"stage1_mode": "selector_seed_search"},
            "generator": {
                "initial_prompt": "prompt",
                "candidate_count": 2,
                "max_rounds": 1,
                "exemplars_per_prompt": 1,
            },
            "meta": {"seed": 42},
            "data": {"dataset_name": "congressional"},
            "selector": {
                "private_knn_k": 1,
                "density_lambda": 0.0,
                "novelty_lambda": 0.0,
                "length_lambda": 0.0,
                "length_floor": 1,
                "length_ceiling": 300,
                "rank_weights": [1.0],
                "top_q": 1,
                "reference_top_k": 1,
                "lambda_generic": 0.2,
                "lambda_redundancy": 0.3,
                "seed_top_k": 20,
                "seed_budget_rule": {"enabled": True, "mode": "length_family"},
                "hard_negative_top_k": 1,
            },
            "privacy": {"enabled": False, "delta": 1e-5},
            "stage1": {"sigma": 0.0, "delta": 1e-5},
        }
        sample_bundle = {
            "train_samples": [
                _FakeSample(" ".join(["short"] * 90)),
                _FakeSample(" ".join(["short"] * 110)),
                _FakeSample(" ".join(["short"] * 130)),
            ],
            "eval_samples": [_FakeSample("eval alpha")],
            "init_samples": [_FakeSample("seed alpha"), _FakeSample("seed beta")],
        }
        decision = SimpleNamespace(
            selected_indices=[0],
            hard_negative_indices=[1],
            hard_negative_reason={1: "boundary_negative"},
            accept_scores=[0.9, 0.2],
            to_dict=lambda: {"selected_indices": [0], "hard_negative_indices": [1]},
        )

        with _round_context_patch(), patch("paper_new_selector.stage1_runner.load_yaml_config", return_value=config), patch(
            "paper_new_selector.stage1_runner.load_text_samples",
            return_value=sample_bundle,
        ), patch(
            "paper_new_selector.stage1_runner.build_candidate_generator",
            return_value=SimpleNamespace(
                generator=_FakeGenerator(),
                text_backend=fake_backend,
                contract={"llm_backend": "vllm"},
            ),
        ), patch(
            "paper_new_selector.stage1_runner.build_embedder_from_config",
            return_value=fake_embedder,
        ), patch(
            "paper_new_selector.stage1_runner.build_private_importance_weights",
            return_value=[1.0, 1.0, 1.0],
        ), patch(
            "paper_new_selector.stage1_runner.compute_private_support",
            return_value=[0.9, 0.2],
        ), patch(
            "paper_new_selector.stage1_runner.apply_gaussian_privacy_noise",
            side_effect=lambda scores, **_: scores,
        ), patch(
            "paper_new_selector.stage1_runner.compute_genericity_penalties",
            return_value=[0.1, 0.3],
        ), patch(
            "paper_new_selector.stage1_runner.greedy_select_candidates",
            return_value=decision,
        ) as greedy_mock, patch(
            "paper_new_selector.stage1_runner.build_boundary_state",
            return_value={"negative_pattern_stats": {"count": 1}},
        ):
            summary = run_stage1("dummy.yaml", validate_only=False)

        greedy_kwargs = greedy_mock.call_args.kwargs
        self.assertEqual(greedy_kwargs["seed_top_k"], 19)
        self.assertEqual(summary["seed_budget"]["configured_seed_top_k"], 20)
        self.assertEqual(summary["seed_budget"]["resolved_seed_top_k"], 19)
        self.assertEqual(summary["seed_budget"]["private_length_median"], 110.0)

    def test_stage1_runner_uses_self_calibrated_seed_budget_path(self):
        fake_backend = _FakeTextBackend()
        fake_embedder = _FakeEmbedder()
        config = {
            "pipeline": {"stage1_mode": "selector_seed_search"},
            "generator": {
                "initial_prompt": "prompt",
                "candidate_count": 2,
                "max_rounds": 1,
                "exemplars_per_prompt": 1,
            },
            "meta": {"seed": 42},
            "selector": {
                "private_knn_k": 1,
                "density_lambda": 0.0,
                "novelty_lambda": 0.0,
                "length_lambda": 0.0,
                "length_floor": 1,
                "length_ceiling": 100,
                "rank_weights": [1.0],
                "top_q": 1,
                "reference_top_k": 1,
                "lambda_generic": 0.2,
                "lambda_redundancy": 0.3,
                "seed_top_k": 20,
                "hard_negative_top_k": 1,
                "seed_budget_rule": {
                    "enabled": True,
                    "mode": "self_calibrated",
                    "candidate_seed_top_k": [18, 19, 20],
                },
            },
            "privacy": {"enabled": False, "delta": 1e-5},
            "stage1": {"sigma": 0.0, "delta": 1e-5},
        }
        sample_bundle = {
            "train_samples": [_FakeSample("private alpha"), _FakeSample("private beta")],
            "eval_samples": [_FakeSample("eval alpha")],
            "init_samples": [_FakeSample("seed alpha"), _FakeSample("seed beta")],
        }
        decision = SimpleNamespace(
            selected_indices=[0],
            hard_negative_indices=[1],
            hard_negative_reason={1: "boundary_negative"},
            accept_scores=[0.9, 0.2],
            to_dict=lambda: {"selected_indices": [0], "hard_negative_indices": [1]},
        )

        with _round_context_patch(), patch(
            "paper_new_selector.stage1_runner.load_yaml_config",
            return_value=config,
        ), patch(
            "paper_new_selector.stage1_runner.load_text_samples",
            return_value=sample_bundle,
        ), patch(
            "paper_new_selector.stage1_runner.build_candidate_generator",
            return_value=SimpleNamespace(
                generator=_FakeGenerator(),
                text_backend=fake_backend,
                contract={"llm_backend": "vllm"},
            ),
        ), patch(
            "paper_new_selector.stage1_runner.build_embedder_from_config",
            return_value=fake_embedder,
        ), patch(
            "paper_new_selector.stage1_runner.build_private_importance_weights",
            return_value=[1.0, 1.0],
        ), patch(
            "paper_new_selector.stage1_runner.compute_private_support",
            return_value=[0.9, 0.2],
        ), patch(
            "paper_new_selector.stage1_runner.apply_gaussian_privacy_noise",
            side_effect=lambda scores, **_: scores,
        ), patch(
            "paper_new_selector.stage1_runner.compute_genericity_penalties",
            return_value=[0.1, 0.3],
        ), patch(
            "paper_new_selector.stage1_runner.resolve_seed_top_k_by_self_calibration",
            return_value={
                "decision": decision,
                "seed_budget_summary": {
                    "configured_seed_top_k": 20,
                    "resolved_seed_top_k": 19,
                    "mode": "self_calibrated",
                    "candidate_seed_top_k": [18, 19, 20],
                    "per_budget_metrics": {},
                    "selected_utility": 0.8,
                    "runner_up_seed_top_k": 20,
                    "runner_up_utility": 0.79,
                    "utility_gap": 0.01,
                    "tiebreak_applied": False,
                    "tiebreak_reason": "argmax_utility",
                    "rule": {"enabled": True, "mode": "self_calibrated"},
                },
            },
        ) as calibration_mock, patch(
            "paper_new_selector.stage1_runner.greedy_select_candidates",
        ) as greedy_mock, patch(
            "paper_new_selector.stage1_runner.build_boundary_state",
            return_value={"negative_pattern_stats": {"count": 1}},
        ):
            summary = run_stage1("dummy.yaml", validate_only=False)

        calibration_mock.assert_called_once()
        greedy_mock.assert_not_called()
        self.assertEqual(summary["seed_budget"]["mode"], "self_calibrated")
        self.assertEqual(summary["seed_budget"]["resolved_seed_top_k"], 19)

    def test_stage1_runner_uses_self_calibrated_constrained_seed_budget_path(self):
        fake_backend = _FakeTextBackend()
        fake_embedder = _FakeEmbedder()
        config = {
            "pipeline": {"stage1_mode": "selector_seed_search"},
            "generator": {
                "initial_prompt": "prompt",
                "candidate_count": 2,
                "max_rounds": 1,
                "exemplars_per_prompt": 1,
            },
            "meta": {"seed": 42},
            "selector": {
                "private_knn_k": 1,
                "density_lambda": 0.0,
                "novelty_lambda": 0.0,
                "length_lambda": 0.0,
                "length_floor": 1,
                "length_ceiling": 100,
                "rank_weights": [1.0],
                "top_q": 1,
                "reference_top_k": 1,
                "lambda_generic": 0.2,
                "lambda_redundancy": 0.3,
                "seed_top_k": 20,
                "hard_negative_top_k": 1,
                "seed_budget_rule": {
                    "enabled": True,
                    "mode": "self_calibrated_constrained",
                    "candidate_seed_top_k": [18, 19, 20, 21, 22],
                    "coverage_constraint": {
                        "metric": "coverage_p25",
                        "relative_ratio": 0.99,
                    },
                },
            },
            "privacy": {"enabled": False, "delta": 1e-5},
            "stage1": {"sigma": 0.0, "delta": 1e-5},
        }
        sample_bundle = {
            "train_samples": [_FakeSample("private alpha"), _FakeSample("private beta")],
            "eval_samples": [_FakeSample("eval alpha")],
            "init_samples": [_FakeSample("seed alpha"), _FakeSample("seed beta")],
        }
        decision = SimpleNamespace(
            selected_indices=[0],
            hard_negative_indices=[1],
            hard_negative_reason={1: "boundary_negative"},
            accept_scores=[0.9, 0.2],
            to_dict=lambda: {"selected_indices": [0], "hard_negative_indices": [1]},
        )

        with _round_context_patch(), patch(
            "paper_new_selector.stage1_runner.load_yaml_config",
            return_value=config,
        ), patch(
            "paper_new_selector.stage1_runner.load_text_samples",
            return_value=sample_bundle,
        ), patch(
            "paper_new_selector.stage1_runner.build_candidate_generator",
            return_value=SimpleNamespace(
                generator=_FakeGenerator(),
                text_backend=fake_backend,
                contract={"llm_backend": "vllm"},
            ),
        ), patch(
            "paper_new_selector.stage1_runner.build_embedder_from_config",
            return_value=fake_embedder,
        ), patch(
            "paper_new_selector.stage1_runner.build_private_importance_weights",
            return_value=[1.0, 1.0],
        ), patch(
            "paper_new_selector.stage1_runner.compute_private_support",
            return_value=[0.9, 0.2],
        ), patch(
            "paper_new_selector.stage1_runner.apply_gaussian_privacy_noise",
            side_effect=lambda scores, **_: scores,
        ), patch(
            "paper_new_selector.stage1_runner.compute_genericity_penalties",
            return_value=[0.1, 0.3],
        ), patch(
            "paper_new_selector.stage1_runner.resolve_seed_top_k_by_self_calibration",
            return_value={
                "decision": decision,
                "seed_budget_summary": {
                    "configured_seed_top_k": 20,
                    "resolved_seed_top_k": 20,
                    "mode": "self_calibrated_constrained",
                    "candidate_seed_top_k": [18, 19, 20, 21, 22],
                    "coverage_constraint": {
                        "metric": "coverage_p25",
                        "relative_ratio": 0.99,
                        "feasible_budgets": [20, 22],
                    },
                    "selection_stage": "feasible_set_utility",
                    "fallback_used": False,
                    "per_budget_metrics": {},
                    "selected_utility": 0.8,
                    "runner_up_seed_top_k": 22,
                    "runner_up_utility": 0.79,
                    "utility_gap": 0.01,
                    "tiebreak_applied": False,
                    "tiebreak_reason": "argmax_feasible_utility",
                    "rule": {"enabled": True, "mode": "self_calibrated_constrained"},
                },
            },
        ) as calibration_mock, patch(
            "paper_new_selector.stage1_runner.greedy_select_candidates",
        ) as greedy_mock, patch(
            "paper_new_selector.stage1_runner.build_boundary_state",
            return_value={"negative_pattern_stats": {"count": 1}},
        ):
            summary = run_stage1("dummy.yaml", validate_only=False)

        calibration_mock.assert_called_once()
        greedy_mock.assert_not_called()
        self.assertEqual(summary["seed_budget"]["mode"], "self_calibrated_constrained")
        self.assertEqual(summary["seed_budget"]["coverage_constraint"]["feasible_budgets"], [20, 22])
        self.assertEqual(summary["seed_budget"]["selection_stage"], "feasible_set_utility")
        self.assertFalse(summary["seed_budget"]["fallback_used"])

    def test_stage1_runner_uses_hybrid_length_lock_for_forums_like_lengths(self):
        fake_backend = _FakeTextBackend()
        fake_embedder = _FakeEmbedder()
        config = {
            "pipeline": {"stage1_mode": "selector_seed_search"},
            "generator": {
                "initial_prompt": "prompt",
                "candidate_count": 2,
                "max_rounds": 1,
                "exemplars_per_prompt": 1,
            },
            "meta": {"seed": 42},
            "selector": {
                "private_knn_k": 1,
                "density_lambda": 0.0,
                "novelty_lambda": 0.0,
                "length_lambda": 0.0,
                "length_floor": 1,
                "length_ceiling": 100,
                "rank_weights": [1.0],
                "top_q": 1,
                "reference_top_k": 1,
                "lambda_generic": 0.2,
                "lambda_redundancy": 0.3,
                "seed_top_k": 20,
                "hard_negative_top_k": 1,
                "seed_budget_rule": {
                    "enabled": True,
                    "mode": "hybrid_length_family_constrained",
                    "length_family_lock_budgets": [22],
                    "fallback_mode": "self_calibrated_constrained",
                    "candidate_seed_top_k": [18, 19, 20, 21, 22],
                },
            },
            "privacy": {"enabled": False, "delta": 1e-5},
            "stage1": {"sigma": 0.0, "delta": 1e-5},
        }
        sample_bundle = {
            "train_samples": [
                _FakeSample(" ".join(["forum"] * 150)),
                _FakeSample(" ".join(["forum"] * 205)),
                _FakeSample(" ".join(["forum"] * 396)),
            ],
            "eval_samples": [_FakeSample("eval alpha")],
            "init_samples": [_FakeSample("seed alpha"), _FakeSample("seed beta")],
        }
        decision = SimpleNamespace(
            selected_indices=[0],
            hard_negative_indices=[1],
            hard_negative_reason={1: "boundary_negative"},
            accept_scores=[0.9, 0.2],
            to_dict=lambda: {"selected_indices": [0], "hard_negative_indices": [1]},
        )

        with _round_context_patch(), patch(
            "paper_new_selector.stage1_runner.load_yaml_config",
            return_value=config,
        ), patch(
            "paper_new_selector.stage1_runner.load_text_samples",
            return_value=sample_bundle,
        ), patch(
            "paper_new_selector.stage1_runner.build_candidate_generator",
            return_value=SimpleNamespace(
                generator=_FakeGenerator(),
                text_backend=fake_backend,
                contract={"llm_backend": "vllm"},
            ),
        ), patch(
            "paper_new_selector.stage1_runner.build_embedder_from_config",
            return_value=fake_embedder,
        ), patch(
            "paper_new_selector.stage1_runner.build_private_importance_weights",
            return_value=[1.0, 1.0, 1.0],
        ), patch(
            "paper_new_selector.stage1_runner.compute_private_support",
            return_value=[0.9, 0.2],
        ), patch(
            "paper_new_selector.stage1_runner.apply_gaussian_privacy_noise",
            side_effect=lambda scores, **_: scores,
        ), patch(
            "paper_new_selector.stage1_runner.compute_genericity_penalties",
            return_value=[0.1, 0.3],
        ), patch(
            "paper_new_selector.stage1_runner.greedy_select_candidates",
            return_value=decision,
        ) as greedy_mock, patch(
            "paper_new_selector.stage1_runner.resolve_seed_top_k_by_self_calibration",
        ) as calibration_mock, patch(
            "paper_new_selector.stage1_runner.build_boundary_state",
            return_value={"negative_pattern_stats": {"count": 1}},
        ):
            summary = run_stage1("dummy.yaml", validate_only=False)

        calibration_mock.assert_not_called()
        greedy_kwargs = greedy_mock.call_args.kwargs
        self.assertEqual(greedy_kwargs["seed_top_k"], 22)
        self.assertEqual(
            summary["seed_budget"]["mode"], "hybrid_length_family_constrained"
        )
        self.assertEqual(summary["seed_budget"]["selection_source"], "length_family_lock")
        self.assertEqual(summary["seed_budget"]["resolved_seed_top_k"], 22)
        self.assertEqual(summary["seed_budget"]["length_family_resolved_seed_top_k"], 22)

    def test_stage1_runner_uses_hybrid_constrained_fallback_for_non_forums_lengths(self):
        fake_backend = _FakeTextBackend()
        fake_embedder = _FakeEmbedder()
        config = {
            "pipeline": {"stage1_mode": "selector_seed_search"},
            "generator": {
                "initial_prompt": "prompt",
                "candidate_count": 2,
                "max_rounds": 1,
                "exemplars_per_prompt": 1,
            },
            "meta": {"seed": 42},
            "selector": {
                "private_knn_k": 1,
                "density_lambda": 0.0,
                "novelty_lambda": 0.0,
                "length_lambda": 0.0,
                "length_floor": 1,
                "length_ceiling": 100,
                "rank_weights": [1.0],
                "top_q": 1,
                "reference_top_k": 1,
                "lambda_generic": 0.2,
                "lambda_redundancy": 0.3,
                "seed_top_k": 20,
                "hard_negative_top_k": 1,
                "seed_budget_rule": {
                    "enabled": True,
                    "mode": "hybrid_length_family_constrained",
                    "length_family_lock_budgets": [22],
                    "fallback_mode": "self_calibrated_constrained",
                    "candidate_seed_top_k": [18, 19, 20, 21, 22],
                },
            },
            "privacy": {"enabled": False, "delta": 1e-5},
            "stage1": {"sigma": 0.0, "delta": 1e-5},
        }
        sample_bundle = {
            "train_samples": [
                _FakeSample(" ".join(["short"] * 80)),
                _FakeSample(" ".join(["short"] * 99)),
                _FakeSample(" ".join(["short"] * 173)),
            ],
            "eval_samples": [_FakeSample("eval alpha")],
            "init_samples": [_FakeSample("seed alpha"), _FakeSample("seed beta")],
        }
        decision = SimpleNamespace(
            selected_indices=[0],
            hard_negative_indices=[1],
            hard_negative_reason={1: "boundary_negative"},
            accept_scores=[0.9, 0.2],
            to_dict=lambda: {"selected_indices": [0], "hard_negative_indices": [1]},
        )

        with _round_context_patch(), patch(
            "paper_new_selector.stage1_runner.load_yaml_config",
            return_value=config,
        ), patch(
            "paper_new_selector.stage1_runner.load_text_samples",
            return_value=sample_bundle,
        ), patch(
            "paper_new_selector.stage1_runner.build_candidate_generator",
            return_value=SimpleNamespace(
                generator=_FakeGenerator(),
                text_backend=fake_backend,
                contract={"llm_backend": "vllm"},
            ),
        ), patch(
            "paper_new_selector.stage1_runner.build_embedder_from_config",
            return_value=fake_embedder,
        ), patch(
            "paper_new_selector.stage1_runner.build_private_importance_weights",
            return_value=[1.0, 1.0, 1.0],
        ), patch(
            "paper_new_selector.stage1_runner.compute_private_support",
            return_value=[0.9, 0.2],
        ), patch(
            "paper_new_selector.stage1_runner.apply_gaussian_privacy_noise",
            side_effect=lambda scores, **_: scores,
        ), patch(
            "paper_new_selector.stage1_runner.compute_genericity_penalties",
            return_value=[0.1, 0.3],
        ), patch(
            "paper_new_selector.stage1_runner.resolve_seed_top_k_by_self_calibration",
            return_value={
                "decision": decision,
                "seed_budget_summary": {
                    "configured_seed_top_k": 20,
                    "resolved_seed_top_k": 18,
                    "mode": "self_calibrated_constrained",
                    "candidate_seed_top_k": [18, 19, 20, 21, 22],
                    "coverage_constraint": {"feasible_budgets": [18, 19, 20, 21, 22]},
                    "selection_stage": "feasible_set_utility",
                    "fallback_used": False,
                    "per_budget_metrics": {},
                    "selected_utility": 0.8,
                    "runner_up_seed_top_k": 19,
                    "runner_up_utility": 0.79,
                    "utility_gap": 0.01,
                    "tiebreak_applied": False,
                    "tiebreak_reason": "argmax_feasible_utility",
                    "rule": {"enabled": True, "mode": "self_calibrated_constrained"},
                },
            },
        ) as calibration_mock, patch(
            "paper_new_selector.stage1_runner.greedy_select_candidates",
        ) as greedy_mock, patch(
            "paper_new_selector.stage1_runner.build_boundary_state",
            return_value={"negative_pattern_stats": {"count": 1}},
        ):
            summary = run_stage1("dummy.yaml", validate_only=False)

        calibration_mock.assert_called_once()
        greedy_mock.assert_not_called()
        calibration_selector_cfg = calibration_mock.call_args.kwargs["selector_cfg"]
        self.assertEqual(
            calibration_selector_cfg["seed_budget_rule"]["mode"],
            "self_calibrated_constrained",
        )
        self.assertEqual(
            summary["seed_budget"]["mode"], "hybrid_length_family_constrained"
        )
        self.assertEqual(
            summary["seed_budget"]["selection_source"],
            "self_calibrated_constrained",
        )
        self.assertEqual(summary["seed_budget"]["resolved_seed_top_k"], 18)
        self.assertEqual(summary["seed_budget"]["length_family_resolved_seed_top_k"], 19)

    def test_stage1_runner_passes_length_modulation_config_to_genericity(self):
        fake_backend = _FakeTextBackend()
        fake_embedder = _FakeEmbedder()
        config = {
            "pipeline": {"stage1_mode": "selector_seed_search"},
            "generator": {
                "initial_prompt": "prompt",
                "candidate_count": 2,
                "max_rounds": 1,
                "exemplars_per_prompt": 1,
            },
            "meta": {"seed": 42},
            "selector": {
                "private_knn_k": 1,
                "density_lambda": 0.0,
                "novelty_lambda": 0.0,
                "length_lambda": 0.0,
                "length_floor": 1,
                "length_ceiling": 100,
                "rank_weights": [1.0],
                "top_q": 1,
                "reference_top_k": 6,
                "reference_rank_weights": [1.0, 0.8, 0.6, 0.4, 0.25, 0.1],
                "genericity_gate_low": 0.78,
                "genericity_gate_high": 0.90,
                "genericity_gate_low_scale": 0.10,
                "genericity_gate_mid_scale": 0.45,
                "length_modulation_enabled": True,
                "length_alpha": 0.3,
                "length_factor_min": 0.2,
                "length_factor_max": 5.0,
                "lambda_generic": 0.2,
                "lambda_redundancy": 0.3,
                "seed_top_k": 1,
                "hard_negative_top_k": 1,
            },
            "privacy": {"enabled": False, "delta": 1e-5},
            "stage1": {"sigma": 0.0, "delta": 1e-5},
        }
        sample_bundle = {
            "train_samples": [_FakeSample("private alpha"), _FakeSample("private beta")],
            "eval_samples": [_FakeSample("eval alpha")],
            "init_samples": [_FakeSample("seed alpha"), _FakeSample("seed beta")],
        }
        decision = SimpleNamespace(
            seed_indices=[0],
            hard_negative_indices=[1],
            selected_scores={0: 0.7, 1: 0.3},
            seed_score_breakdown=[],
            hard_negative_score_breakdown=[],
            selected_indices=[0],
            hard_negative_reason={1: "boundary_negative"},
            accept_scores=[0.9, 0.2],
            to_dict=lambda: {"selected_indices": [0], "hard_negative_indices": [1]},
        )
        with _round_context_patch(), patch(
            "paper_new_selector.stage1_runner.load_yaml_config",
            return_value=config,
        ), patch(
            "paper_new_selector.stage1_runner.load_text_samples",
            return_value=sample_bundle,
        ), patch(
            "paper_new_selector.stage1_runner.build_candidate_generator",
            return_value=SimpleNamespace(
                generator=_FakeGenerator(),
                text_backend=fake_backend,
                contract={"llm_backend": "vllm"},
            ),
        ), patch(
            "paper_new_selector.stage1_runner.build_embedder_from_config",
            return_value=fake_embedder,
        ), patch(
            "paper_new_selector.stage1_runner.build_private_importance_weights",
            return_value=[1.0, 1.0],
        ), patch(
            "paper_new_selector.stage1_runner.compute_private_support",
            return_value=[0.9, 0.2],
        ), patch(
            "paper_new_selector.stage1_runner.apply_gaussian_privacy_noise",
            side_effect=lambda scores, **_: scores,
        ), patch(
            "paper_new_selector.stage1_runner.compute_genericity_penalties",
            return_value=[0.1, 0.3],
        ) as genericity_mock, patch(
            "paper_new_selector.stage1_runner.greedy_select_candidates",
            return_value=decision,
        ), patch(
            "paper_new_selector.stage1_runner.build_boundary_state",
            return_value={"negative_pattern_stats": {"count": 1}},
        ):
            run_stage1("dummy.yaml", validate_only=False)

        kwargs = genericity_mock.call_args.kwargs
        self.assertTrue(kwargs["length_modulation_enabled"])
        self.assertEqual(kwargs["length_alpha"], 0.3)
        self.assertEqual(kwargs["length_factor_min"], 0.2)
        self.assertEqual(kwargs["length_factor_max"], 5.0)
        self.assertIn("candidate_lengths", kwargs)
