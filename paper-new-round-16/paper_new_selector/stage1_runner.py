from __future__ import annotations

import math
import random
import statistics
from pathlib import Path
from typing import Any

from .boundary import build_boundary_state
from .budget_calibration import resolve_seed_top_k_by_self_calibration
from .generator_bridge import build_candidate_generator
from .genericity import compute_genericity_penalties
from .importance import build_private_importance_weights
from .runtime_cleanup import release_runtime_memory
from .selector import greedy_select_candidates
from .support import apply_gaussian_privacy_noise, compute_private_support
from .thesis_bridge import (
    build_embedder_from_config,
    load_text_samples,
    load_yaml_config,
)


def _clean_candidate_texts(texts: list[str]) -> list[str]:
    cleaned: list[str] = []
    for text in texts:
        normalized = str(text).strip()
        if not normalized:
            continue
        if len(normalized.split()) < 2:
            continue
        cleaned.append(normalized)
    return cleaned


def _vectorize(embedder: Any, texts: list[str]) -> list[list[float]]:
    return [list(map(float, row)) for row in embedder.embed_texts(texts)]


def _select_seed_samples(
    init_samples: list[Any],
    *,
    exemplar_count: int,
    round_id: int,
    meta_seed: int,
) -> list[Any]:
    if not init_samples:
        return []
    exemplar_count = max(1, min(exemplar_count, len(init_samples)))
    rng = random.Random(int(meta_seed) + int(round_id))
    if exemplar_count >= len(init_samples):
        return list(init_samples)
    selected_indices = sorted(rng.sample(range(len(init_samples)), exemplar_count))
    return [init_samples[index] for index in selected_indices]


def _percentile_nearest_rank(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    if percentile <= 0:
        return float(min(values))
    if percentile >= 100:
        return float(max(values))
    sorted_values = sorted(int(value) for value in values)
    rank = math.ceil((float(percentile) / 100.0) * len(sorted_values))
    return float(sorted_values[max(0, rank - 1)])


def compute_private_length_stats(private_lengths: list[int]) -> dict[str, float]:
    if not private_lengths:
        return {"mean": 0.0, "median": 0.0, "p75": 0.0}
    return {
        "mean": float(statistics.mean(private_lengths)),
        "median": float(statistics.median(private_lengths)),
        "p75": _percentile_nearest_rank(private_lengths, 75),
    }


def resolve_seed_top_k(selector_cfg: dict[str, Any], private_lengths: list[int]) -> int:
    rule_cfg = dict(selector_cfg.get("seed_budget_rule", {}))
    if not bool(rule_cfg.get("enabled", False)):
        return int(selector_cfg["seed_top_k"])
    if not private_lengths:
        return int(selector_cfg["seed_top_k"])

    mode = str(rule_cfg.get("mode", "length_family"))
    if mode != "length_family":
        raise ValueError(f"Unsupported seed_budget_rule.mode: {mode}")

    stats = compute_private_length_stats(private_lengths)
    median_len = stats["median"]
    mean_len = stats["mean"]
    p75_len = stats["p75"]

    if median_len <= 120:
        return 19
    if p75_len >= 390 or (mean_len >= 335 and median_len >= 200):
        return 22
    if mean_len >= 340:
        return 18
    return 20


def run_stage1_with_runtime(
    config_path: str | Path,
    *,
    validate_only: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_yaml_config(config_path)
    sample_bundle = load_text_samples(config_path)
    generator_handle = build_candidate_generator(config_path)
    embedder: Any | None = None
    if validate_only:
        return (
            {
                "mode": str(config["pipeline"]["stage1_mode"]),
                "train_count": len(sample_bundle["train_samples"]),
                "eval_count": len(sample_bundle["eval_samples"]),
                "init_count": len(sample_bundle["init_samples"]),
                "generator_contract": dict(generator_handle.contract),
                "shared_session": (
                    generator_handle.shared_session.to_dict()
                    if getattr(generator_handle, "shared_session", None) is not None
                    else None
                ),
                "boundary_state": {
                    "reject_score_ceiling": 0.0,
                    "reject_score_floor": 0.0,
                    "negative_centroid": [],
                    "negative_pattern_stats": {"count": 0},
                },
            },
            {
                "generator_handle": generator_handle,
                "shared_session": getattr(generator_handle, "shared_session", None),
                "embedder": None,
            },
        )

    try:
        embedder = build_embedder_from_config(config_path)
        private_samples = sample_bundle["train_samples"]
        init_samples = sample_bundle["init_samples"]
        prompt_text = str(config["generator"]["initial_prompt"])
        candidate_count = int(config["generator"]["candidate_count"])
        max_rounds = int(config["generator"].get("max_rounds", max(1, candidate_count)))
        meta_seed = int(config.get("meta", {}).get("seed", 42))
        exemplar_count = int(config["generator"]["exemplars_per_prompt"])

        candidate_texts: list[str] = []
        round_id = 0
        while len(candidate_texts) < candidate_count and round_id < max_rounds:
            seed_samples = _select_seed_samples(
                init_samples,
                exemplar_count=exemplar_count,
                round_id=round_id,
                meta_seed=meta_seed,
            )
            from thesis_platform.core.context import RoundContext

            round_ctx = RoundContext(
                round_id=round_id,
                prompt_text=prompt_text,
                public_seed_samples=seed_samples,
                config=config,
                output_dir=None,
                text_backend=generator_handle.text_backend,
                sample_id_prefix="paper_new_selector",
                sample_source="selector_candidate",
            )
            generated = generator_handle.generator.generate(round_ctx)
            before = len(candidate_texts)
            candidate_texts.extend(
                _clean_candidate_texts([sample.render_text() for sample in generated])
            )
            if len(candidate_texts) == before:
                raise RuntimeError(
                    "Stage 1 candidate generation produced no usable texts. "
                    "Check llm.generator model/backend or candidate-cleaning constraints."
                )
            round_id += 1
        if len(candidate_texts) < candidate_count:
            raise RuntimeError(
                f"Stage 1 candidate generation stopped after {round_id} rounds with only "
                f"{len(candidate_texts)} usable candidates, below candidate_count={candidate_count}."
            )
        candidate_texts = candidate_texts[:candidate_count]

        private_texts = [sample.render_text() for sample in private_samples]
        init_texts = [sample.render_text() for sample in init_samples]
        private_vectors = _vectorize(embedder, private_texts)
        candidate_vectors = _vectorize(embedder, candidate_texts)
        reference_vectors = _vectorize(embedder, init_texts)
        private_lengths = [len(text.split()) for text in private_texts]

        selector_cfg = config["selector"]
        stage1_cfg = dict(config.get("stage1", {}))
        privacy_cfg = dict(config.get("privacy", {}))
        private_weights = build_private_importance_weights(
            private_vectors=private_vectors,
            private_lengths=private_lengths,
            private_knn_k=int(selector_cfg["private_knn_k"]),
            density_lambda=float(selector_cfg["density_lambda"]),
            novelty_lambda=float(selector_cfg["novelty_lambda"]),
            length_lambda=float(selector_cfg["length_lambda"]),
            length_floor=int(selector_cfg["length_floor"]),
            length_ceiling=int(selector_cfg["length_ceiling"]),
        )
        raw_private_support = compute_private_support(
            private_vectors=private_vectors,
            candidate_vectors=candidate_vectors,
            private_weights=private_weights,
            rank_weights=list(selector_cfg["rank_weights"]),
            top_q=int(selector_cfg["top_q"]),
        )
        privacy_enabled = bool(privacy_cfg.get("enabled", False)) and not bool(
            stage1_cfg.get("privacy_disabled", False)
        )
        privacy_sigma = float(stage1_cfg.get("sigma", 0.0))
        private_support = apply_gaussian_privacy_noise(
            raw_private_support,
            enabled=privacy_enabled,
            sigma=privacy_sigma,
            seed=meta_seed,
        )
        _dataset_name = str(config.get("data", {}).get("dataset_name", ""))
        if _dataset_name == "forums":
            _overrides = [
                ("_forums_lambda_generic", "lambda_generic"),
                ("_forums_lambda_redundancy", "lambda_redundancy"),
                ("_forums_seed_top_k", "seed_top_k"),
                ("_forums_gate_low", "genericity_gate_low"),
                ("_forums_gate_high", "genericity_gate_high"),
                ("_forums_low_scale", "genericity_gate_low_scale"),
                ("_forums_mid_scale", "genericity_gate_mid_scale"),
            ]
            for _src_key, _tgt_key in _overrides:
                if _src_key in selector_cfg:
                    selector_cfg[_tgt_key] = float(selector_cfg[_src_key])
        candidate_lengths = [len(text.split()) for text in candidate_texts]
        private_length_stats = compute_private_length_stats(private_lengths)
        genericity_penalty = compute_genericity_penalties(
            candidate_vectors=candidate_vectors,
            reference_vectors=reference_vectors,
            reference_top_k=int(selector_cfg["reference_top_k"]),
            reference_rank_weights=list(selector_cfg.get("reference_rank_weights", [])),
            apply_gate=True,
            gate_low=float(selector_cfg.get("genericity_gate_low", 0.0)),
            gate_high=float(selector_cfg.get("genericity_gate_high", 1.0)),
            low_scale=float(selector_cfg.get("genericity_gate_low_scale", 1.0)),
            mid_scale=float(selector_cfg.get("genericity_gate_mid_scale", 1.0)),
            candidate_lengths=candidate_lengths,
            length_modulation_enabled=bool(
                selector_cfg.get("length_modulation_enabled", False)
            ),
            length_alpha=float(selector_cfg.get("length_alpha", 0.0)),
            length_factor_min=float(selector_cfg.get("length_factor_min", 0.2)),
            length_factor_max=float(selector_cfg.get("length_factor_max", 5.0)),
        )
        rule_cfg = dict(selector_cfg.get("seed_budget_rule", {}))
        rule_mode = str(rule_cfg.get("mode", "length_family"))
        if bool(rule_cfg.get("enabled", False)) and rule_mode in {
            "self_calibrated",
            "self_calibrated_constrained",
        }:
            calibration_result = resolve_seed_top_k_by_self_calibration(
                selector_cfg=selector_cfg,
                candidate_vectors=candidate_vectors,
                candidate_texts=candidate_texts,
                private_vectors=private_vectors,
                private_support=private_support,
                genericity_penalty=genericity_penalty,
            )
            decision = calibration_result["decision"]
            seed_budget_summary = dict(calibration_result["seed_budget_summary"])
        else:
            resolved_seed_top_k = resolve_seed_top_k(selector_cfg, private_lengths)
            decision = greedy_select_candidates(
                candidate_vectors=candidate_vectors,
                candidate_texts=candidate_texts,
                private_support=private_support,
                genericity_penalty=genericity_penalty,
                lambda_generic=float(selector_cfg["lambda_generic"]),
                lambda_redundancy=float(selector_cfg["lambda_redundancy"]),
                seed_top_k=resolved_seed_top_k,
                hard_negative_top_k=int(selector_cfg["hard_negative_top_k"]),
            )
            seed_budget_summary = {
                "configured_seed_top_k": int(selector_cfg["seed_top_k"]),
                "resolved_seed_top_k": resolved_seed_top_k,
                "rule": rule_cfg,
                "mode": rule_mode if bool(rule_cfg.get("enabled", False)) else "disabled",
                "private_length_mean": private_length_stats["mean"],
                "private_length_median": private_length_stats["median"],
                "private_length_p75": private_length_stats["p75"],
            }

        reject_scores = [
            decision.accept_scores[index] for index in decision.hard_negative_indices
        ]
        reject_vectors = [
            candidate_vectors[index] for index in decision.hard_negative_indices
        ]
        boundary_state = build_boundary_state(
            reject_scores=reject_scores, reject_vectors=reject_vectors
        )
        selected_texts = [candidate_texts[index] for index in decision.selected_indices]
        hard_negative_texts = [
            candidate_texts[index] for index in decision.hard_negative_indices
        ]
        return {
            "mode": str(config["pipeline"]["stage1_mode"]),
            "selected_indices": decision.selected_indices,
            "hard_negative_indices": decision.hard_negative_indices,
            "selected_texts": selected_texts,
            "hard_negative_texts": hard_negative_texts,
            "hard_negative_reason": decision.hard_negative_reason,
            "boundary_state": boundary_state,
            "generator_contract": dict(generator_handle.contract),
            "privacy": {
                "enabled": privacy_enabled,
                "sigma": privacy_sigma,
                "delta": float(stage1_cfg.get("delta", privacy_cfg.get("delta", 0.0))),
            },
            "seed_budget": seed_budget_summary,
            "decision": decision.to_dict(),
            "shared_session": (
                generator_handle.shared_session.to_dict()
                if getattr(generator_handle, "shared_session", None) is not None
                else None
            ),
        }, {
            "generator_handle": generator_handle,
            "shared_session": getattr(generator_handle, "shared_session", None),
            "embedder": embedder,
        }
    except Exception:
        release_runtime_memory(
            getattr(generator_handle, "text_backend", None), embedder
        )
        raise


def run_stage1(
    config_path: str | Path, *, validate_only: bool = False
) -> dict[str, Any]:
    summary, runtime = run_stage1_with_runtime(config_path, validate_only=validate_only)
    release_runtime_memory(
        getattr(runtime.get("generator_handle"), "text_backend", None),
        runtime.get("embedder"),
    )
    return summary
