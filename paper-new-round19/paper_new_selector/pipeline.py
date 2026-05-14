from __future__ import annotations

from pathlib import Path
from typing import Any

from .eval_bridge import prepare_eval_runtime, run_eval
from .pretext_bridge import prepare_bootstrap_runtime
from .runtime_cleanup import release_runtime_memory
from .stage1_runner import run_stage1_with_runtime
from .synthetic_contract import resolve_downstream_synthetic_texts
from .thesis_bridge import load_yaml_config, resolve_output_root, write_json, write_jsonl


def _has_shared_backend(shared_session: Any) -> bool:
    if isinstance(shared_session, dict):
        return shared_session.get("backend") is not None
    return getattr(shared_session, "backend", None) is not None


def _normalize_collection_config(config: dict[str, Any]) -> dict[str, Any]:
    raw_cfg = dict(config.get("collection", {}))
    return {
        "enabled": bool(raw_cfg.get("enabled", False)),
        "schema_version": str(
            raw_cfg.get("schema_version", "round23_collection_v1")
        ),
        "write_context_summary": bool(raw_cfg.get("write_context_summary", True)),
        "write_budget_table": bool(raw_cfg.get("write_budget_table", True)),
        "write_final_result_summary": bool(
            raw_cfg.get("write_final_result_summary", True)
        ),
        "write_round_table": bool(raw_cfg.get("write_round_table", False)),
        "write_runtime_table": bool(raw_cfg.get("write_runtime_table", False)),
    }


def _build_final_result_summary(
    *,
    config: dict[str, Any],
    output_root: Path,
    stage1_summary: dict[str, Any],
    stage2_summary: dict[str, Any],
    eval_summary: dict[str, Any],
    collection_cfg: dict[str, Any],
) -> dict[str, Any]:
    seed_budget = dict(stage1_summary.get("seed_budget", {}))
    return {
        "schema_version": str(collection_cfg["schema_version"]),
        "experiment_id": str(config.get("meta", {}).get("experiment_id", "")),
        "dataset_name": str(config.get("data", {}).get("dataset_name", "")),
        "output_root": str(output_root),
        "stage1_mode": str(config["pipeline"]["stage1_mode"]),
        "stage2_mode": str(config["pipeline"]["stage2_mode"]),
        "resolved_seed_top_k": seed_budget.get("resolved_seed_top_k"),
        "candidate_seed_top_k": list(seed_budget.get("candidate_seed_top_k", [])),
        "selected_count": len(stage1_summary.get("selected_texts", [])),
        "synthetic_train_count": int(stage2_summary.get("generated_count", 0)),
        "generation_path": stage2_summary.get("generation_path"),
        "eval_enabled": bool(eval_summary.get("enabled", False)),
        "eval_mode": eval_summary.get("mode"),
        "eval_status": eval_summary.get("status"),
        "best_top1": eval_summary.get("best_top1"),
        "best_top3": eval_summary.get("best_top3"),
        "best_top5": eval_summary.get("best_top5"),
        "best_top10": eval_summary.get("best_top10"),
    }


def _write_collection_artifacts(
    *,
    config: dict[str, Any],
    output_root: Path,
    stage1_summary: dict[str, Any],
    stage2_summary: dict[str, Any],
    eval_summary: dict[str, Any],
    collection_cfg: dict[str, Any],
) -> None:
    if not bool(collection_cfg.get("enabled", False)):
        return
    collection_root = output_root / "collection"
    if collection_cfg["write_context_summary"] and stage1_summary.get("context_summary") is not None:
        write_json(collection_root / "context_summary.json", stage1_summary["context_summary"])
    if collection_cfg["write_budget_table"]:
        budget_rows = list(
            stage1_summary.get("budget_collection", {}).get("per_budget_rows", [])
        )
        write_jsonl(collection_root / "budget_table.jsonl", budget_rows)
    if collection_cfg["write_final_result_summary"]:
        write_json(
            collection_root / "final_result_summary.json",
            _build_final_result_summary(
                config=config,
                output_root=output_root,
                stage1_summary=stage1_summary,
                stage2_summary=stage2_summary,
                eval_summary=eval_summary,
                collection_cfg=collection_cfg,
            ),
        )


def run_pipeline(config_path: str | Path, *, validate_only: bool = False) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    collection_cfg = _normalize_collection_config(config)
    stage1_summary, stage1_runtime = run_stage1_with_runtime(config_path, validate_only=validate_only)
    bootstrap_runtime = prepare_bootstrap_runtime(config_path)
    eval_runtime = prepare_eval_runtime(config_path)

    summary: dict[str, Any] = {
        "stage1_mode": str(config["pipeline"]["stage1_mode"]),
        "stage2_mode": str(config["pipeline"]["stage2_mode"]),
        "generator_contract": dict(stage1_summary["generator_contract"]),
        "stage1": stage1_summary,
        "stage2": {
            "bootstrap_cfg": dict(bootstrap_runtime["bootstrap_cfg"]),
            "model_path": str(bootstrap_runtime["model_path"]),
            "build_bootstrap_prompts": bootstrap_runtime["build_bootstrap_prompts"].__name__,
            "generate_bootstrapped_samples": bootstrap_runtime["generate_bootstrapped_samples"].__name__,
            "generate_with_shared_session": bootstrap_runtime["generate_with_shared_session"].__name__,
        },
        "eval": eval_runtime,
    }
    if collection_cfg["enabled"]:
        summary["collection"] = dict(collection_cfg)

    if validate_only:
        return summary

    output_root = resolve_output_root(config_path)
    write_json(output_root / "stage1_summary.json", stage1_summary)
    if stage1_summary.get("seed_budget", {}).get("mode") in {
        "self_calibrated",
        "self_calibrated_constrained",
        "hybrid_length_family_constrained",
        "hierarchical_shape_routing",
    }:
        write_json(output_root / "stage1_budget_calibration.json", stage1_summary["seed_budget"])

    try:
        direct_outputs = list(stage1_summary.get("direct_synthetic_texts", []))
        bootstrap_outputs: list[str] = []

        if bool(stage1_summary.get("skip_bootstrap", False)):
            summary["stage2"]["prompt_count"] = 0
            summary["stage2"]["generated_count"] = len(direct_outputs)
            summary["stage2"]["synthetic_outputs"] = direct_outputs
        else:
            selected_texts = list(stage1_summary["selected_texts"])
            prompt_list = bootstrap_runtime["build_bootstrap_prompts"](
                selected_texts,
                num_prompts=int(bootstrap_runtime["bootstrap_cfg"]["num_prompts"]),
                seed=int(config.get("meta", {}).get("seed", 42)),
            )
            shared_session = stage1_runtime.get("shared_session")
            if _has_shared_backend(shared_session):
                bootstrap_outputs = bootstrap_runtime["generate_with_shared_session"](
                    prompt_list,
                    shared_session=shared_session,
                    bootstrap_cfg=bootstrap_runtime["bootstrap_cfg"],
                )
                summary["stage2"]["generation_path"] = "shared_session"
            else:
                bootstrap_outputs = bootstrap_runtime["generate_bootstrapped_samples"](
                    prompt_list,
                    bootstrap_runtime["model_path"],
                    bootstrap_runtime["bootstrap_cfg"],
                )
                summary["stage2"]["generation_path"] = "standalone_bootstrap"
            summary["stage2"]["prompt_count"] = len(prompt_list)
            summary["stage2"]["generated_count"] = len(bootstrap_outputs)
            summary["stage2"]["synthetic_outputs"] = bootstrap_outputs
        generated_outputs = resolve_downstream_synthetic_texts(
            stage1_summary=stage1_summary,
            bootstrap_outputs=bootstrap_outputs,
        )
        summary["stage2"]["generated_count"] = len(generated_outputs)
        summary["stage2"]["synthetic_outputs"] = generated_outputs
    finally:
        release_runtime_memory(
            getattr(stage1_runtime.get("generator_handle"), "text_backend", None),
            stage1_runtime.get("embedder"),
        )

    if eval_runtime.get("enabled", False):
        summary["eval"] = run_eval(
            synthetic_texts=generated_outputs,
            config_path=config_path,
            output_dir=output_root / "eval",
        )
    _write_collection_artifacts(
        config=config,
        output_root=output_root,
        stage1_summary=stage1_summary,
        stage2_summary=summary["stage2"],
        eval_summary=summary["eval"],
        collection_cfg=collection_cfg,
    )
    return summary
