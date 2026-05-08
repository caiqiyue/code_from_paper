from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .eval_bridge import prepare_eval_runtime, run_eval
from .pretext_bridge import prepare_bootstrap_runtime
from .runtime_cleanup import release_runtime_memory
from .stage1_runner import run_stage1_with_runtime
from .synthetic_contract import resolve_downstream_synthetic_texts
from .thesis_bridge import load_yaml_config, resolve_output_root, write_json


def _has_shared_backend(shared_session: Any) -> bool:
    if isinstance(shared_session, dict):
        return shared_session.get("backend") is not None
    return getattr(shared_session, "backend", None) is not None


def _run_eval_in_subprocess(
    *,
    synthetic_texts: list[str],
    config_path: str | Path,
    output_dir: Path,
) -> dict[str, Any]:
    synthetic_path = output_dir.parent / "eval_synthetic_texts.json"
    write_json(synthetic_path, list(synthetic_texts))
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "paper_new_selector.eval_subprocess_runner",
            "--config",
            str(config_path),
            "--synthetic-path",
            str(synthetic_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def run_pipeline(config_path: str | Path, *, validate_only: bool = False) -> dict[str, Any]:
    config = load_yaml_config(config_path)
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
        summary["eval"] = _run_eval_in_subprocess(
            synthetic_texts=generated_outputs,
            config_path=config_path,
            output_dir=output_root / "eval",
        )
    return summary
