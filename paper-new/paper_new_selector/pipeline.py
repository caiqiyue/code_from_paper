from __future__ import annotations

from pathlib import Path
from typing import Any

from .eval_bridge import prepare_eval_runtime, run_eval
from .pretext_bridge import prepare_bootstrap_runtime
from .stage1_runner import run_stage1
from .thesis_bridge import load_yaml_config, resolve_output_root


def run_pipeline(config_path: str | Path, *, validate_only: bool = False) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    stage1_summary = run_stage1(config_path, validate_only=validate_only)
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
        },
        "eval": eval_runtime,
    }

    if validate_only:
        return summary

    selected_texts = list(stage1_summary["selected_texts"])
    prompt_list = bootstrap_runtime["build_bootstrap_prompts"](
        selected_texts,
        num_prompts=int(bootstrap_runtime["bootstrap_cfg"]["num_prompts"]),
        seed=int(config.get("meta", {}).get("seed", 42)),
    )
    generated_outputs = bootstrap_runtime["generate_bootstrapped_samples"](
        prompt_list,
        bootstrap_runtime["model_path"],
        bootstrap_runtime["bootstrap_cfg"],
    )
    summary["stage2"]["prompt_count"] = len(prompt_list)
    summary["stage2"]["generated_count"] = len(generated_outputs)
    summary["stage2"]["synthetic_outputs"] = generated_outputs
    if eval_runtime.get("enabled", False):
        summary["eval"] = run_eval(
            synthetic_texts=generated_outputs,
            config_path=config_path,
            output_dir=resolve_output_root(config_path) / "eval",
        )
    return summary
