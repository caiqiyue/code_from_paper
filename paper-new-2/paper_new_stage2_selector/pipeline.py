from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .bootstrap_bridge import attach_generated_outputs, build_bootstrap_prompt_records
from .eval_bridge import run_eval_from_stage2_dir, write_selected_stage2_dir
from .selector import select_seed_aware_records
from .thesis_bridge import (
    build_embedder_from_config,
    load_yaml_config,
    prepare_bootstrap_runtime,
    resolve_output_root,
    run_pretext_stage1,
)


def _records_payload(records: list[Any]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for record in records:
        payload.append(
            {
                "record_index": int(record.record_index),
                "prompt_index": int(record.prompt_index),
                "seed_texts": list(record.seed_texts),
                "raw_text": str(record.raw_text),
                "baseline_text": str(record.baseline_text),
                "consistency_score": float(record.consistency_score),
                "template_penalty": float(record.template_penalty),
                "duplicate_penalty": float(record.duplicate_penalty),
                "final_score": float(record.final_score),
                "rejected_reason": str(record.rejected_reason),
            }
        )
    return payload


def embed_records(records: list[Any], config_path: str | Path) -> tuple[list[list[float]], list[list[list[float]]]]:
    embedder = build_embedder_from_config(config_path)
    try:
        generated_inputs = [record.raw_text for record in records]
        generated_vectors = [list(map(float, row)) for row in embedder.embed_texts(generated_inputs)]
        seed_vectors = []
        for record in records:
            seed_vectors.append([list(map(float, row)) for row in embedder.embed_texts(record.seed_texts)])
        return generated_vectors, seed_vectors
    finally:
        if hasattr(embedder, "release"):
            embedder.release()


def run_pipeline(config_path: str | Path, *, validate_only: bool = False) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    bootstrap_runtime = prepare_bootstrap_runtime(config_path)
    summary: dict[str, Any] = {
        "stage1_mode": str(config["pipeline"]["stage1_mode"]),
        "stage2_mode": str(config["pipeline"]["stage2_mode"]),
        "stage2": {
            "bootstrap_cfg": dict(bootstrap_runtime["bootstrap_cfg"]),
            "model_path": str(bootstrap_runtime["model_path"]),
            "selector": dict(config["selector"]),
        },
    }
    if validate_only:
        return summary

    stage1_runtime = run_pretext_stage1(config_path)
    prompt_records = build_bootstrap_prompt_records(
        stage1_runtime["seed_texts"],
        num_prompts=int(config["bootstrap"]["num_prompts"]),
        seed=int(config.get("meta", {}).get("seed", 42)),
    )
    raw_outputs = bootstrap_runtime["generate_bootstrapped_samples"](
        [record.prompt_text for record in prompt_records],
        bootstrap_runtime["model_path"],
        bootstrap_runtime["bootstrap_cfg"],
    )
    generated_records = attach_generated_outputs(prompt_records, raw_outputs)
    generated_vectors, prompt_seed_vectors = embed_records(generated_records, config_path)
    selection_result = select_seed_aware_records(
        records=generated_records,
        generated_vectors=generated_vectors,
        prompt_seed_vectors=prompt_seed_vectors,
        selector_cfg=dict(config["selector"]),
    )

    output_root = resolve_output_root(config_path)
    stage2_dir = write_selected_stage2_dir(
        [record.raw_text for record in selection_result.selected_records],
        output_dir=output_root,
    )
    metadata_path = stage2_dir / "selection_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "raw_generated_count": len(raw_outputs),
                "raw_clean_count": int(selection_result.raw_clean_count),
                "target_count": int(selection_result.target_count),
                "selected_records": _records_payload(selection_result.selected_records),
                "rejected_records": _records_payload(selection_result.rejected_records),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary["stage1"] = {
        "stage1_dir": str(stage1_runtime["stage1_dir"]),
        "seed_count": len(stage1_runtime["seed_texts"]),
    }
    summary["stage2"]["raw_generated_count"] = len(raw_outputs)
    summary["stage2"]["raw_clean_count"] = int(selection_result.raw_clean_count)
    summary["stage2"]["selected_count"] = len(selection_result.selected_records)
    summary["stage2"]["target_count"] = int(selection_result.target_count)
    summary["stage2"]["selected_stage2_dir"] = str(stage2_dir)
    summary["stage2"]["selection_metadata_path"] = str(metadata_path)
    if bool(config.get("pipeline", {}).get("run_eval", True)):
        summary["eval"] = run_eval_from_stage2_dir(config_path, stage2_dir=stage2_dir, output_dir=output_root / "eval")
    else:
        summary["eval"] = {"enabled": False, "mode": "disabled"}
    return summary
