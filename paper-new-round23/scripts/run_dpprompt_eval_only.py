#!/usr/bin/env python3
"""Standalone GPT2 eval runner for dp-prompt experiments.

Called by run_e1_dpprompt_baseline.py AFTER the vLLM generation subprocess has
fully exited. Running eval in a fresh subprocess guarantees that vLLM's raw
cuMalloc GPU memory is fully reclaimed by the OS before GPT2 loads.

Usage:
    python run_dpprompt_eval_only.py \
        --effective-config <path> \
        --pipeline-output-dir <path> \
        --experiment-id <id>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROUND23_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROUND23_ROOT.parent
DPPROMPT_ROOT = REPO_ROOT / "dp-prompt"


def _ensure_paths() -> None:
    for p in [str(DPPROMPT_ROOT), str(REPO_ROOT)]:
        if p not in sys.path:
            sys.path.insert(0, p)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GPT2 eval for a dp-prompt experiment")
    parser.add_argument("--effective-config", required=True, help="Path to materialized effective config YAML")
    parser.add_argument("--pipeline-output-dir", required=True, help="dp-prompt pipeline output directory")
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()

    _ensure_paths()

    import yaml
    from dp_prompt.pretext_bridge import (
        build_thesis_eval_config,
        ensure_thesis_platform_importable,
        export_and_run_small_eval,
        resolve_repo_root,
    )
    from dp_prompt.utils.io import ensure_dir, write_json

    ensure_thesis_platform_importable(resolve_repo_root())

    effective_config_path = Path(args.effective_config)
    pipeline_output_dir = Path(args.pipeline_output_dir)
    experiment_id = args.experiment_id

    config = yaml.safe_load(effective_config_path.read_text(encoding="utf-8"))
    data_cfg = config.get("data", {}) or {}

    corpus_path = pipeline_output_dir / "sanitized_corpus.json"
    if not corpus_path.exists():
        print(f"[eval_only] ERROR: corpus not found at {corpus_path}", file=sys.stderr)
        return 1
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    synthetic_texts = [str(r.get("sanitized_text", "")) for r in corpus]

    stage2_dir = ensure_dir(pipeline_output_dir / "stage2")
    eval_dir = ensure_dir(pipeline_output_dir / "eval")

    thesis_config = build_thesis_eval_config(
        experiment_id=experiment_id,
        dataset_name=str(data_cfg.get("dataset_name", "unknown")),
        train_path=str(data_cfg["train_path"]),
        eval_path=str(data_cfg["eval_path"]),
        initialization_path=str(data_cfg["initialization_path"]),
        output_root=str(pipeline_output_dir),
        train_limit=data_cfg.get("train_limit"),
        eval_limit=data_cfg.get("eval_limit"),
        initialization_limit=data_cfg.get("initialization_limit"),
        max_samples_per_client=int(data_cfg.get("max_samples_per_client", 8)),
        linux_small_eval_mode=str(
            config.get("evaluation", {}).get("linux_small_eval_mode", "gpt2")
        ),
    )

    print(f"[eval_only] Running GPT2 eval for {experiment_id} ({len(synthetic_texts)} synthetic texts)")
    eval_summary = export_and_run_small_eval(
        thesis_config,
        synthetic_texts=synthetic_texts,
        stage2_dir=stage2_dir,
        eval_dir=eval_dir,
    )
    write_json(eval_dir / "eval_small_summary.json", eval_summary)
    print(f"[eval_only] Done. best_top1={eval_summary.get('best_top1')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
