#!/usr/bin/env python3
"""Run a single E1 DP-Prompt baseline experiment.

Invokes the existing dp-prompt pretext-style pipeline (which paraphrases each
private document via a vLLM-served LLM with temperature-based randomization),
then writes a runner-compatible sidecar JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from round23_runtime_utils import load_yaml_with_inherits


ROUND23_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROUND23_ROOT.parent
DPPROMPT_ROOT = REPO_ROOT / "dp-prompt"


def _build_dpprompt_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_entries = [str(DPPROMPT_ROOT), str(REPO_ROOT)]
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    if env.get("CUDA_VISIBLE_DEVICES") and not env.get("CUDA_DEVICE_ORDER"):
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return env


def _materialize_runtime_config(
    *,
    source_config_path: Path,
    output_root: Path,
    experiment_id: str,
) -> tuple[Path, Path]:
    """Render an effective YAML config and return (effective_config_path, pipeline_output_dir).

    The dp-prompt pipeline writes to ``runtime.output_root / experiment.id``. We
    set ``runtime.output_root`` to ``output_root.parent`` so the actual pipeline
    output directory is ``output_root.parent / experiment_id``. Both the effective
    config path and that pipeline output directory are returned so callers can read
    artifacts from the correct location.
    """
    merged = load_yaml_with_inherits(source_config_path)
    # Force experiment.id to match what the runner expects so the dp-prompt
    # pipeline writes into output_root.parent / experiment_id predictably.
    merged.setdefault("experiment", {})
    merged["experiment"]["id"] = experiment_id
    merged.setdefault("runtime", {})
    # dp-prompt pipeline creates output_dir = runtime.output_root / experiment.id
    merged["runtime"]["output_root"] = str(output_root.parent.resolve())
    output_root.mkdir(parents=True, exist_ok=True)
    effective_path = output_root / f"{experiment_id}_dpprompt_effective_config.yaml"
    # Drop _meta (added by dp-prompt config loader) to avoid noise in serialized form
    merged.pop("_meta", None)
    with effective_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(merged, handle, sort_keys=False)
    # pipeline_output_dir is where the dp-prompt pipeline actually writes its artifacts
    pipeline_output_dir = output_root.parent.resolve() / experiment_id
    return effective_path, pipeline_output_dir


def _run_dpprompt_pipeline(config_path: Path, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "dp_prompt.cli", "--config", str(config_path)],
        capture_output=True,
        text=True,
        cwd=str(DPPROMPT_ROOT),
        env=_build_dpprompt_subprocess_env(),
        timeout=timeout_seconds,
    )


def _collect_dpprompt_artifacts(pipeline_output_dir: Path) -> dict[str, Any]:
    eval_summary_path = pipeline_output_dir / "eval" / "eval_small_summary.json"
    experiment_summary_path = pipeline_output_dir / "experiment_summary.json"
    if not eval_summary_path.exists():
        raise FileNotFoundError(
            f"Expected DP-Prompt eval summary at {eval_summary_path}. Did the dp-prompt pipeline run?"
        )
    eval_payload = json.loads(eval_summary_path.read_text(encoding="utf-8"))
    artifacts: dict[str, Any] = {
        "runtime_output_root": str(pipeline_output_dir.resolve()),
        "eval_summary_path": str(eval_summary_path.resolve()),
        "eval_summary": eval_payload,
    }
    if experiment_summary_path.exists():
        artifacts["experiment_summary_path"] = str(experiment_summary_path.resolve())
    sanitized_corpus = pipeline_output_dir / "sanitized_corpus.json"
    if sanitized_corpus.exists():
        artifacts["sanitized_corpus_path"] = str(sanitized_corpus.resolve())
    return artifacts


def _write_runtime_sidecar(
    *,
    output_root: Path,
    experiment_id: str,
    dataset_name: str,
    effective_config_path: Path,
    runtime_artifacts: dict[str, Any],
    config_payload: dict[str, Any],
) -> Path:
    privacy_cfg = dict(config_payload.get("privacy", {}) or {})
    generation_cfg = dict(config_payload.get("generation", {}) or {})
    synthetic_count = 0
    sanitized_path = Path(runtime_artifacts.get("sanitized_corpus_path", ""))
    if sanitized_path.exists():
        try:
            with sanitized_path.open("r", encoding="utf-8") as handle:
                synthetic_count = len(json.load(handle))
        except Exception:
            synthetic_count = 0
    sidecar = {
        "budget_policy_type": "dpprompt",
        "dataset_name": dataset_name,
        "experiment_id": experiment_id,
        "synthetic_count": synthetic_count,
        "privacy": {
            "temperature": privacy_cfg.get("temperature"),
            "logits_clipping": privacy_cfg.get("logits_clipping"),
            "max_generated_tokens": privacy_cfg.get("max_generated_tokens"),
        },
        "generation": generation_cfg,
        "effective_config_path": str(effective_config_path.resolve()),
        "runtime_artifacts": runtime_artifacts,
    }
    sidecar_path = output_root / f"{experiment_id}_dpprompt_runtime.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")
    return sidecar_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a single E1 DP-Prompt baseline experiment")
    parser.add_argument("--config", required=True, help="Path to dp-prompt experiment YAML")
    parser.add_argument("--output-root", required=True, help="Output root directory for this run (absolute path)")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument(
        "--reference-budget",
        type=int,
        default=20,
        help="Accepted for runner compatibility; DP-Prompt does not use a budget knob.",
    )
    args = parser.parse_args()

    source_config = Path(args.config)
    if not source_config.is_absolute():
        source_config = (ROUND23_ROOT / source_config).resolve()
    raw_cfg = load_yaml_with_inherits(source_config)
    experiment_cfg = dict(raw_cfg.get("experiment", {}) or {})
    data_cfg = dict(raw_cfg.get("data", {}) or {})
    experiment_id = str(experiment_cfg.get("id", source_config.stem))
    dataset_name = str(data_cfg.get("dataset_name", "unknown"))

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[e1_dpprompt] experiment_id={experiment_id} dataset={dataset_name} output_root={output_root}")

    effective_config, pipeline_output_dir = _materialize_runtime_config(
        source_config_path=source_config,
        output_root=output_root,
        experiment_id=experiment_id,
    )
    print(f"[e1_dpprompt] pipeline_output_dir={pipeline_output_dir}")

    print(f"[e1_dpprompt] Invoking dp-prompt pipeline with effective config: {effective_config}")
    result = _run_dpprompt_pipeline(effective_config, timeout_seconds=args.timeout_seconds)
    print(result.stdout, end="")
    if result.returncode != 0:
        print(f"[ERROR] dp-prompt pipeline failed:\n{result.stderr}", file=sys.stderr)
        return int(result.returncode)

    runtime_artifacts = _collect_dpprompt_artifacts(pipeline_output_dir)
    sidecar_path = _write_runtime_sidecar(
        output_root=output_root,
        experiment_id=experiment_id,
        dataset_name=dataset_name,
        effective_config_path=effective_config,
        runtime_artifacts=runtime_artifacts,
        config_payload=raw_cfg,
    )
    print(f"[e1_dpprompt] Done. Sidecar: {sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
