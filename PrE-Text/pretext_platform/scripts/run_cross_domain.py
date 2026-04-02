"""Cross-domain transfer runner for PrE-Text.

This script now completes the target-domain evaluation instead of stopping after
synthetic corpus generation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pretext_platform.core.config import load_experiment_config
from pretext_platform.core.pipeline import run_pipeline


DEFAULT_TARGET_CONFIGS = {
    "jobs": "configs/experiments/jobs_real_eps129.yaml",
    "forums": "configs/experiments/forums_real_eps129.yaml",
    "microblog": "configs/experiments/microblog_real_eps129.yaml",
}


def _disable_sections(config, *section_names: str) -> None:
    for section_name in section_names:
        config.raw.setdefault(section_name, {})["enabled"] = False


def _ensure_stage2_corpus(target_experiment_dir: Path, synthetic_corpus_path: Path) -> Path:
    stage2_dir = target_experiment_dir / "stage2"
    stage2_dir.mkdir(parents=True, exist_ok=True)
    target_path = stage2_dir / "llama7b_text_syn.json"
    shutil.copy2(synthetic_corpus_path, target_path)
    return target_path


def run_stage1_only(config_path: str, output_root: Path) -> dict:
    config = load_experiment_config(config_path)
    base_id = config.raw.setdefault("meta", {}).get("experiment_id", Path(config_path).stem)
    config.raw["meta"]["experiment_id"] = f"{base_id}_stage1"
    config.raw.setdefault("paths", {})["output_root"] = str(output_root)
    _disable_sections(config, "bootstrap", "eval_small", "eval_large", "eval_glue")
    return run_pipeline(config)


def run_bootstrap_only(config_path: str, output_root: Path, experiment_id: str) -> dict:
    config = load_experiment_config(config_path)
    config.raw.setdefault("meta", {})["experiment_id"] = experiment_id
    config.raw.setdefault("paths", {})["output_root"] = str(output_root)
    config.raw.setdefault("stage1", {})["enabled"] = False
    _disable_sections(config, "eval_small", "eval_large", "eval_glue")
    return run_pipeline(config)


def run_target_evaluation(
    target_config_path: str,
    output_root: Path,
    experiment_id: str,
    synthetic_corpus_path: Path,
) -> dict:
    config = load_experiment_config(target_config_path)
    config.raw.setdefault("meta", {})["experiment_id"] = experiment_id
    config.raw.setdefault("paths", {})["output_root"] = str(output_root)
    config.raw.setdefault("stage1", {})["enabled"] = False
    config.raw.setdefault("bootstrap", {})["enabled"] = False
    config.raw.setdefault("eval_large", {})["enabled"] = True
    if "eval_small" in config.raw:
        config.raw["eval_small"]["enabled"] = False
    if "eval_glue" in config.raw:
        config.raw["eval_glue"]["enabled"] = False

    target_experiment_dir = Path(config.raw["paths"]["output_root"]).resolve() / experiment_id
    staged_corpus_path = _ensure_stage2_corpus(target_experiment_dir, synthetic_corpus_path)
    result = run_pipeline(config)
    result["synthetic_corpus_path"] = str(staged_corpus_path)
    return result


def run_cross_domain_transfer(
    source_config_path: str,
    target_dataset: str,
    target_eval_config_path: str | None = None,
    output_base: str | None = None,
) -> dict:
    source_config = load_experiment_config(source_config_path)
    source_dataset_name = source_config.raw.get("data", {}).get("dataset_name", Path(source_config_path).stem)

    if target_eval_config_path is None:
        target_eval_config_path = DEFAULT_TARGET_CONFIGS[target_dataset]

    if output_base is None:
        output_root = source_config.raw.get("paths", {}).get("output_root", "./outputs/pretext_platform")
        output_base_path = Path(output_root) / "cross_domain" / f"{source_dataset_name}_to_{target_dataset}"
    else:
        output_base_path = Path(output_base)
    output_base_path = output_base_path.resolve()
    output_base_path.mkdir(parents=True, exist_ok=True)

    source_stage1_root = output_base_path / "source_stage1_outputs"
    source_bootstrap_root = output_base_path / "source_bootstrap_outputs"
    target_eval_root = output_base_path / "target_eval_outputs"

    print(f"\n{'=' * 60}")
    print("Cross-Domain Transfer Experiment")
    print(f"Source dataset: {source_dataset_name}")
    print(f"Target dataset: {target_dataset}")
    print(f"Output base: {output_base_path}")
    print(f"{'=' * 60}\n")

    stage1_start = time.time()
    stage1_result = run_stage1_only(source_config_path, source_stage1_root)
    stage1_elapsed = time.time() - stage1_start

    bootstrap_experiment_id = f"{source_dataset_name}_to_{target_dataset}_bootstrap"
    bootstrap_start = time.time()
    bootstrap_result = run_bootstrap_only(source_config_path, source_bootstrap_root, bootstrap_experiment_id)
    bootstrap_elapsed = time.time() - bootstrap_start

    synthetic_corpus_path = (
        Path(bootstrap_result["experiment_dir"]).resolve() / "stage2" / "llama7b_text_syn.json"
    )
    if not synthetic_corpus_path.exists():
        raise FileNotFoundError(f"Synthetic corpus not found at {synthetic_corpus_path}")

    target_experiment_id = f"{source_dataset_name}_to_{target_dataset}_eval"
    eval_start = time.time()
    target_eval_result = run_target_evaluation(
        target_eval_config_path,
        target_eval_root,
        target_experiment_id,
        synthetic_corpus_path,
    )
    eval_elapsed = time.time() - eval_start

    summary = {
        "experiment_id": f"{source_dataset_name}_to_{target_dataset}",
        "source_dataset": source_dataset_name,
        "target_dataset": target_dataset,
        "source_config": source_config_path,
        "target_eval_config": target_eval_config_path,
        "output_directory": str(output_base_path),
        "synthetic_corpus_path": str(synthetic_corpus_path),
        "stages": {
            "stage1": {
                "status": "completed",
                "elapsed_hours": stage1_elapsed / 3600,
                "experiment_dir": stage1_result["experiment_dir"],
            },
            "bootstrap": {
                "status": "completed",
                "elapsed_hours": bootstrap_elapsed / 3600,
                "experiment_dir": bootstrap_result["experiment_dir"],
            },
            "target_eval": {
                "status": "completed",
                "elapsed_hours": eval_elapsed / 3600,
                "experiment_dir": target_eval_result["experiment_dir"],
                "metrics": target_eval_result["stages"].get("eval_large", {}),
            },
        },
        "total_elapsed_hours": (stage1_elapsed + bootstrap_elapsed + eval_elapsed) / 3600,
    }

    summary_path = output_base_path / "cross_domain_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary saved to: {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cross-domain transfer experiments for PrE-Text.")
    parser.add_argument("--source-config", required=True, help="Source experiment config path.")
    parser.add_argument(
        "--target-dataset",
        required=True,
        choices=sorted(DEFAULT_TARGET_CONFIGS),
        help="Target dataset name.",
    )
    parser.add_argument(
        "--target-eval-config",
        default=None,
        help="Optional target evaluation config path. Defaults to the ε=1.29 target config.",
    )
    parser.add_argument("--output-base", type=str, default=None, help="Base output directory.")
    args = parser.parse_args()

    summary = run_cross_domain_transfer(
        args.source_config,
        args.target_dataset,
        args.target_eval_config,
        args.output_base,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
