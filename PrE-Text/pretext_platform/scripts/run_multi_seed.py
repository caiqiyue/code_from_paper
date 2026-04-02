"""Run PrE-Text experiments with multiple seeds for statistical significance.

Usage:
    python -m pretext_platform.scripts.run_multi_seed --config configs/experiments/jobs_real_eps129.yaml --seeds 42 123 456
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pretext_platform.core.config import load_experiment_config
from pretext_platform.core.pipeline import run_pipeline


def _apply_stage_selection(config, stage: str) -> None:
    if stage == "all":
        return
    if stage == "stage1":
        config.raw.setdefault("stage1", {})["enabled"] = True
        config.raw.setdefault("bootstrap", {})["enabled"] = False
        config.raw.setdefault("eval_small", {})["enabled"] = False
        config.raw.setdefault("eval_large", {})["enabled"] = False
        if "eval_glue" in config.raw:
            config.raw["eval_glue"]["enabled"] = False
        return
    if stage == "bootstrap":
        config.raw.setdefault("stage1", {})["enabled"] = False
        config.raw.setdefault("bootstrap", {})["enabled"] = True
        config.raw.setdefault("eval_small", {})["enabled"] = False
        config.raw.setdefault("eval_large", {})["enabled"] = False
        if "eval_glue" in config.raw:
            config.raw["eval_glue"]["enabled"] = False
        return
    if stage == "eval":
        config.raw.setdefault("stage1", {})["enabled"] = False
        config.raw.setdefault("bootstrap", {})["enabled"] = False
        config.raw.setdefault("eval_small", {})["enabled"] = False
        config.raw.setdefault("eval_large", {})["enabled"] = True
        if "eval_glue" in config.raw:
            config.raw["eval_glue"]["enabled"] = False


def run_with_seed(config_path: str, seed: int, output_base: Path, *, stage: str) -> dict:
    """Run one experiment with a specific seed."""
    config = load_experiment_config(config_path)

    # Override the seed in the config's meta section
    config.raw["meta"]["seed"] = seed
    # Update experiment_id to include seed
    base_id = config.raw["meta"].get("experiment_id", Path(config_path).stem)
    config.raw["meta"]["experiment_id"] = f"{base_id}_seed{seed}"

    # Create output directory for this seed
    seed_output_dir = output_base / f"seed{seed}"
    config.raw["paths"]["output_root"] = str(seed_output_dir)
    _apply_stage_selection(config, stage)

    print(f"\n{'='*60}")
    print(f"Running with seed={seed}")
    print(f"Stage selection: {stage}")
    print(f"Output directory: {seed_output_dir}")
    print(f"{'='*60}\n")

    start_time = time.time()
    try:
        result = run_pipeline(config)
        elapsed = time.time() - start_time
        print(f"\nSeed {seed} completed in {elapsed/3600:.2f} hours")
        return {
            "seed": seed,
            "status": "success",
            "elapsed_hours": elapsed / 3600,
            "experiment_dir": result.get("experiment_dir", str(seed_output_dir / config.raw["meta"]["experiment_id"])),
            "result": result,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\nSeed {seed} FAILED after {elapsed/3600:.2f} hours: {e}")
        return {
            "seed": seed,
            "status": "failed",
            "elapsed_hours": elapsed / 3600,
            "error": str(e),
        }


def run_multi_seed(
    config_path: str,
    seeds: list[int],
    output_base: Path | None = None,
    *,
    stage: str = "all",
) -> dict:
    """Run experiments with multiple seeds sequentially.

    Args:
        config_path: Path to experiment YAML config
        seeds: List of random seeds to use
        output_base: Base output directory (default: ./outputs/multi_seed/)

    Returns:
        Summary dict with results for each seed
    """
    config = load_experiment_config(config_path)
    base_experiment_id = config.raw["meta"].get("experiment_id", Path(config_path).stem)

    if output_base is None:
        output_base = Path(config.raw.get("paths", {}).get("output_root", "./outputs"))
        output_base = output_base / "multi_seed" / base_experiment_id

    output_base = output_base.resolve()
    output_base.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in seeds:
        result = run_with_seed(config_path, seed, output_base, stage=stage)
        results.append(result)

        # Save intermediate results after each seed
        with (output_base / "multi_seed_results.json").open("w") as f:
            json.dump(
                {
                    "base_experiment_id": base_experiment_id,
                    "seeds_run": seeds,
                    "completed": len([r for r in results if r["status"] == "success"]),
                    "failed": len([r for r in results if r["status"] == "failed"]),
                    "results": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    # Compute statistics
    successful_results = [r for r in results if r["status"] == "success"]
    if successful_results:
        # Extract metrics from successful runs
        # Note: Actual metric extraction would depend on the experiment output format
        summary = {
            "base_experiment_id": base_experiment_id,
            "seeds": seeds,
            "stage": stage,
            "successful_runs": len(successful_results),
            "failed_runs": len(results) - len(successful_results),
            "total_time_hours": sum(r.get("elapsed_hours", 0) for r in results),
            "results": results,
        }
    else:
        summary = {
            "base_experiment_id": base_experiment_id,
            "seeds": seeds,
            "stage": stage,
            "successful_runs": 0,
            "failed_runs": len(results),
            "results": results,
        }

    # Save final summary
    with (output_base / "multi_seed_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Multi-seed run completed!")
    print(f"Base experiment: {base_experiment_id}")
    print(f"Seeds: {seeds}")
    print(f"Successful: {summary['successful_runs']}/{len(results)}")
    print(f"Total time: {summary['total_time_hours']:.2f} hours")
    print(f"Output directory: {output_base}")
    print(f"{'='*60}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PrE-Text experiments with multiple seeds for statistical significance."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML config.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="List of random seeds to use (default: 42 123 456).",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default=None,
        help="Base output directory for all seed runs.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["stage1", "bootstrap", "eval", "all"],
        default="all",
        help="Which stage to run (default: all).",
    )

    args = parser.parse_args()

    output_base = Path(args.output_base) if args.output_base else None

    summary = run_multi_seed(args.config, args.seeds, output_base, stage=args.stage)

    # Print summary
    print("\nSummary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
