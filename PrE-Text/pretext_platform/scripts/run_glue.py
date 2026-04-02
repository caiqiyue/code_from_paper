"""Run GLUE downstream evaluation on PrE-Text synthetic corpus.

Usage:
    # Validate local GLUE datasets before running
    python -m pretext_platform.scripts.run_glue --validate

    # Validate with custom dataset root
    python -m pretext_platform.scripts.run_glue --validate --dataset-root /path/to/datasets

    # Run single task
    python -m pretext_platform.scripts.run_glue --config configs/experiments/jobs_real_eps129.yaml --task sst2

    # Run multiple tasks
    python -m pretext_platform.scripts.run_glue --config configs/experiments/jobs_real_eps129.yaml --tasks sst2 qqp qnli

    # Run all supported tasks
    python -m pretext_platform.scripts.run_glue --config configs/experiments/jobs_real_eps129.yaml --tasks all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pretext_platform.core.config import load_experiment_config
from pretext_platform.core.pipeline import run_glue_eval
from pretext_platform.evaluation.glue_classification_eval import (
    print_glue_validation_report,
    validate_local_glue_datasets,
)


SUPPORTED_TASKS = ["sst2", "qqp", "qnli", "imdb", "rotten_tomatoes"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GLUE downstream evaluation on PrE-Text synthetic corpus."
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate local GLUE dataset availability and exit.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Custom dataset root path (default: from config or ../thesis_platform/datasets).",
    )
    parser.add_argument(
        "--config",
        required=False,
        help="Path to experiment YAML config (must have Stage2 output at stage2/llama7b_text_syn.json).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Single GLUE task to evaluate (sst2, qqp, qnli, imdb, rotten_tomatoes).",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Multiple GLUE tasks to evaluate.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate.",
    )

    args = parser.parse_args()

    # Handle --validate flag
    if args.validate:
        if args.dataset_root:
            dataset_root = Path(args.dataset_root)
        elif args.config:
            config = load_experiment_config(args.config)
            dataset_root = config.dataset_root()
        else:
            # Default path
            script_dir = Path(__file__).parent.parent.parent
            dataset_root = script_dir / "../thesis_platform/datasets"
            dataset_root = dataset_root.resolve()

        print_glue_validation_report(dataset_root)
        validation = validate_local_glue_datasets(dataset_root)
        sys.exit(0 if validation["all_required_available"] else 1)

    # Resolve task list
    if args.tasks and "all" in args.tasks:
        tasks = SUPPORTED_TASKS
    elif args.task:
        tasks = [args.task]
    elif args.tasks:
        tasks = args.tasks
    else:
        tasks = ["sst2"]

    # Validate tasks
    for task in tasks:
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task: {task}. Supported: {SUPPORTED_TASKS}")

    # Load config
    config = load_experiment_config(args.config)

    # Override eval_glue settings if provided
    if args.epochs is not None:
        config.raw.setdefault("eval_glue", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.raw.setdefault("eval_glue", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        config.raw.setdefault("eval_glue", {})["learning_rate"] = args.lr

    # Set eval_glue config
    config.raw.setdefault("eval_glue", {})["enabled"] = True
    config.raw["eval_glue"]["tasks"] = tasks

    # Create a dummy stage2 dir with symlink to actual stage2 output
    # This allows running GLUE eval independently after stage1+bootstrap complete
    experiment_dir = config.output_root() / config.experiment_id()
    stage2_dir = experiment_dir / "stage2"

    if not stage2_dir.exists():
        raise FileNotFoundError(
            f"Stage2 output not found at {stage2_dir}. "
            f"Please run stage1 and bootstrap first, or specify correct --config."
        )

    llama7b_path = stage2_dir / "llama7b_text_syn.json"
    if not llama7b_path.exists():
        raise FileNotFoundError(
            f"Synthetic corpus not found at {llama7b_path}. "
            f"Please ensure bootstrap stage completed successfully."
        )

    print(f"\n{'='*60}")
    print(f"Running GLUE Evaluation")
    print(f"Tasks: {tasks}")
    print(f"Stage2 output: {stage2_dir}")
    print(f"Output: {experiment_dir / 'eval_glue'}")
    print(f"{'='*60}\n")

    summaries = run_glue_eval(config)

    # Print results
    print(f"\n{'='*60}")
    print("GLUE Evaluation Results:")
    print(f"{'='*60}")
    for task in tasks:
        key = f"glue_{task}"
        if key in summaries:
            s = summaries[key]
            metrics = s.metrics
            print(f"\n{task.upper()}:")
            print(f"  Best Accuracy: {metrics.get('best_accuracy', 'N/A'):.4f}")
            print(f"  Correct: {metrics.get('correct', 'N/A')}/{metrics.get('total', 'N/A')}")
            print(f"  Data Source: {metrics.get('data_source', 'N/A')}")

    # Save summary
    output_file = experiment_dir / "eval_glue" / "glue_summary.json"
    results = {
        task: {
            "best_accuracy": summaries[f"glue_{task}"].metrics.get("best_accuracy", 0),
            "correct": summaries[f"glue_{task}"].metrics.get("correct", 0),
            "total": summaries[f"glue_{task}"].metrics.get("total", 0),
        }
        for task in tasks
        if f"glue_{task}" in summaries
    }
    with output_file.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
