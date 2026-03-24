from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from pretext_platform.core.config import load_experiment_config


def _convert_paths(obj):
    """Recursively convert Path objects to strings for JSON serialization."""
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, dict)):
        return [_convert_paths(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _convert_paths(v) for k, v in obj.items()}
    elif hasattr(obj, "__fspath__"):
        return str(obj)
    return obj


def main() -> None:
    """Run only the small-model downstream evaluation (GPT-2 or DistilGPT2).

    If c4_checkpoint.pth exists, uses DistilGPT2 with warm-start.
    Otherwise, uses GPT-2 directly from HuggingFace (no checkpoint needed).
    """

    from pretext_platform.core.models import resolve_model_paths
    from pretext_platform.data.loaders import load_dataset_bundle
    from pretext_platform.evaluation.distilgpt2_eval import run_distilgpt2_eval
    from pretext_platform.evaluation.gpt2_eval import run_gpt2_eval

    parser = argparse.ArgumentParser(
        description="Run small-model evaluation (GPT-2 or DistilGPT2)."
    )
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    model_paths = resolve_model_paths(config)
    dataset_bundle = load_dataset_bundle(config)

    # Determine which evaluation function to use
    if model_paths.c4_checkpoint is not None and model_paths.c4_checkpoint.is_file():
        # Use DistilGPT2 with warm-start checkpoint
        from pretext_platform.core.pipeline import _experiment_dir
        from pretext_platform.core.io_utils import ensure_dir

        experiment_dir = _experiment_dir(config)
        stage2_dir = experiment_dir / "stage2"
        output_dir = experiment_dir / "eval_small"
        summary = run_distilgpt2_eval(config, dataset_bundle, model_paths, stage2_dir, output_dir)
    else:
        # Use GPT-2 without checkpoint (Windows compatible)
        print("c4_checkpoint.pth not found, using GPT-2 from HuggingFace instead")
        from pretext_platform.core.pipeline import _experiment_dir
        from pretext_platform.core.io_utils import ensure_dir

        experiment_dir = _experiment_dir(config)
        stage2_dir = experiment_dir / "stage2"
        output_dir = ensure_dir(experiment_dir / "eval_small")
        summary = run_gpt2_eval(config, dataset_bundle, model_paths, stage2_dir, output_dir)

    summary_dict = _convert_paths(asdict(summary))
    print(json.dumps(summary_dict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
