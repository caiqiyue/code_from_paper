from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from pretext_platform.core.config import load_experiment_config
from pretext_platform.core.io_utils import write_json


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

    The execution mode follows `eval_small.eval_mode` in the resolved config:
    - `gpt2`: always use base GPT-2 from HuggingFace
    - `distilgpt2`: require `c4_checkpoint.pth`, otherwise fail fast
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

    eval_mode = str(config.eval_small.get("eval_mode", "gpt2")).strip().lower()
    from pretext_platform.core.pipeline import _experiment_dir
    from pretext_platform.core.io_utils import ensure_dir

    experiment_dir = _experiment_dir(config)
    stage2_dir = experiment_dir / "stage2"
    output_dir = ensure_dir(experiment_dir / "eval_small")

    if eval_mode == "distilgpt2":
        if model_paths.c4_checkpoint is None or not model_paths.c4_checkpoint.is_file():
            raise FileNotFoundError(
                "eval_small.eval_mode=distilgpt2 requires c4_checkpoint.pth, but no checkpoint was found."
            )
        summary = run_distilgpt2_eval(config, dataset_bundle, model_paths, stage2_dir, output_dir)
    else:
        if eval_mode != "gpt2":
            print(f"Unknown eval_small.eval_mode={eval_mode!r}, falling back to GPT-2.")
        summary = run_gpt2_eval(config, dataset_bundle, model_paths, stage2_dir, output_dir)

    write_json(experiment_dir / "eval_small_summary.json", summary)
    write_json(experiment_dir / "resolved_config.json", config.raw)

    summary_dict = _convert_paths(asdict(summary))
    print(json.dumps(summary_dict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
