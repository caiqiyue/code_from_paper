from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from pretext_platform.core.config import load_experiment_config
from pretext_platform.core.pipeline import run_stage1


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
    """Run only Stage 1 from one experiment config."""

    parser = argparse.ArgumentParser(description="Run Stage 1 Private Evolution from one config.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    args = parser.parse_args()
    summary = run_stage1(load_experiment_config(args.config))
    summary_dict = _convert_paths(asdict(summary))
    print(json.dumps(summary_dict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
