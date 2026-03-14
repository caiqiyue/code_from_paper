from __future__ import annotations

import argparse
import json

from pretext_platform.core.config import load_experiment_config
from pretext_platform.core.pipeline import run_stage1


def main() -> None:
    """Run only Stage 1 from one experiment config."""

    parser = argparse.ArgumentParser(description="Run Stage 1 Private Evolution from one config.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    args = parser.parse_args()
    summary = run_stage1(load_experiment_config(args.config))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
