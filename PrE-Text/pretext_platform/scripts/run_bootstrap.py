from __future__ import annotations

import argparse
import json

from pretext_platform.core.config import load_experiment_config
from pretext_platform.core.pipeline import run_bootstrap


def main() -> None:
    """Run only Stage 2 bootstrap from one experiment config."""

    parser = argparse.ArgumentParser(description="Run Stage 2 bootstrap from one config.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    args = parser.parse_args()
    summary = run_bootstrap(load_experiment_config(args.config))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
