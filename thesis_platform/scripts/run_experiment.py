from __future__ import annotations

import argparse
import json

from thesis_platform.core.pipeline import run_pipeline


def main() -> None:
    """Run one experiment config and print the summary JSON to stdout."""

    parser = argparse.ArgumentParser(description="Run a single thesis platform experiment.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    args = parser.parse_args()
    summary = run_pipeline(args.config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
