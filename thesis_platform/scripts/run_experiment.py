from __future__ import annotations

import argparse
import json

from thesis_platform.core.pipeline import run_pipeline


def main() -> None:
    """Run one experiment config and print the summary JSON to stdout."""

    parser = argparse.ArgumentParser(description="Run a single thesis platform experiment.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint or an explicit resume_dir when available.")
    parser.add_argument("--resume_dir", help="Explicit experiment output directory to resume.")
    args = parser.parse_args()
    summary = run_pipeline(args.config, resume=args.resume, resume_dir=args.resume_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
