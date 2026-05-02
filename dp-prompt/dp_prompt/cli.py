from __future__ import annotations

import argparse

from dp_prompt.config import load_experiment_config
from dp_prompt.runners.document_pipeline import run_document_pipeline
from dp_prompt.runners.pretext_pipeline import run_pretext_style_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a DP-Prompt experiment.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    pipeline_mode = str(config.get("experiment", {}).get("pipeline_mode", "document")).strip().lower()
    if pipeline_mode == "pretext_style":
        run_pretext_style_pipeline(config)
    else:
        run_document_pipeline(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
