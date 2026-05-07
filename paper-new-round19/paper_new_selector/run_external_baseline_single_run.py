from __future__ import annotations

import argparse
import json

from .external_baselines.single_run_runner import run_external_single_run_from_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a config-driven external baseline single-run screening experiment."
    )
    parser.add_argument("--config", required=True, help="Path to the external baseline YAML config.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Build stage1 summary from the config without launching downstream eval.",
    )
    args = parser.parse_args()

    result = run_external_single_run_from_config(
        args.config,
        validate_only=args.validate_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
