from __future__ import annotations

import argparse
import json

from paper_new_selector.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the paper-new single-node selector pipeline.")
    parser.add_argument("--config", required=True, help="Path to the selector config YAML.")
    parser.add_argument("--validate-only", action="store_true", help="Resolve contracts without heavy execution.")
    args = parser.parse_args()

    summary = run_pipeline(args.config, validate_only=args.validate_only)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
