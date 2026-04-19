from __future__ import annotations

import argparse
import json

from pretext_platform.core.pipeline import run_pipeline


def main() -> None:
    """Run one federated PrE-Text experiment from a YAML config."""

    parser = argparse.ArgumentParser(description="Run one federated PrE-Text experiment.")
    parser.add_argument("--config", required=True, help="Path to a federated PrE-Text YAML config.")
    args = parser.parse_args()

    summary = run_pipeline(args.config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
