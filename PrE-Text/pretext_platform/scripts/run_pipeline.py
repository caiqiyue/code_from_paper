from __future__ import annotations

import argparse
import json
import sys

from pretext_platform.core.pipeline import run_pipeline
from pretext_platform.core.preflight import format_preflight_report, run_preflight


def main() -> None:
    """Run all enabled stages for one experiment config."""

    parser = argparse.ArgumentParser(description="Run the full PrE-Text pipeline from one config.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run preflight validation only and exit without executing any stage.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip preflight validation and run stages directly.",
    )
    args = parser.parse_args()

    if not args.skip_preflight or args.validate_only:
        report = run_preflight(args.config)
        if args.validate_only:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
            if report.warnings:
                print(format_preflight_report(report), file=sys.stderr)
            sys.exit(0 if report.ready else 1)
        if not report.ready:
            print(format_preflight_report(report), file=sys.stderr)
            sys.exit(1)
        if report.warnings:
            print(format_preflight_report(report), file=sys.stderr)

    summary = run_pipeline(args.config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
