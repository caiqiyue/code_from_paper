from __future__ import annotations

import argparse
import json
import sys

from .pipeline import run_pipeline


def _force_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    _force_utf8_stdio()
    parser = argparse.ArgumentParser(description="Run the paper-new-2 seed-aware Stage 2 selector pipeline.")
    parser.add_argument("--config", required=True, help="Path to the YAML experiment config.")
    parser.add_argument("--validate-only", action="store_true", help="Validate config and resolved runtime contracts only.")
    args = parser.parse_args()
    summary = run_pipeline(args.config, validate_only=args.validate_only)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
