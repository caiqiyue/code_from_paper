from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from thesis_platform.core.pipeline import run_pipeline
from thesis_platform.core.io_utils import write_json


def main() -> None:
    """Run every experiment config under one directory tree."""

    parser = argparse.ArgumentParser(description="Run every experiment config under a directory.")
    parser.add_argument("--config_dir", required=True, help="Directory containing experiment YAML files.")
    parser.add_argument("--resume", action="store_true", help="Pass --resume through to each experiment run.")
    parser.add_argument("--stop_on_error", action="store_true", help="Stop the batch immediately when one config fails.")
    parser.add_argument("--summary_path", default="", help="Optional JSON path for incremental matrix results.")
    args = parser.parse_args()
    config_dir = Path(args.config_dir).resolve()
    results = []
    for path in sorted(config_dir.rglob("*.yaml")):
        try:
            summary = run_pipeline(str(path), resume=args.resume)
            results.append(
                {
                    "config_path": str(path),
                    "status": "completed",
                    "summary": summary,
                }
            )
        except Exception as exc:
            failure = {
                "config_path": str(path),
                "status": "failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            results.append(failure)
            if args.summary_path:
                write_json(Path(args.summary_path).resolve(), {"results": results})
            if args.stop_on_error:
                raise
        if args.summary_path:
            write_json(Path(args.summary_path).resolve(), {"results": results})
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
