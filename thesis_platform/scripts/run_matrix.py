from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_platform.core.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every experiment config under a directory.")
    parser.add_argument("--config_dir", required=True, help="Directory containing experiment YAML files.")
    args = parser.parse_args()
    config_dir = Path(args.config_dir).resolve()
    results = []
    for path in sorted(config_dir.rglob("*.yaml")):
        results.append(run_pipeline(str(path)))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
