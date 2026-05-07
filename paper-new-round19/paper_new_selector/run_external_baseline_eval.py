from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .external_baselines.common_eval import run_external_stage1_summary_eval
except ImportError:  # pragma: no cover - supports direct script execution
    import sys

    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from paper_new_selector.external_baselines.common_eval import (
        run_external_stage1_summary_eval,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    output_dir = str(args.output_dir).strip() or None
    result = run_external_stage1_summary_eval(
        summary_path=Path(args.summary_json),
        config_path=args.config,
        output_dir=output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
