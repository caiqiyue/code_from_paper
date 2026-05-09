from __future__ import annotations

import argparse
import json
from pathlib import Path

from .eval_bridge import run_eval


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--synthetic-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    synthetic_path = Path(args.synthetic_path)
    synthetic_texts = json.loads(synthetic_path.read_text(encoding="utf-8"))
    summary = run_eval(
        synthetic_texts=list(synthetic_texts),
        config_path=args.config,
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
