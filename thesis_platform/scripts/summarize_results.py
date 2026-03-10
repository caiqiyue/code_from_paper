from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize experiment result JSON files.")
    parser.add_argument("--input", required=True, help="Directory containing experiment outputs.")
    args = parser.parse_args()

    root = Path(args.input).resolve()
    rows = []
    for summary_path in sorted(root.rglob("metrics_summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "experiment_id": payload.get("experiment_id"),
                "round_count": payload.get("round_count"),
                "final_prompt_length": len(str(payload.get("final_prompt", "")).split()),
            }
        )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
