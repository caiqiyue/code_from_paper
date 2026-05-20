from __future__ import annotations

from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_new_selector.thesis_e1_main_runner import run_e1_batch  # noqa: E402


def main() -> int:
    result = run_e1_batch(ROOT, mode="thesis_main_seen_smoke", dry_run=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
