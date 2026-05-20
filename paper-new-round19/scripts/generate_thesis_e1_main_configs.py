from __future__ import annotations

from pathlib import Path
import argparse
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_new_selector.thesis_e1_main_runner import materialize_e1_configs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate thesis E1 baseline configs and manifest")
    parser.add_argument("--mode", default="thesis_main_seen_pilot")
    args = parser.parse_args()
    generated = materialize_e1_configs(ROOT, mode=args.mode)
    print(f"generated_configs={len(generated)}")
    if generated:
        print(f"first_config={generated[0]}")
        print(f"last_config={generated[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
