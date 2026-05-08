from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_new_selector.repeat10_baseline_runner import materialize_repeat10_configs


def main() -> None:
    generated = materialize_repeat10_configs(ROOT)
    print(f"generated_configs={len(generated)}")
    if generated:
        print(f"first_config={generated[0]}")
        print(f"last_config={generated[-1]}")


if __name__ == "__main__":
    main()
