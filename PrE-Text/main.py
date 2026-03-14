"""Legacy Stage 1 entry point kept as a thin wrapper over pretext_platform."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pretext_platform.core.legacy import build_legacy_config
from pretext_platform.core.pipeline import run_stage1


def parse_args() -> argparse.Namespace:
    """Parse the original Stage 1 CLI arguments."""

    parser = argparse.ArgumentParser("Run PE-text.")
    parser.add_argument("-datadir", type=str, default="", help="Dataset name prefix for the train/eval JSON files.")
    parser.add_argument("-outputdir", type=str, required=True, help="Base directory where experiment outputs are written.")
    parser.add_argument("-cachedir", type=str, required=False, help="Deprecated legacy argument kept for compatibility.")
    parser.add_argument("-sensitivity", type=int, required=True, help="Maximum number of samples per client.")
    parser.add_argument("-delta", type=float, default=3e-6, help="Target delta in (epsilon, delta)-DP.")
    parser.add_argument("-sigma", type=float, required=True, help="Noise-to-sensitivity ratio.")
    parser.add_argument("-mask", type=float, default=0.3, help="Fraction of valid tokens to mask in each mutation round.")
    parser.add_argument("-lookahead", type=int, default=4, help="Number of future mutations averaged for candidate scoring.")
    parser.add_argument("-multiplier", type=int, default=4, help="Multiplier controlling the synthetic population size.")
    parser.add_argument("-seq_len", type=int, default=64, help="Maximum sequence length for tokenization.")
    parser.add_argument("-t_steps", type=int, default=2, help="How many mask-fill passes to apply per mutation.")
    parser.add_argument("-trial", type=int, default=0, help="Trial identifier appended to the experiment directory.")
    parser.add_argument("-H_multiplier", type=float, default=0.25, help="Multiplier controlling histogram threshold H.")
    return parser.parse_args()


def main() -> None:
    """Translate the legacy arguments to the new config-driven Stage 1 runner."""

    args = parse_args()
    del args.cachedir
    config = build_legacy_config(args, base_dir=Path(__file__).resolve().parent, stage="stage1")
    summary = run_stage1(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
