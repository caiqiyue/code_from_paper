"""Legacy LLaMA2 evaluation entry point kept as a thin wrapper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pretext_platform.core.legacy import build_legacy_config
from pretext_platform.core.pipeline import run_eval_large


def parse_args() -> argparse.Namespace:
    """Parse the original LLaMA2 evaluation CLI arguments."""

    parser = argparse.ArgumentParser("Evaluate on downstream LLaMA-2 next token prediction task.")
    parser.add_argument("-datadir", type=str, default="", help="Dataset name prefix for the eval JSON file.")
    parser.add_argument("-outputdir", type=str, required=True, help="Base directory where experiment outputs are stored.")
    parser.add_argument("-cachedir", type=str, required=False, help="Deprecated legacy argument kept for compatibility.")
    parser.add_argument("-sensitivity", type=int, required=True, help="Maximum number of samples per client.")
    parser.add_argument("-delta", type=float, default=3e-6, help="Target delta in (epsilon, delta)-DP.")
    parser.add_argument("-sigma", type=float, required=True, help="Noise-to-sensitivity ratio used by the corresponding generation run.")
    parser.add_argument("-mask", type=float, default=0.3, help="Mask ratio used by the corresponding generation run.")
    parser.add_argument("-lookahead", type=int, default=4, help="Lookahead count used by the corresponding generation run.")
    parser.add_argument("-multiplier", type=int, default=4, help="Synthetic population multiplier used during stage one.")
    parser.add_argument("-seq_len", type=int, default=64, help="Sequence length used by the corresponding generation run.")
    parser.add_argument("-t_steps", type=int, default=2, help="Mutation steps used by the corresponding generation run.")
    parser.add_argument("-trial", type=int, default=0, help="Trial identifier appended to the experiment directory.")
    parser.add_argument("-H_multiplier", type=float, default=0.25, help="Histogram threshold multiplier used by the generation run.")
    return parser.parse_args()


def main() -> None:
    """Translate the legacy arguments to the new config-driven LLaMA2 runner."""

    args = parse_args()
    del args.cachedir
    config = build_legacy_config(args, base_dir=Path(__file__).resolve().parent, stage="eval_large")
    summary = run_eval_large(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
