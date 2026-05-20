#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-}"
if [[ -z "$MODEL_DIR" ]]; then
  echo "usage: bash scripts/run_round23_unseen_dataset_final_eval_repeat40.sh <model-dir> [extra-runner-args...]"
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

shift
bash scripts/run_round23_dynamic_experiments.sh unseen_dataset_final_eval "$MODEL_DIR" "$@"
