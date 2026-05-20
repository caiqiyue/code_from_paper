#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-}"
if [[ -z "$MODEL_DIR" ]]; then
  echo "usage: bash scripts/run_thesis_e1_smoke20_sequential.sh <round23-controller-bundle> [extra-runner-args...]"
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1

shift 1

cd "$ROOT"
python paper-new-round23/scripts/generate_round23_experiment_configs.py --mode thesis_main_seen_smoke
python paper-new-round19/scripts/generate_thesis_e1_main_configs.py --mode thesis_main_seen_smoke

cd "$ROOT/paper-new-round19"
python -m paper_new_selector.thesis_e1_main_runner \
  --mode thesis_main_seen_smoke \
  --execute \
  --reset-summary \
  "$@"

cd "$ROOT"
python paper-new-round23/scripts/round23_dynamic_experiment_runner.py \
  --mode thesis_main_seen_smoke \
  --model-dir "$MODEL_DIR" \
  --reset-summary \
  "$@"
