#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
MODEL_DIR="${2:-}"
if [[ -z "$MODE" || -z "$MODEL_DIR" ]]; then
  echo "usage: bash scripts/run_round23_dynamic_experiments.sh <real_smoke|quick_compare> <model-dir> [extra-runner-args...]"
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && -z "${CUDA_DEVICE_ORDER:-}" ]]; then
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
fi

shift 2
python scripts/round23_dynamic_experiment_runner.py --mode "$MODE" --model-dir "$MODEL_DIR" "$@"
