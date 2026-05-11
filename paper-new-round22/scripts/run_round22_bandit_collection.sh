#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
  echo "usage: bash scripts/run_round22_bandit_collection.sh <smoke|full> [extra-runner-args...]"
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && -z "${CUDA_DEVICE_ORDER:-}" ]]; then
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
fi

shift
python scripts/round22_bandit_collection_runner.py --mode "$MODE" "$@"
