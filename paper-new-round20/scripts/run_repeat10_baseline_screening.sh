#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${REPEAT10_CUDA_VISIBLE_DEVICES:-1}"
python -m paper_new_selector.repeat10_baseline_runner
