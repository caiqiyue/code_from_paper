#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && -z "${CUDA_DEVICE_ORDER:-}" ]]; then
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
fi
python -m paper_new_selector.repeat15_runner
