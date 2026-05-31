#!/usr/bin/env bash
# Run PrE-Text screening experiments on unseen2 datasets (imdb, openreview).
# 30 experiments total: 2 datasets × 15 seeds.
# Executes sequentially, retries up to 3 times on failure (10s delay between retries).
# Outputs: paper-new-round-14/logs/pretext_screening_15rounds_unseen2_summary.tsv
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1
TARGET_GPU_INDEX="${TARGET_GPU_INDEX:-${CUDA_VISIBLE_DEVICES}}"
RESET_SUMMARY="${RESET_SUMMARY:-1}"

ARGS=(
  --manifest-path "paper-new-round-14/configs/experiments/pretext_screening_unseen2/pretext_screening_unseen2_manifest.tsv"
  --summary-path "paper-new-round-14/logs/pretext_screening_15rounds_unseen2_summary.tsv"
  --log-dir "paper-new-round-14/logs/pretext_screening_unseen2_logs"
  --python-executable "${PYTHON_BIN}"
  --max-attempts 3
  --retry-delay-seconds 10
  --min-free-gb-for-vllm 2
  --gpu-wait-poll-seconds 60
  --gpu-wait-timeout-seconds 43200
  --target-gpu-index "${TARGET_GPU_INDEX}"
)

if [[ "${RESET_SUMMARY}" == "1" ]]; then
  ARGS+=(--reset-summary)
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" "paper-new-round-14/scripts/run_pretext_screening_unseen2_runner.py" "${ARGS[@]}"
