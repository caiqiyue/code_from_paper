#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TARGET_GPU_INDEX="${TARGET_GPU_INDEX:-1}"
RESET_SUMMARY="${RESET_SUMMARY:-1}"

export CUDA_VISIBLE_DEVICES="${TARGET_GPU_INDEX}"

ARGS=(
  --manifest-path "paper-new-round-14/configs/experiments/single_node_tuning_e6_round14_budget_sweep/smoke90/round14_lineage_e6_smoke90_manifest.tsv"
  --summary-path "paper-new-round-14/logs/round14_lineage_e6_smoke90_summary.tsv"
  --log-dir "paper-new-round-14/logs/round14_lineage_e6_smoke90_logs"
  --python-executable "${PYTHON_BIN}"
  --max-attempts 3
  --retry-delay-seconds 10
)

if [[ "${RESET_SUMMARY}" == "1" ]]; then
  ARGS+=(--reset-summary)
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" "paper-new-round-14/scripts/run_round14_lineage_manifest.py" "${ARGS[@]}"
