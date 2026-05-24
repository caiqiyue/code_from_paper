#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_TRAIN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
CONFIG_PATH="configs/round23_e3_policy_quality_1200_all6.json"
DRY_RUN=""

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN="--dry-run"
elif [[ -n "${1:-}" ]]; then
  CONFIG_PATH="$1"
  DRY_RUN="${2:-}"
fi

cd "${MODEL_TRAIN_ROOT}"

read_config_value() {
  local key="$1"
  "${PYTHON_BIN}" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]])' "${CONFIG_PATH}" "${key}"
}

CMD=(
  "${PYTHON_BIN}" eval_round23_e3_policy_quality.py
  --controller-context-table "$(read_config_value controller_context_table)"
  --round19-replay-table "$(read_config_value round19_replay_table)"
  --model-dir "$(read_config_value model_dir)"
  --model-family "$(read_config_value model_family)"
  --feature-version "$(read_config_value feature_version)"
  --config "$(read_config_value model_config)"
  --output-dir "$(read_config_value output_dir)"
  --scope "$(read_config_value scope)"
)

if [[ "${DRY_RUN}" == "--dry-run" ]]; then
  printf 'Working directory: %s\n' "${MODEL_TRAIN_ROOT}"
  printf 'Config: %s\n' "${CONFIG_PATH}"
  printf 'Command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"
