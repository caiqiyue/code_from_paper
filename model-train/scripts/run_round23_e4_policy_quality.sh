#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_TRAIN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
CONFIG_PATH="configs/round23_e4_a_all6_absolute_k_policy_quality.json"
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
  "${PYTHON_BIN}" eval_round23_e4_round_count.py
  --controller-context-table "$(read_config_value controller_context_table)"
  --round19-replay-table "$(read_config_value round19_replay_table)"
  --absolute-k-model-dir "$(read_config_value absolute_k_model_dir)"
  --absolute-k-model-family "$(read_config_value absolute_k_model_family)"
  --absolute-k-feature-version "$(read_config_value absolute_k_feature_version)"
  --absolute-k-config "$(read_config_value absolute_k_model_config)"
  --round23-model-dir "$(read_config_value round23_model_dir)"
  --round23-model-family "$(read_config_value round23_model_family)"
  --round23-feature-version "$(read_config_value round23_feature_version)"
  --round23-config "$(read_config_value round23_model_config)"
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
