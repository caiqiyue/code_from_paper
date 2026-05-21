#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-}"
if [[ -z "$MODEL_DIR" ]]; then
  echo "usage: bash scripts/run_thesis_e2_extra_unseen_repeat15_135_sequential.sh <round23-controller-bundle> [extra-runner-args...]"
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1

shift 1

COMMON_ARGS=(
  --max-attempts 3
  --min-free-gb-for-vllm 2
)

cd "$ROOT"
python paper-new-round23/scripts/generate_round23_experiment_configs.py --mode thesis_e2_extra_unseen_repeat15
python paper-new-round19/scripts/generate_thesis_e1_main_configs.py --mode thesis_e2_extra_unseen_repeat15

set +e
STATUS=0

cd "$ROOT/paper-new-round19"
python -m paper_new_selector.thesis_e1_main_runner \
  --mode thesis_e2_extra_unseen_repeat15 \
  --execute \
  "${COMMON_ARGS[@]}" \
  "$@"
BASE_STATUS=$?
if [[ "$BASE_STATUS" -ne 0 ]]; then
  STATUS="$BASE_STATUS"
fi

cd "$ROOT"
python paper-new-round23/scripts/round23_dynamic_experiment_runner.py \
  --mode thesis_e2_extra_unseen_repeat15 \
  --model-dir "$MODEL_DIR" \
  "${COMMON_ARGS[@]}" \
  "$@"
R23_STATUS=$?
if [[ "$R23_STATUS" -ne 0 ]]; then
  STATUS="$R23_STATUS"
fi

exit "$STATUS"
