#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1
TARGET_GPU_INDEX="${TARGET_GPU_INDEX:-${CUDA_VISIBLE_DEVICES}}"
RESET_SUMMARY="${RESET_SUMMARY:-1}"

LOCK_ROOT="$ROOT/paper-new-round23/logs/locks"
LOCK_DIR="$LOCK_ROOT/e4_all6_repeat15_270.lock"
mkdir -p "$LOCK_ROOT"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  LOCK_PID="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
  if [[ -n "$LOCK_PID" ]] && kill -0 "$LOCK_PID" 2>/dev/null; then
    echo "Another E4 all6 repeat15 run is already active: pid=$LOCK_PID"
    exit 3
  fi
  echo "Removing stale lock: $LOCK_DIR"
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR"
fi
echo "$$" > "$LOCK_DIR/pid"
trap 'rm -rf "$LOCK_DIR"' EXIT INT TERM

COMMON_ARGS=(
  --max-attempts 3
  --retry-delay-seconds 10
  --retry-all-failures
  --min-free-gb-for-vllm 2
  --target-gpu-index "$TARGET_GPU_INDEX"
)

cd "$ROOT"
python paper-new-round23/scripts/generate_round23_e4_experiment_configs.py \
  --mode e4_a_oneshot_all6_repeat15 \
  --mode e4_b_keepk0_all6_repeat15 \
  --mode e4_c_three_round_stress_all6_repeat15

set +e
STATUS=0

for MODE in \
  e4_a_oneshot_all6_repeat15 \
  e4_b_keepk0_all6_repeat15 \
  e4_c_three_round_stress_all6_repeat15
do
  MODE_ARGS=("${COMMON_ARGS[@]}")
  if [[ "$RESET_SUMMARY" == "1" ]]; then
    MODE_ARGS+=(--reset-summary)
  fi
  python paper-new-round23/scripts/round23_dynamic_experiment_runner.py \
    --mode "$MODE" \
    "${MODE_ARGS[@]}" \
    "$@"
  RUN_STATUS=$?
  if [[ "$RUN_STATUS" -ne 0 ]]; then
    STATUS="$RUN_STATUS"
  fi
done

exit "$STATUS"
