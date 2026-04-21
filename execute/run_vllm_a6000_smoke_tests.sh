#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ENV_NAME="${ENV_NAME:-caiqiyue-vllm}"
CONDA_SH="${CONDA_SH:-${HOME}/anaconda3/etc/profile.d/conda.sh}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/execute/logs}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-5}"

SN_CONFIG="${SN_CONFIG:-thesis_platform/configs/experiments/single_node_formal/sn_test_vllm_a6000.yaml}"
SP_CONFIG="${SP_CONFIG:-configs/experiments/single_node_formal/sp_test_vllm_a6000.yaml}"

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_ROOT}"
MASTER_LOG="${LOG_ROOT}/vllm_a6000_smoke_${STAMP}.log"
MONITOR_LOG="${LOG_ROOT}/vllm_a6000_smoke_gpu_${STAMP}.log"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${MASTER_LOG}"
}

find_a6000_index() {
  if [[ -n "${A6000_GPU_INDEX:-}" ]]; then
    printf '%s\n' "${A6000_GPU_INDEX}"
    return 0
  fi
  nvidia-smi --query-gpu=index,name --format=csv,noheader | \
    awk -F, 'BEGIN { IGNORECASE=1 } $2 ~ /A6000/ { gsub(/[[:space:]]/, "", $1); print $1; exit }'
}

emit_gpu_status() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    {
      printf '%s\n' '--- gpu status ---'
      nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.free,memory.total --format=csv,noheader
      printf '%s\n' '--- compute apps ---'
      nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader || true
    } | tee -a "${MASTER_LOG}"
  else
    log "nvidia-smi not found."
  fi
}

start_monitor() {
  (
    while true; do
      printf '=== %s ===\n' "$(date '+%F %T')"
      nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.free,memory.total --format=csv,noheader || true
      printf '%s\n' '--- compute apps ---'
      nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader || true
      sleep "${MONITOR_INTERVAL}"
    done
  ) >> "${MONITOR_LOG}" 2>&1 &
  MONITOR_PID=$!
}

stop_monitor() {
  if [[ -n "${MONITOR_PID:-}" ]] && kill -0 "${MONITOR_PID}" >/dev/null 2>&1; then
    kill "${MONITOR_PID}" >/dev/null 2>&1 || true
    wait "${MONITOR_PID}" >/dev/null 2>&1 || true
  fi
}

run_and_log() {
  local label="$1"
  shift
  log "START ${label}"
  set +e
  "$@" 2>&1 | tee -a "${MASTER_LOG}"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    log "FAIL ${label} rc=${rc}"
    return "${rc}"
  fi
  log "PASS ${label}"
}

run_sn_smoke() {
  cd "${REPO_ROOT}"
  python -m thesis_platform.scripts.run_experiment --config "${SN_CONFIG}"
}

run_sp_smoke() {
  cd "${REPO_ROOT}/PrE-Text"
  python -m pretext_platform.scripts.run_pipeline --config "${SP_CONFIG}"
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  log "nvidia-smi is required to select and verify the A6000 GPU."
  exit 1
fi

A6000_INDEX="$(find_a6000_index)"
if [[ -z "${A6000_INDEX}" ]]; then
  log "Could not find an NVIDIA RTX A6000. Set A6000_GPU_INDEX=<index> to override."
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | tee -a "${MASTER_LOG}" || true
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${A6000_INDEX}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

log "repo_root=${REPO_ROOT}"
log "env_name=${ENV_NAME}"
log "a6000_physical_index=${A6000_INDEX}"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log "master_log=${MASTER_LOG}"
log "monitor_log=${MONITOR_LOG}"
emit_gpu_status

if [[ ! -f "${CONDA_SH}" ]]; then
  log "conda activation script not found: ${CONDA_SH}"
  exit 1
fi

source "${CONDA_SH}"
conda activate "${ENV_NAME}"

python - <<'PY' 2>&1 | tee -a "${MASTER_LOG}"
import os
import torch

print("python_visible_cuda_devices=" + str(os.environ.get("CUDA_VISIBLE_DEVICES")))
print("torch_cuda_available=" + str(torch.cuda.is_available()))
print("torch_cuda_device_count=" + str(torch.cuda.device_count()))
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("Expected exactly one visible CUDA device after CUDA_VISIBLE_DEVICES filtering.")
name = torch.cuda.get_device_name(0)
print("torch_cuda_device_0_name=" + name)
if "A6000" not in name:
    raise SystemExit(f"Expected visible CUDA device 0 to be A6000, got {name!r}.")
PY

start_monitor
trap stop_monitor EXIT

run_and_log "SN vLLM A6000 smoke" run_sn_smoke

run_and_log "SP vLLM A6000 smoke" run_sp_smoke

stop_monitor
emit_gpu_status
log "DONE vLLM A6000 smoke tests."
