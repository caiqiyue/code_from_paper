#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="/root/autodl-tmp/caiqiyue/code_from_paper"
PRETEXT_ROOT="${REPO_ROOT}/PrE-Text"
CONDA_SH="/root/miniconda3/etc/profile.d/conda.sh"
DEFAULT_INTERVAL_SECONDS=1800

INTERVAL_SECONDS="${DEFAULT_INTERVAL_SECONDS}"
START_FROM=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  run_formal_experiment_queue.sh [--interval-seconds 1800] [--start-from GC1] [--dry-run]

Behavior:
  - Runs the formal experiment queue serially.
  - Switches conda env automatically for thesis_platform vs PrE-Text.
  - Polls status every 30 minutes by default.
  - If one experiment fails, records the failure and continues.
  - Alias experiments are recorded in the summary without rerunning the same config.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval-seconds)
      INTERVAL_SECONDS="$2"
      shift 2
      ;;
    --start-from)
      START_FROM="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Missing conda init script: ${CONDA_SH}" >&2
  exit 1
fi

if ! [[ "${INTERVAL_SECONDS}" =~ ^[0-9]+$ ]] || [[ "${INTERVAL_SECONDS}" -le 0 ]]; then
  echo "--interval-seconds must be a positive integer" >&2
  exit 1
fi

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${REPO_ROOT}/outputs/formal_scheduler/${RUN_STAMP}"
MASTER_LOG="${RUN_ROOT}/scheduler.log"
SUMMARY_TSV="${RUN_ROOT}/summary.tsv"
mkdir -p "${RUN_ROOT}"

log() {
  local message="$1"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${message}" | tee -a "${MASTER_LOG}"
}

exp_log() {
  local log_path="$1"
  local message="$2"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${message}" >> "${log_path}"
}

append_summary() {
  local experiment_id="$1"
  local project="$2"
  local status="$3"
  local exit_code="$4"
  local alias_of="$5"
  local started_at="$6"
  local finished_at="$7"
  local log_path="$8"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${experiment_id}" "${project}" "${status}" "${exit_code}" "${alias_of}" "${started_at}" "${finished_at}" "${log_path}" \
    >> "${SUMMARY_TSV}"
}

printf 'experiment_id\tproject\tstatus\texit_code\talias_of\tstarted_at\tfinished_at\tlog_path\n' > "${SUMMARY_TSV}"

declare -A EXP_PROJECT
declare -A EXP_CONFIG
declare -A EXP_WORKDIR
declare -A EXP_ENV
declare -A EXP_KIND
declare -A EXP_ALIAS_OF
QUEUE=()

register_run() {
  local experiment_id="$1"
  local project="$2"
  local config_path="$3"
  local workdir="$4"
  local env_name="$5"
  local kind="$6"
  EXP_PROJECT["${experiment_id}"]="${project}"
  EXP_CONFIG["${experiment_id}"]="${config_path}"
  EXP_WORKDIR["${experiment_id}"]="${workdir}"
  EXP_ENV["${experiment_id}"]="${env_name}"
  EXP_KIND["${experiment_id}"]="${kind}"
  QUEUE+=("${experiment_id}")
}

register_alias() {
  local experiment_id="$1"
  local target_id="$2"
  local project="$3"
  EXP_PROJECT["${experiment_id}"]="${project}"
  EXP_ALIAS_OF["${experiment_id}"]="${target_id}"
}

# Recommended order:
# 1. Build both methods' main baselines first.
# 2. Finish low-risk thesis formal groups.
# 3. Run the full PrE-Text privacy sweep.
# 4. Leave the large-model high-risk jobs to the end.

register_run "GC1" "thesis_platform" "thesis_platform/configs/experiments/linux/jobs_real_datainf_v3_linux.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"
register_run "GC2" "PrE-Text" "configs/experiments/jobs_real_eps129.yaml" "${PRETEXT_ROOT}" "pretext" "pretext"
register_run "GC3" "thesis_platform" "thesis_platform/configs/experiments/linux/congressional_real_datainf_v3_linux.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"
register_run "GC4" "PrE-Text" "configs/experiments/congressional_real_eps129.yaml" "${PRETEXT_ROOT}" "pretext" "pretext"
register_run "GC6" "thesis_platform" "thesis_platform/configs/experiments/linux/forums_real_datainf_v3_linux.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"
register_run "GC8" "PrE-Text" "configs/experiments/forums_real_eps129.yaml" "${PRETEXT_ROOT}" "pretext" "pretext"
register_run "GC7" "thesis_platform" "thesis_platform/configs/experiments/linux/microblog_real_datainf_v3_linux.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"
register_run "GC9" "PrE-Text" "configs/experiments/microblog_real_eps129.yaml" "${PRETEXT_ROOT}" "pretext" "pretext"

register_run "GA1" "thesis_platform" "thesis_platform/configs/experiments/linux/jobs_real_datainf_v3_ga1_no_critique_linux.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"
register_run "GA2" "thesis_platform" "thesis_platform/configs/experiments/linux/jobs_real_datainf_v3_ga2_no_routing_linux.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"
register_run "GA4" "thesis_platform" "thesis_platform/configs/experiments/linux/jobs_real_datainf_v3_momentum0_linux.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"
register_run "GA5" "thesis_platform" "thesis_platform/configs/experiments/linux/jobs_real_datainf_v3_ga5_uid_agg_linux.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"
register_run "GA6" "thesis_platform" "thesis_platform/configs/experiments/linux/jobs_real_datainf_v3_ga6_random_sel_linux.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"
register_run "GA7" "thesis_platform" "thesis_platform/configs/experiments/linux/jobs_real_datainf_v3_ga7_random_ret_linux.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"

register_run "GP1" "thesis_platform" "thesis_platform/configs/experiments/linux/jobs_real_datainf_v3_gp1_eps05_linux.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"
register_run "GP3" "thesis_platform" "thesis_platform/configs/experiments/linux/jobs_real_datainf_v3_gp3_eps758_linux.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"
register_run "GP4" "thesis_platform" "thesis_platform/configs/experiments/linux/jobs_real_datainf_v3_gp4_no_privacy_linux.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"

register_run "GS2" "thesis_platform" "thesis_platform/configs/experiments/linux/jobs_real_datainf_v3_gs2_gradmm_linux.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"
register_run "GS3" "thesis_platform" "thesis_platform/configs/experiments/linux/jobs_real_datainf_v3_gs3_ira_linux.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"

register_run "GAgg2" "thesis_platform" "thesis_platform/configs/experiments/linux/jobs_real_datainf_v3_gagg2_dbscan_attn_linux.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"
register_run "GAgg3" "thesis_platform" "thesis_platform/configs/experiments/linux/jobs_real_datainf_v3_gagg3_uid_linux.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"
register_run "GAgg4" "thesis_platform" "thesis_platform/configs/experiments/linux/jobs_real_datainf_v3_gagg4_summ_linux.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"

register_run "GScale1" "thesis_platform" "thesis_platform/configs/experiments/linux/jobs_real_datainf_v3_scale_8clients.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"
register_run "GScale3" "thesis_platform" "thesis_platform/configs/experiments/linux/jobs_real_datainf_v3_scale_32clients.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"

register_run "PT-P1" "PrE-Text" "configs/experiments/jobs_real_eps05.yaml" "${PRETEXT_ROOT}" "pretext" "pretext"
register_run "PT-P3" "PrE-Text" "configs/experiments/jobs_real_eps758.yaml" "${PRETEXT_ROOT}" "pretext" "pretext"
register_run "PT-P4" "PrE-Text" "configs/experiments/jobs_real_no_privacy.yaml" "${PRETEXT_ROOT}" "pretext" "pretext"
register_run "PT-P5" "PrE-Text" "configs/experiments/congressional_real_eps05.yaml" "${PRETEXT_ROOT}" "pretext" "pretext"
register_run "PT-P7" "PrE-Text" "configs/experiments/congressional_real_eps758.yaml" "${PRETEXT_ROOT}" "pretext" "pretext"
register_run "PT-P8" "PrE-Text" "configs/experiments/congressional_real_no_privacy.yaml" "${PRETEXT_ROOT}" "pretext" "pretext"
register_run "PT-P9" "PrE-Text" "configs/experiments/forums_real_eps05.yaml" "${PRETEXT_ROOT}" "pretext" "pretext"
register_run "PT-P11" "PrE-Text" "configs/experiments/forums_real_eps758.yaml" "${PRETEXT_ROOT}" "pretext" "pretext"
register_run "PT-P12" "PrE-Text" "configs/experiments/forums_real_no_privacy.yaml" "${PRETEXT_ROOT}" "pretext" "pretext"
register_run "PT-P13" "PrE-Text" "configs/experiments/microblog_real_eps05.yaml" "${PRETEXT_ROOT}" "pretext" "pretext"
register_run "PT-P15" "PrE-Text" "configs/experiments/microblog_real_eps758.yaml" "${PRETEXT_ROOT}" "pretext" "pretext"
register_run "PT-P16" "PrE-Text" "configs/experiments/microblog_real_no_privacy.yaml" "${PRETEXT_ROOT}" "pretext" "pretext"

register_run "GTransfer1" "thesis_platform" "thesis_platform/configs/experiments/linux/jobs_to_forums_transfer.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"
register_run "GTransfer2" "thesis_platform" "thesis_platform/configs/experiments/linux/jobs_to_microblog_transfer.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"
register_run "GTransfer3" "thesis_platform" "thesis_platform/configs/experiments/linux/jobs_to_congressional_transfer.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"
register_run "GTransfer4" "thesis_platform" "thesis_platform/configs/experiments/linux/forums_to_jobs_transfer.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"
register_run "GTransfer5" "thesis_platform" "thesis_platform/configs/experiments/linux/microblog_to_jobs_transfer.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"
register_run "GTransfer6" "thesis_platform" "thesis_platform/configs/experiments/linux/congressional_to_jobs_transfer.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"

register_run "GRun1" "thesis_platform" "thesis_platform/configs/experiments/linux/jobs_real_datainf_v3_run1_linux.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"
register_run "GRun2" "thesis_platform" "thesis_platform/configs/experiments/linux/jobs_real_datainf_v3_run2_linux.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"
register_run "GRun3" "thesis_platform" "thesis_platform/configs/experiments/linux/jobs_real_datainf_v3_run3_linux.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"

register_run "GC5" "thesis_platform" "thesis_platform/configs/experiments/linux/jobs_real_datainf_v3_gc5_linux.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"
register_run "GM2" "thesis_platform" "thesis_platform/configs/experiments/linux/jobs_real_datainf_v3_gm2_llama2_7b_linux.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"
register_run "GM4" "thesis_platform" "thesis_platform/configs/experiments/linux/jobs_real_datainf_v3_gm4_linux.yaml" "${REPO_ROOT}" "caiqiyue" "thesis"

register_alias "GP2" "GC1" "thesis_platform"
register_alias "GAgg1" "GC1" "thesis_platform"
register_alias "GScale2" "GC1" "thesis_platform"
register_alias "GM1" "GC1" "thesis_platform"
register_alias "GM3" "GC5" "thesis_platform"
register_alias "PT-P2" "GC2" "PrE-Text"
register_alias "PT-P6" "GC4" "PrE-Text"
register_alias "PT-P10" "GC8" "PrE-Text"
register_alias "PT-P14" "GC9" "PrE-Text"

emit_gpu_status() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | \
      sed 's/^/  GPU /' | tee -a "${MASTER_LOG}" >/dev/null || true
  else
    log "nvidia-smi not found; GPU status skipped."
  fi
}

cleanup_gpu_context() {
  local env_name="$1"
  local workdir="$2"

  bash -lc "source '${CONDA_SH}' && conda activate '${env_name}' && cd '${workdir}' && python - <<'PY'
import gc

try:
    import torch
except Exception:
    torch = None

gc.collect()
if torch is not None and torch.cuda.is_available():
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
PY" >/dev/null 2>&1 || true
}

cleanup_after_experiment() {
  local experiment_id="$1"
  local pid="$2"
  local env_name="$3"
  local workdir="$4"
  local log_path="$5"

  log "CLEANUP ${experiment_id}: clearing residual processes and GPU cache."
  exp_log "${log_path}" "CLEANUP started for ${experiment_id}."

  if kill -0 "${pid}" >/dev/null 2>&1; then
    kill -TERM "-${pid}" >/dev/null 2>&1 || true
    sleep 5
  fi
  kill -KILL "-${pid}" >/dev/null 2>&1 || true

  cleanup_gpu_context "${env_name}" "${workdir}"
  emit_gpu_status

  exp_log "${log_path}" "CLEANUP finished for ${experiment_id}."
}

emit_runtime_status() {
  local experiment_id="$1"
  local kind="${EXP_KIND[${experiment_id}]}"
  local config_path="${EXP_CONFIG[${experiment_id}]}"
  local workdir="${EXP_WORKDIR[${experiment_id}]}"
  local env_name="${EXP_ENV[${experiment_id}]}"

  if [[ "${kind}" == "thesis" ]]; then
    bash -lc "source '${CONDA_SH}' && conda activate '${env_name}' && cd '${workdir}' && python - '${config_path}' <<'PY'
import json
import sys
from pathlib import Path
from thesis_platform.core.config import load_experiment_config

config = load_experiment_config(sys.argv[1])
experiment_id = str(config.meta.get('experiment_id', config.path.stem))
pointer = Path(config.output_root()) / f'{experiment_id}_latest.json'
print(f'  experiment_dir_pointer={pointer}')
if pointer.exists():
    payload = json.loads(pointer.read_text(encoding='utf-8'))
    print(f\"  latest_status={payload.get('status')}\")
    experiment_dir = Path(payload.get('experiment_dir', ''))
    print(f\"  experiment_dir={experiment_dir}\")
    run_state = Path(payload.get('run_state_path', ''))
    if run_state.exists():
        state = json.loads(run_state.read_text(encoding='utf-8'))
        completed = state.get('completed_rounds')
        total = state.get('rounds_total')
        current = state.get('current_round')
        phase = state.get('phase')
        downstream = state.get('downstream_status')
        print(f\"  phase={phase} current_round={current} completed_rounds={completed}/{total} downstream_status={downstream}\")
else:
    print('  latest_pointer_missing')
PY" 2>/dev/null | tee -a "${MASTER_LOG}" >/dev/null || true
  else
    bash -lc "source '${CONDA_SH}' && conda activate '${env_name}' && cd '${workdir}' && python - '${config_path}' <<'PY'
import sys
from pathlib import Path
from pretext_platform.core.config import load_experiment_config

config = load_experiment_config(sys.argv[1])
experiment_id = config.experiment_id()
run_dir = Path(config.output_root()) / experiment_id
print(f'  run_dir={run_dir}')
for name in ['stage1_summary.json', 'stage2_summary.json', 'eval_small_summary.json', 'metrics_summary.json']:
    path = run_dir / name
    print(f\"  {name}={'present' if path.exists() else 'missing'}\")
PY" 2>/dev/null | tee -a "${MASTER_LOG}" >/dev/null || true
  fi
}

verify_success() {
  local experiment_id="$1"
  local kind="${EXP_KIND[${experiment_id}]}"
  local config_path="${EXP_CONFIG[${experiment_id}]}"
  local workdir="${EXP_WORKDIR[${experiment_id}]}"
  local env_name="${EXP_ENV[${experiment_id}]}"

  if [[ "${kind}" == "thesis" ]]; then
    bash -lc "source '${CONDA_SH}' && conda activate '${env_name}' && cd '${workdir}' && python - '${config_path}' <<'PY'
import json
import sys
from pathlib import Path
from thesis_platform.core.config import load_experiment_config

config = load_experiment_config(sys.argv[1])
experiment_id = str(config.meta.get('experiment_id', config.path.stem))
pointer = Path(config.output_root()) / f'{experiment_id}_latest.json'
if not pointer.exists():
    raise SystemExit(1)
payload = json.loads(pointer.read_text(encoding='utf-8'))
status = payload.get('status')
experiment_dir = Path(payload.get('experiment_dir', ''))
metrics = experiment_dir / 'metrics_summary.json'
raise SystemExit(0 if status == 'completed' and metrics.exists() else 1)
PY" >/dev/null 2>&1
  else
    bash -lc "source '${CONDA_SH}' && conda activate '${env_name}' && cd '${workdir}' && python - '${config_path}' <<'PY'
import sys
from pathlib import Path
from pretext_platform.core.config import load_experiment_config

config = load_experiment_config(sys.argv[1])
run_dir = Path(config.output_root()) / config.experiment_id()
stage2_json = run_dir / 'stage2' / 'llama7b_text_syn.json'
eval_small = run_dir / 'eval_small_summary.json'
raise SystemExit(0 if stage2_json.exists() and eval_small.exists() else 1)
PY" >/dev/null 2>&1
  fi
}

start_experiment() {
  local experiment_id="$1"
  local kind="${EXP_KIND[${experiment_id}]}"
  local config_path="${EXP_CONFIG[${experiment_id}]}"
  local workdir="${EXP_WORKDIR[${experiment_id}]}"
  local env_name="${EXP_ENV[${experiment_id}]}"
  local log_path="${RUN_ROOT}/${experiment_id}.log"

  local inner_cmd=""
  if [[ "${kind}" == "thesis" ]]; then
    inner_cmd="python -m thesis_platform.scripts.run_experiment --config '${config_path}'"
  else
    inner_cmd="python -m pretext_platform.scripts.run_pipeline --config '${config_path}' && python -m pretext_platform.scripts.run_eval_small --config '${config_path}'"
  fi

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "DRY RUN ${experiment_id}: env=${env_name} workdir=${workdir} cmd=${inner_cmd}"
    return 0
  fi

  exp_log "${log_path}" "EXPERIMENT ${experiment_id} started."
  exp_log "${log_path}" "ENV=${env_name} WORKDIR=${workdir}"
  exp_log "${log_path}" "COMMAND=${inner_cmd}"
  setsid bash -lc "source '${CONDA_SH}' && conda activate '${env_name}' && cd '${workdir}' && ${inner_cmd}" >> "${log_path}" 2>&1 &
  echo "$!"
}

monitor_experiment() {
  local experiment_id="$1"
  local pid="$2"
  local log_path="${RUN_ROOT}/${experiment_id}.log"

  while kill -0 "${pid}" >/dev/null 2>&1; do
    log "STATUS ${experiment_id}: pid=${pid} still running."
    emit_gpu_status
    emit_runtime_status "${experiment_id}"
    tail -n 5 "${log_path}" 2>/dev/null | sed 's/^/  LOG /' | tee -a "${MASTER_LOG}" >/dev/null || true
    sleep "${INTERVAL_SECONDS}"
  done
}

record_aliases() {
  for experiment_id in "GP2" "GAgg1" "GScale2" "GM1" "GM3" "PT-P2" "PT-P6" "PT-P10" "PT-P14"; do
    local target_id="${EXP_ALIAS_OF[${experiment_id}]}"
    local status="alias_waiting"
    local exit_code="-"
    if grep -q "^${target_id}"$'\t'"${EXP_PROJECT[${target_id}]}"$'\t'"success"$'\t' "${SUMMARY_TSV}" 2>/dev/null; then
      status="alias_reused"
    elif grep -q "^${target_id}"$'\t'"${EXP_PROJECT[${target_id}]}"$'\t'"failed"$'\t' "${SUMMARY_TSV}" 2>/dev/null; then
      status="alias_source_failed"
    fi
    append_summary "${experiment_id}" "${EXP_PROJECT[${experiment_id}]}" "${status}" "${exit_code}" "${target_id}" "-" "-" "-"
  done
}

log "Scheduler output root: ${RUN_ROOT}"
log "Polling interval: ${INTERVAL_SECONDS} seconds"
log "Formal experiment IDs in document: 57"
log "Unique recommended executions in this queue: 48"

start_index=0
if [[ -n "${START_FROM}" ]]; then
  found=0
  for i in "${!QUEUE[@]}"; do
    if [[ "${QUEUE[$i]}" == "${START_FROM}" ]]; then
      start_index="${i}"
      found=1
      break
    fi
  done
  if [[ "${found}" -ne 1 ]]; then
    echo "Unknown --start-from experiment id: ${START_FROM}" >&2
    exit 1
  fi
fi

for i in "${!QUEUE[@]}"; do
  if (( i < start_index )); then
    continue
  fi

  experiment_id="${QUEUE[$i]}"
  project="${EXP_PROJECT[${experiment_id}]}"
  env_name="${EXP_ENV[${experiment_id}]}"
  workdir="${EXP_WORKDIR[${experiment_id}]}"
  log_path="${RUN_ROOT}/${experiment_id}.log"
  started_at="$(date '+%Y-%m-%d %H:%M:%S')"
  log "START ${experiment_id} (${project})"
  exp_log "${log_path}" "START_TIME=${started_at}"

  pid_or_zero="$(start_experiment "${experiment_id}")"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    finished_at="$(date '+%Y-%m-%d %H:%M:%S')"
    exp_log "${log_path}" "END_TIME=${finished_at}"
    exp_log "${log_path}" "RESULT=dry_run"
    append_summary "${experiment_id}" "${project}" "dry_run" "-" "-" "${started_at}" "${finished_at}" "${RUN_ROOT}/${experiment_id}.log"
    continue
  fi

  pid="${pid_or_zero}"
  monitor_experiment "${experiment_id}" "${pid}"
  wait "${pid}"
  exit_code="$?"
  cleanup_after_experiment "${experiment_id}" "${pid}" "${env_name}" "${workdir}" "${log_path}"
  finished_at="$(date '+%Y-%m-%d %H:%M:%S')"

  if [[ "${exit_code}" -eq 0 ]] && verify_success "${experiment_id}"; then
    log "DONE ${experiment_id} succeeded."
    exp_log "${log_path}" "END_TIME=${finished_at}"
    exp_log "${log_path}" "RESULT=success"
    append_summary "${experiment_id}" "${project}" "success" "${exit_code}" "-" "${started_at}" "${finished_at}" "${RUN_ROOT}/${experiment_id}.log"
  else
    log "FAIL ${experiment_id} exited with code ${exit_code}; continuing to next experiment."
    exp_log "${log_path}" "END_TIME=${finished_at}"
    exp_log "${log_path}" "RESULT=failed"
    exp_log "${log_path}" "EXIT_CODE=${exit_code}"
    append_summary "${experiment_id}" "${project}" "failed" "${exit_code}" "-" "${started_at}" "${finished_at}" "${RUN_ROOT}/${experiment_id}.log"
  fi
done

record_aliases
log "All queued experiments processed. Summary: ${SUMMARY_TSV}"
