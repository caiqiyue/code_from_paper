#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BG_ROOT="${REPO_ROOT}/thesis_platform/workspace/bg_runs"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "${BG_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  run_experiment_bg.sh start --config <config.yaml> [--tag <name>] [--resume] [--resume_dir <dir>]
  run_experiment_bg.sh status --tag <name>
  run_experiment_bg.sh logs --tag <name>
  run_experiment_bg.sh stop --tag <name>

Environment:
  PYTHON_BIN   Python executable to use. Defaults to python3.
EOF
}

require_tag_dir() {
  local tag="$1"
  local tag_dir="${BG_ROOT}/${tag}"
  if [[ ! -d "${tag_dir}" ]]; then
    echo "Missing background run tag: ${tag}" >&2
    exit 1
  fi
  printf '%s' "${tag_dir}"
}

resolve_metadata() {
  local config_path="$1"
  "${PYTHON_BIN}" - "$config_path" <<'PY'
import json
import sys
from thesis_platform.core.config import load_experiment_config

config = load_experiment_config(sys.argv[1])
print(json.dumps({
    "config_path": str(config.path),
    "experiment_id": str(config.meta.get("experiment_id", config.path.stem)),
    "output_root": str(config.output_root()),
}, ensure_ascii=False))
PY
}

start_run() {
  local config=""
  local tag=""
  local resume_flag=""
  local resume_dir=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --config)
        config="$2"
        shift 2
        ;;
      --tag)
        tag="$2"
        shift 2
        ;;
      --resume)
        resume_flag="--resume"
        shift
        ;;
      --resume_dir)
        resume_dir="$2"
        shift 2
        ;;
      *)
        echo "Unknown argument: $1" >&2
        usage
        exit 1
        ;;
    esac
  done

  if [[ -z "${config}" ]]; then
    echo "--config is required" >&2
    exit 1
  fi

  local config_abs
  config_abs="$(cd "$(dirname "${config}")" && pwd)/$(basename "${config}")"
  if [[ ! -f "${config_abs}" ]]; then
    echo "Config not found: ${config_abs}" >&2
    exit 1
  fi

  if [[ -n "${resume_dir}" ]]; then
    resume_dir="$(cd "${resume_dir}" && pwd)"
  fi

  if [[ -z "${tag}" ]]; then
    tag="$(basename "${config_abs}" .yaml)-$(date +%Y%m%d_%H%M%S)"
  fi

  local tag_dir="${BG_ROOT}/${tag}"
  mkdir -p "${tag_dir}"
  local metadata_file="${tag_dir}/metadata.json"
  local pid_file="${tag_dir}/pid"
  local log_file="${tag_dir}/stdout.log"
  local cmd_file="${tag_dir}/command.txt"

  if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" >/dev/null 2>&1; then
    echo "Run tag ${tag} is already active with PID $(cat "${pid_file}")" >&2
    exit 1
  fi

  resolve_metadata "${config_abs}" > "${metadata_file}"
  if [[ -n "${resume_dir}" ]]; then
    printf 'cd %s && %s -m thesis_platform.scripts.run_experiment --config %s %s --resume_dir %s\n' \
      "${REPO_ROOT}" "${PYTHON_BIN}" "${config_abs}" "${resume_flag}" "${resume_dir}" > "${cmd_file}"
  else
    printf 'cd %s && %s -m thesis_platform.scripts.run_experiment --config %s %s\n' \
      "${REPO_ROOT}" "${PYTHON_BIN}" "${config_abs}" "${resume_flag}" > "${cmd_file}"
  fi

  (
    cd "${REPO_ROOT}"
    if [[ -n "${resume_dir}" ]]; then
      nohup "${PYTHON_BIN}" -m thesis_platform.scripts.run_experiment --config "${config_abs}" ${resume_flag} --resume_dir "${resume_dir}" >> "${log_file}" 2>&1 &
    else
      nohup "${PYTHON_BIN}" -m thesis_platform.scripts.run_experiment --config "${config_abs}" ${resume_flag} >> "${log_file}" 2>&1 &
    fi
    echo $! > "${pid_file}"
  )

  echo "Started tag=${tag}"
  echo "PID=$(cat "${pid_file}")"
  echo "Log=${log_file}"
  echo "Metadata=${metadata_file}"
}

status_run() {
  local tag=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --tag)
        tag="$2"
        shift 2
        ;;
      *)
        echo "Unknown argument: $1" >&2
        exit 1
        ;;
    esac
  done
  if [[ -z "${tag}" ]]; then
    echo "--tag is required" >&2
    exit 1
  fi

  local tag_dir
  tag_dir="$(require_tag_dir "${tag}")"
  local pid_file="${tag_dir}/pid"
  local metadata_file="${tag_dir}/metadata.json"
  local pid="unknown"
  local process_status="not_started"
  if [[ -f "${pid_file}" ]]; then
    pid="$(cat "${pid_file}")"
    if kill -0 "${pid}" >/dev/null 2>&1; then
      process_status="running"
    else
      process_status="exited"
    fi
  fi

  echo "tag=${tag}"
  echo "pid=${pid}"
  echo "process_status=${process_status}"
  echo "log_file=${tag_dir}/stdout.log"

  if [[ -f "${metadata_file}" ]]; then
    "${PYTHON_BIN}" - "${metadata_file}" <<'PY'
import json
import pathlib
import sys

metadata = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
pointer_path = pathlib.Path(metadata["output_root"]) / f'{metadata["experiment_id"]}_latest.json'
print(f'experiment_id={metadata["experiment_id"]}')
print(f'output_root={metadata["output_root"]}')
print(f'latest_pointer={pointer_path}')
if pointer_path.exists():
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    print(f'latest_experiment_dir={pointer.get("experiment_dir")}')
    print(f'latest_status={pointer.get("status")}')
    run_state_path = pathlib.Path(pointer.get("run_state_path", ""))
    if run_state_path.exists():
        run_state = json.loads(run_state_path.read_text(encoding="utf-8"))
        print(f'run_phase={run_state.get("phase")}')
        print(f'completed_rounds={run_state.get("completed_rounds")}/{run_state.get("rounds_total")}')
        print(f'current_round={run_state.get("current_round")}')
        print(f'downstream_status={run_state.get("downstream_status")}')
        print(f'metrics_summary={run_state.get("artifacts", {}).get("metrics_summary_path")}')
PY
  fi
}

logs_run() {
  local tag=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --tag)
        tag="$2"
        shift 2
        ;;
      *)
        echo "Unknown argument: $1" >&2
        exit 1
        ;;
    esac
  done
  if [[ -z "${tag}" ]]; then
    echo "--tag is required" >&2
    exit 1
  fi
  local tag_dir
  tag_dir="$(require_tag_dir "${tag}")"
  tail -n 80 -f "${tag_dir}/stdout.log"
}

stop_run() {
  local tag=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --tag)
        tag="$2"
        shift 2
        ;;
      *)
        echo "Unknown argument: $1" >&2
        exit 1
        ;;
    esac
  done
  if [[ -z "${tag}" ]]; then
    echo "--tag is required" >&2
    exit 1
  fi

  local tag_dir
  tag_dir="$(require_tag_dir "${tag}")"
  local pid_file="${tag_dir}/pid"
  if [[ ! -f "${pid_file}" ]]; then
    echo "No pid file found for tag=${tag}" >&2
    exit 1
  fi

  local pid
  pid="$(cat "${pid_file}")"
  if kill -0 "${pid}" >/dev/null 2>&1; then
    kill -TERM "${pid}"
    echo "Sent SIGTERM to PID ${pid}"
  else
    echo "PID ${pid} is not running"
  fi
}

command="${1:-}"
if [[ -z "${command}" ]]; then
  usage
  exit 1
fi
shift || true

case "${command}" in
  start) start_run "$@" ;;
  status) status_run "$@" ;;
  logs) logs_run "$@" ;;
  stop) stop_run "$@" ;;
  *) usage; exit 1 ;;
esac
