#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STATE_DIR="${REPO_ROOT}/thesis_platform/open_model"
LOG_FILE="${STATE_DIR}/download_models_include_optional.log"
PID_FILE="${STATE_DIR}/download_models_include_optional.pid"
CLASH_DIR="${REPO_ROOT}/clash_for_linux"
CLASH_BIN="${CLASH_DIR}/clash"
CLASH_CONFIG="${CLASH_DIR}/config.yaml"
CLASH_ENV_FILE="${CLASH_DIR}/run.txt"
CLASH_LOG_FILE="${STATE_DIR}/clash_for_linux.log"
CLASH_PID_FILE="${STATE_DIR}/clash_for_linux.pid"
CLASH_HOST="${CLASH_HOST:-127.0.0.1}"
CLASH_PORT="${CLASH_PORT:-7890}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
START_CLASH="${START_CLASH:-1}"
ACTION="${1:-start}"

mkdir -p "${STATE_DIR}"

read_pid() {
  local pid_file="$1"
  if [[ ! -f "${pid_file}" ]]; then
    return 1
  fi
  cat "${pid_file}"
}

pid_file_running() {
  local pid_file="$1"
  if [[ ! -f "${pid_file}" ]]; then
    return 1
  fi

  local pid
  pid="$(read_pid "${pid_file}")"
  if [[ -z "${pid}" ]]; then
    return 1
  fi

  kill -0 "${pid}" 2>/dev/null
}

cleanup_stale_pid() {
  local pid_file="$1"
  if [[ -f "${pid_file}" ]] && ! pid_file_running "${pid_file}"; then
    rm -f "${pid_file}"
  fi
}

proxy_port_open() {
  bash -lc "exec 3<>/dev/tcp/${CLASH_HOST}/${CLASH_PORT}" >/dev/null 2>&1
}

wait_for_proxy() {
  local attempts="${1:-30}"

  for ((i=1; i<=attempts; i++)); do
    if proxy_port_open; then
      return 0
    fi
    sleep 1
  done

  return 1
}

load_proxy_env() {
  if [[ ! -f "${CLASH_ENV_FILE}" ]]; then
    echo "Missing proxy environment file: ${CLASH_ENV_FILE}" >&2
    exit 1
  fi

  set +u
  # shellcheck disable=SC1090
  source "${CLASH_ENV_FILE}"
  set -u

  if [[ -n "${http_proxy:-}" ]]; then
    export http_proxy
    export HTTP_PROXY="${HTTP_PROXY:-${http_proxy}}"
  fi
  if [[ -n "${https_proxy:-}" ]]; then
    export https_proxy
    export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy}}"
  fi
  if [[ -n "${all_proxy:-}" ]]; then
    export all_proxy
    export ALL_PROXY="${ALL_PROXY:-${all_proxy}}"
  fi
}

start_clash() {
  if [[ "${START_CLASH}" != "1" ]]; then
    echo "Skipping Clash startup because START_CLASH=${START_CLASH}."
    return 0
  fi

  cleanup_stale_pid "${CLASH_PID_FILE}"

  if pid_file_running "${CLASH_PID_FILE}"; then
    echo "Clash is already running. PID: $(read_pid "${CLASH_PID_FILE}")"
    echo "Clash log file: ${CLASH_LOG_FILE}"
    return 0
  fi

  if proxy_port_open; then
    echo "Proxy port ${CLASH_HOST}:${CLASH_PORT} is already reachable."
    echo "Assuming Clash is already managed outside this script."
    return 0
  fi

  if [[ ! -x "${CLASH_BIN}" ]]; then
    echo "Clash executable not found or not executable: ${CLASH_BIN}" >&2
    exit 1
  fi
  if [[ ! -f "${CLASH_CONFIG}" ]]; then
    echo "Missing Clash config: ${CLASH_CONFIG}" >&2
    exit 1
  fi

  cd "${REPO_ROOT}"
  nohup "${CLASH_BIN}" -d "${CLASH_DIR}" >> "${CLASH_LOG_FILE}" 2>&1 &
  local pid=$!
  echo "${pid}" > "${CLASH_PID_FILE}"

  if ! wait_for_proxy 30; then
    echo "Clash did not become ready on ${CLASH_HOST}:${CLASH_PORT}." >&2
    echo "Check the Clash log: ${CLASH_LOG_FILE}" >&2
    kill "${pid}" 2>/dev/null || true
    rm -f "${CLASH_PID_FILE}"
    exit 1
  fi

  echo "Started Clash in the background."
  echo "Clash PID: ${pid}"
  echo "Clash log file: ${CLASH_LOG_FILE}"
}

start_download() {
  cleanup_stale_pid "${PID_FILE}"

  if pid_file_running "${PID_FILE}"; then
    echo "Model download is already running. PID: $(read_pid "${PID_FILE}")"
    echo "Log file: ${LOG_FILE}"
    exit 0
  fi

  start_clash
  load_proxy_env

  cd "${REPO_ROOT}"
  nohup "${PYTHON_BIN}" -m thesis_platform.scripts.download_models --include-optional >> "${LOG_FILE}" 2>&1 &
  local pid=$!
  echo "${pid}" > "${PID_FILE}"

  echo "Started model download in the background."
  echo "PID: ${pid}"
  echo "Python: ${PYTHON_BIN}"
  if [[ -n "${http_proxy:-}" ]]; then
    echo "http_proxy: ${http_proxy}"
  fi
  if [[ -n "${https_proxy:-}" ]]; then
    echo "https_proxy: ${https_proxy}"
  fi
  echo "Log file: ${LOG_FILE}"
}

show_status() {
  cleanup_stale_pid "${PID_FILE}"
  cleanup_stale_pid "${CLASH_PID_FILE}"

  if pid_file_running "${PID_FILE}"; then
    echo "Model download is running. PID: $(read_pid "${PID_FILE}")"
  else
    echo "Model download is not running."
  fi
  echo "Model log file: ${LOG_FILE}"

  if pid_file_running "${CLASH_PID_FILE}"; then
    echo "Clash is running. PID: $(read_pid "${CLASH_PID_FILE}")"
  elif proxy_port_open; then
    echo "Proxy port ${CLASH_HOST}:${CLASH_PORT} is reachable."
    echo "Clash appears to be running, but this script is not managing its PID."
  else
    echo "Clash is not running."
  fi
  echo "Clash log file: ${CLASH_LOG_FILE}"
}

stop_download() {
  cleanup_stale_pid "${PID_FILE}"
  cleanup_stale_pid "${CLASH_PID_FILE}"

  if pid_file_running "${PID_FILE}"; then
    local pid
    pid="$(read_pid "${PID_FILE}")"
    kill "${pid}"
    echo "Stopped model download. PID: ${pid}"
  else
    echo "Model download is not running."
  fi
  rm -f "${PID_FILE}"

  if pid_file_running "${CLASH_PID_FILE}"; then
    local clash_pid
    clash_pid="$(read_pid "${CLASH_PID_FILE}")"
    kill "${clash_pid}"
    echo "Stopped Clash. PID: ${clash_pid}"
    rm -f "${CLASH_PID_FILE}"
  elif [[ -f "${CLASH_PID_FILE}" ]]; then
    rm -f "${CLASH_PID_FILE}"
  else
    echo "Clash PID is not managed by this script, or Clash is not running."
  fi
}

show_logs() {
  if [[ ! -f "${LOG_FILE}" ]]; then
    echo "Model log file does not exist yet: ${LOG_FILE}"
    exit 0
  fi

  tail -n "${TAIL_LINES:-50}" "${LOG_FILE}"
}

show_clash_logs() {
  if [[ ! -f "${CLASH_LOG_FILE}" ]]; then
    echo "Clash log file does not exist yet: ${CLASH_LOG_FILE}"
    exit 0
  fi

  tail -n "${TAIL_LINES:-50}" "${CLASH_LOG_FILE}"
}

case "${ACTION}" in
  start)
    start_download
    ;;
  status)
    show_status
    ;;
  stop)
    stop_download
    ;;
  logs)
    show_logs
    ;;
  clash-logs)
    show_clash_logs
    ;;
  *)
    echo "Usage: $0 [start|status|stop|logs|clash-logs]"
    exit 1
    ;;
esac
