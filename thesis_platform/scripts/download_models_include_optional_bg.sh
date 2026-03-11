#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STATE_DIR="${REPO_ROOT}/thesis_platform/open_model"
LOG_FILE="${STATE_DIR}/download_models_include_optional.log"
PID_FILE="${STATE_DIR}/download_models_include_optional.pid"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
ACTION="${1:-start}"

mkdir -p "${STATE_DIR}"

is_running() {
  if [[ ! -f "${PID_FILE}" ]]; then
    return 1
  fi

  local pid
  pid="$(cat "${PID_FILE}")"
  if [[ -z "${pid}" ]]; then
    return 1
  fi

  kill -0 "${pid}" 2>/dev/null
}

start_download() {
  if is_running; then
    echo "Model download is already running. PID: $(cat "${PID_FILE}")"
    echo "Log file: ${LOG_FILE}"
    exit 0
  fi

  cd "${REPO_ROOT}"
  nohup "${PYTHON_BIN}" -m thesis_platform.scripts.download_models --include-optional >> "${LOG_FILE}" 2>&1 &
  local pid=$!
  echo "${pid}" > "${PID_FILE}"

  echo "Started model download in the background."
  echo "PID: ${pid}"
  echo "Python: ${PYTHON_BIN}"
  echo "Log file: ${LOG_FILE}"
}

show_status() {
  if is_running; then
    echo "Model download is running. PID: $(cat "${PID_FILE}")"
    echo "Log file: ${LOG_FILE}"
  else
    echo "Model download is not running."
    if [[ -f "${PID_FILE}" ]]; then
      echo "Removing stale PID file: ${PID_FILE}"
      rm -f "${PID_FILE}"
    fi
    if [[ -f "${LOG_FILE}" ]]; then
      echo "Last log file: ${LOG_FILE}"
    fi
  fi
}

stop_download() {
  if ! is_running; then
    echo "Model download is not running."
    rm -f "${PID_FILE}"
    exit 0
  fi

  local pid
  pid="$(cat "${PID_FILE}")"
  kill "${pid}"
  rm -f "${PID_FILE}"
  echo "Stopped model download. PID: ${pid}"
}

show_logs() {
  if [[ ! -f "${LOG_FILE}" ]]; then
    echo "Log file does not exist yet: ${LOG_FILE}"
    exit 0
  fi

  tail -n "${TAIL_LINES:-50}" "${LOG_FILE}"
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
  *)
    echo "Usage: $0 [start|status|stop|logs]"
    exit 1
    ;;
esac
