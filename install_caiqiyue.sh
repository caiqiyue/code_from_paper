#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_REPO_ROOT_DEFAULT="/root/caiqiyue/code_from_paper"
if [[ -d "${SERVER_REPO_ROOT_DEFAULT}" ]]; then
    DEFAULT_REPO_ROOT="${SERVER_REPO_ROOT_DEFAULT}"
else
    DEFAULT_REPO_ROOT="${SCRIPT_DIR}"
fi

REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
THESIS_DIR="${THESIS_DIR:-${REPO_ROOT}/thesis_platform}"

ENV_NAME="${ENV_NAME:-caiqiyue}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_SH="${CONDA_SH:-${CONDA_ROOT}/etc/profile.d/conda.sh}"
CONDA_BIN="${CONDA_BIN:-${CONDA_ROOT}/bin/conda}"
CLASH_DIR="${CLASH_DIR:-${REPO_ROOT}/clash_for_linux}"
CLASH_BIN="${CLASH_BIN:-${CLASH_DIR}/clash}"
CLASH_CONFIG="${CLASH_CONFIG:-${CLASH_DIR}/config.yaml}"
CLASH_ENV_FILE="${CLASH_ENV_FILE:-${CLASH_DIR}/run.txt}"
CLASH_CONTROL_PORT="${CLASH_CONTROL_PORT:-9090}"
CLASH_PROXY_PORT="${CLASH_PROXY_PORT:-7890}"
START_CLASH="${START_CLASH:-true}"

INSTALL_VLLM="${INSTALL_VLLM:-true}"
INSTALL_XFORMERS="${INSTALL_XFORMERS:-false}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-false}"
VLLM_SPEC="${VLLM_SPEC:-vllm}"
XFORMERS_SPEC="${XFORMERS_SPEC:-xformers}"
FLASH_ATTN_SPEC="${FLASH_ATTN_SPEC:-flash-attn}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
LOG_FILE="${1:-${REPO_ROOT}/install_caiqiyue.log}"

# Synced against thesis_platform code paths, thesis_platform/requirements.txt
# and the local `conda list -n experiment` result on 2026-03-30.
TORCH_SPEC="${TORCH_SPEC:-torch==2.7.1}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers>=5.3.0}"
BASE_DEPS=(
    "accelerate>=1.13.0"
    "datasets>=4.7.0"
    "huggingface-hub>=1.6.0"
    "sentencepiece>=0.2.1"
    "tokenizers>=0.22.2"
    "safetensors>=0.7.0"
    "numpy>=2.4.3"
    "pandas>=3.0.1"
    "scikit-learn>=1.8.0"
    "scipy>=1.17.1"
    "tqdm>=4.67.3"
    "PyYAML>=6.0.3"
    "sentence-transformers>=5.2.3"
    "peft>=0.18.1"
    "opacus>=1.5.4"
    "bitsandbytes>=0.49.2"
    "faiss-cpu>=1.13.2"
    "tiktoken>=0.12.0"
)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

warn() {
    log "WARN: $1"
}

fail() {
    log "ERROR: $1"
    exit 1
}

log_section() {
    log ""
    log "============================================"
    log "  $1"
    log "============================================"
}

run_logged() {
    log "RUN: $*"
    "$@" 2>&1 | tee -a "${LOG_FILE}"
}

ensure_command() {
    command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

port_open() {
    local host="$1"
    local port="$2"
    if ! command -v nc >/dev/null 2>&1; then
        return 1
    fi
    nc -z "${host}" "${port}" >/dev/null 2>&1
}

load_proxy_env() {
    if [[ -f "${CLASH_ENV_FILE}" ]]; then
        log "Loading proxy variables from ${CLASH_ENV_FILE}"
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
        return
    fi

    export HTTP_PROXY="http://127.0.0.1:${CLASH_PROXY_PORT}"
    export HTTPS_PROXY="http://127.0.0.1:${CLASH_PROXY_PORT}"
    export ALL_PROXY="socks5://127.0.0.1:${CLASH_PROXY_PORT}"
}

python_has_module() {
    local module_name="$1"
    "${PYTHON_BIN}" - <<PY >/dev/null 2>&1
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec("${module_name}") else 1)
PY
}

resolve_package_version() {
    local package_name="$1"
    "${PYTHON_BIN}" - <<PY
from importlib.metadata import version

print(version("${package_name}"))
PY
}

maybe_start_clash() {
    log_section "Step 0: Prepare optional Clash proxy"

    if [[ "${START_CLASH}" != "true" ]]; then
        log "START_CLASH=${START_CLASH}, skipping Clash startup."
        return
    fi

    if [[ ! -x "${CLASH_BIN}" ]]; then
        warn "Clash binary not found at ${CLASH_BIN}. Continuing without proxy."
        return
    fi

    if ! command -v nc >/dev/null 2>&1; then
        warn "netcat (nc) is not installed. Skipping Clash health check and proxy auto-start."
        return
    fi

    if port_open "127.0.0.1" "${CLASH_PROXY_PORT}"; then
        log "Clash proxy is already reachable on port ${CLASH_PROXY_PORT}."
    else
        log "Starting Clash from ${CLASH_DIR} ..."
        (
            cd "${CLASH_DIR}"
            nohup "${CLASH_BIN}" -d "${CLASH_DIR}" >/tmp/clash.log 2>&1 &
        )
        sleep 3

        if ! port_open "127.0.0.1" "${CLASH_PROXY_PORT}" && ! port_open "127.0.0.1" "${CLASH_CONTROL_PORT}"; then
            warn "Clash did not become ready. Continuing without proxy. See /tmp/clash.log if needed."
            return
        fi
        log "Clash started successfully."
    fi

    load_proxy_env
    log "Proxy exported: ${HTTP_PROXY:-<unset>}"
}

log_section "Start installation for ${ENV_NAME}"
log "Script directory: ${SCRIPT_DIR}"
log "Repository root: ${REPO_ROOT}"
log "thesis_platform directory: ${THESIS_DIR}"
log "Target Python version: ${PYTHON_VERSION}"
log "Install vLLM on Linux: ${INSTALL_VLLM}"
log "Install xformers: ${INSTALL_XFORMERS}"
log "Install flash-attn: ${INSTALL_FLASH_ATTN}"
log "vLLM spec: ${VLLM_SPEC}"
log "Log file: ${LOG_FILE}"

[[ -d "${REPO_ROOT}" ]] || fail "Repository root does not exist: ${REPO_ROOT}"
[[ -d "${THESIS_DIR}" ]] || fail "thesis_platform directory does not exist: ${THESIS_DIR}"
mkdir -p "$(dirname "${LOG_FILE}")"

if [[ ! -x "${CONDA_BIN}" ]]; then
    ensure_command conda
    CONDA_BIN="$(command -v conda)"
fi
[[ -f "${CONDA_SH}" ]] || fail "Conda activation script not found: ${CONDA_SH}"

if command -v nvidia-smi >/dev/null 2>&1; then
    log_section "GPU probe"
    run_logged nvidia-smi
fi

maybe_start_clash

log_section "Step 1: Recreate conda environment"

cd "${REPO_ROOT}"

if "${CONDA_BIN}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    log "Removing existing conda environment: ${ENV_NAME}"
    run_logged "${CONDA_BIN}" env remove -n "${ENV_NAME}" -y
else
    log "No existing conda environment named ${ENV_NAME}."
fi

run_logged "${CONDA_BIN}" create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y

CONDA_BASE="$("${CONDA_BIN}" info --base)"
source "${CONDA_SH}"
conda activate "${ENV_NAME}"

PYTHON_BIN="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
PIP_BIN="${CONDA_BASE}/envs/${ENV_NAME}/bin/pip"

[[ -x "${PYTHON_BIN}" ]] || fail "Python binary not found: ${PYTHON_BIN}"
[[ -x "${PIP_BIN}" ]] || fail "pip binary not found: ${PIP_BIN}"
export PATH="${CONDA_BASE}/envs/${ENV_NAME}/bin:${PATH}"

log_section "Step 2: Upgrade packaging toolchain"
run_logged "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
log "Active python: $(${PYTHON_BIN} --version)"
log "Active pip: $(${PIP_BIN} --version)"

log_section "Step 3: Install Linux server extras first when requested"
OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"

if [[ "${INSTALL_VLLM}" == "true" ]]; then
    if [[ "${OS_NAME}" != "Linux" ]]; then
        warn "vLLM install skipped because host OS is ${OS_NAME}, not Linux."
    elif [[ "${ARCH_NAME}" != "x86_64" && "${ARCH_NAME}" != "aarch64" ]]; then
        warn "vLLM install skipped because architecture ${ARCH_NAME} is unsupported by default wheel flow."
    else
        log "Installing ${VLLM_SPEC} in the fresh environment before explicit torch/transformers pinning."
        run_logged "${PYTHON_BIN}" -m pip install "${VLLM_SPEC}"
        if python_has_module "vllm"; then
            log "vLLM version: $(resolve_package_version "vllm")"
        else
            fail "vLLM install command succeeded but module import probe failed."
        fi
    fi
else
    log "INSTALL_VLLM=false, skipping vLLM."
fi

log_section "Step 4: Ensure PyTorch and Transformers are available"

if python_has_module "torch"; then
    log "Torch already available: $(resolve_package_version "torch")"
else
    log "Torch is missing; installing ${TORCH_SPEC} from ${TORCH_INDEX_URL}"
    run_logged "${PYTHON_BIN}" -m pip install "${TORCH_SPEC}" --index-url "${TORCH_INDEX_URL}"
fi

if python_has_module "transformers"; then
    log "Transformers already available: $(resolve_package_version "transformers")"
else
    log "Transformers is missing; installing ${TRANSFORMERS_SPEC}"
    run_logged "${PYTHON_BIN}" -m pip install "${TRANSFORMERS_SPEC}"
fi

log_section "Step 5: Install thesis_platform direct dependencies"
run_logged "${PYTHON_BIN}" -m pip install --upgrade "${BASE_DEPS[@]}"

if [[ "${INSTALL_XFORMERS}" == "true" ]]; then
    if [[ "${OS_NAME}" == "Linux" ]]; then
        log "Installing ${XFORMERS_SPEC} (optional Linux acceleration package)."
        run_logged "${PYTHON_BIN}" -m pip install "${XFORMERS_SPEC}"
    else
        warn "xformers skipped because host OS is ${OS_NAME}."
    fi
fi

if [[ "${INSTALL_FLASH_ATTN}" == "true" ]]; then
    if [[ "${OS_NAME}" == "Linux" ]]; then
        log "Installing ${FLASH_ATTN_SPEC} with --no-build-isolation."
        run_logged "${PYTHON_BIN}" -m pip install "${FLASH_ATTN_SPEC}" --no-build-isolation
    else
        warn "flash-attn skipped because host OS is ${OS_NAME}."
    fi
fi

log_section "Step 6: Verify environment"

run_logged "${PYTHON_BIN}" -m pip check

"${PYTHON_BIN}" - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
import importlib
import importlib.util
from importlib.metadata import version
import platform
import sys

print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print("=" * 60)

checks = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("datasets", "datasets"),
    ("accelerate", "accelerate"),
    ("peft", "peft"),
    ("opacus", "opacus"),
    ("bitsandbytes", "bitsandbytes"),
    ("faiss", "faiss-cpu"),
    ("sentence_transformers", "sentence-transformers"),
    ("tiktoken", "tiktoken"),
    ("vllm", "vllm"),
]

for module_name, package_name in checks:
    if importlib.util.find_spec(module_name) is None:
        print(f"{package_name}: not installed")
        continue
    try:
        installed_version = version(package_name)
    except Exception:
        installed_version = "installed"
    try:
        importlib.import_module(module_name)
        print(f"{package_name}: {installed_version}")
    except Exception as exc:
        print(f"{package_name}: installed as {installed_version}, import failed -> {exc}")

print("=" * 60)

try:
    import torch

    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            print(f"  GPU {index}: {torch.cuda.get_device_name(index)}")
except Exception as exc:
    print(f"torch runtime probe failed: {exc}")
PY

log_section "Installation complete"
log "Environment name: ${ENV_NAME}"
log "Activation command: conda activate ${ENV_NAME}"
log "Verification command: python -m unittest discover -s thesis_platform/tests -p 'test_thesis_platform*.py' -v"
