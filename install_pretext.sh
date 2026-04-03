#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="${SCRIPT_DIR}"
for candidate in \
    "/root/autodl-tmp/caiqiyue/code_from_paper" \
    "/root/caiqiyue/code_from_paper"; do
    if [[ -d "${candidate}" ]]; then
        DEFAULT_REPO_ROOT="${candidate}"
        break
    fi
done

REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
PRETEXT_DIR="${PRETEXT_DIR:-${REPO_ROOT}/PrE-Text}"

ENV_NAME="${ENV_NAME:-pretext}"
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
CLASH_HOST="${CLASH_HOST:-127.0.0.1}"
START_CLASH="${START_CLASH:-true}"

DATA_ROOT_DEFAULT="${DATA_ROOT_DEFAULT:-/root/autodl-tmp}"
if [[ ! -d "${DATA_ROOT_DEFAULT}" ]]; then
    DATA_ROOT_DEFAULT="${REPO_ROOT}"
fi
DATA_ROOT="${DATA_ROOT:-${DATA_ROOT_DEFAULT}}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-${DATA_ROOT}/conda-envs/${ENV_NAME}}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${DATA_ROOT}/.cache/pip}"
HF_HOME="${HF_HOME:-${DATA_ROOT}/.cache/huggingface}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

INSTALL_VLLM="${INSTALL_VLLM:-false}"
INSTALL_XFORMERS="${INSTALL_XFORMERS:-false}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-false}"
VLLM_SPEC="${VLLM_SPEC:-vllm}"
XFORMERS_SPEC="${XFORMERS_SPEC:-xformers}"
FLASH_ATTN_SPEC="${FLASH_ATTN_SPEC:-flash-attn}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-${PRETEXT_DIR}/requirements.txt}"
LOG_FILE="${1:-${REPO_ROOT}/install_pretext.log}"

# PrE-Text uses torch>=2.6 with CUDA 12.4
TORCH_SPEC="${TORCH_SPEC:-torch>=2.6}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers>=5.3}"

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
    if command -v nc >/dev/null 2>&1; then
        nc -z "${host}" "${port}" >/dev/null 2>&1
        return $?
    fi
    timeout 1 bash -c "cat < /dev/null > /dev/tcp/${host}/${port}" >/dev/null 2>&1
}

export_clash_proxy() {
    local http_proxy_url="http://${CLASH_HOST}:${CLASH_PROXY_PORT}"
    local socks_proxy_url="socks5://${CLASH_HOST}:${CLASH_PROXY_PORT}"

    export http_proxy="${http_proxy_url}"
    export https_proxy="${http_proxy_url}"
    export HTTP_PROXY="${http_proxy_url}"
    export HTTPS_PROXY="${http_proxy_url}"
    export all_proxy="${socks_proxy_url}"
    export ALL_PROXY="${socks_proxy_url}"
    log "Proxy exported to ${http_proxy_url}"
}

conda_env_name_exists() {
    "${CONDA_BIN}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"
}

remove_existing_conda_env() {
    log_section "Step 0: Remove existing pretext environment"

    local removed_any="false"

    if conda_env_name_exists; then
        log "Removing existing named conda environment: ${ENV_NAME}"
        run_logged "${CONDA_BIN}" env remove -n "${ENV_NAME}" -y
        removed_any="true"
    else
        log "No named conda environment called ${ENV_NAME}."
    fi

    if [[ -d "${CONDA_ENV_PREFIX}" ]]; then
        log "Removing existing prefixed conda environment: ${CONDA_ENV_PREFIX}"
        run_logged "${CONDA_BIN}" env remove -p "${CONDA_ENV_PREFIX}" -y
        removed_any="true"
    else
        log "No prefixed conda environment at ${CONDA_ENV_PREFIX}."
    fi

    if [[ "${removed_any}" == "false" ]]; then
        log "No existing pretext environment needed removal."
    fi
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
    log_section "Step 1: Ensure Clash is running and export port 7890"

    if [[ "${START_CLASH}" != "true" ]]; then
        log "START_CLASH=${START_CLASH}, skipping Clash startup and proxy export."
        return
    fi

    if port_open "${CLASH_HOST}" "${CLASH_PROXY_PORT}"; then
        log "Clash proxy is already reachable on port ${CLASH_PROXY_PORT}."
    else
        [[ -x "${CLASH_BIN}" ]] || fail "Clash binary not found: ${CLASH_BIN}"
        [[ -f "${CLASH_CONFIG}" ]] || fail "Clash config not found: ${CLASH_CONFIG}"
        log "Starting Clash from ${CLASH_DIR} ..."
        (
            cd "${CLASH_DIR}"
            nohup "${CLASH_BIN}" -d "${CLASH_DIR}" -f "${CLASH_CONFIG}" >/tmp/clash.log 2>&1 &
        )
        sleep 5

        if ! port_open "${CLASH_HOST}" "${CLASH_PROXY_PORT}" && ! port_open "${CLASH_HOST}" "${CLASH_CONTROL_PORT}"; then
            warn "Clash did not become ready on port ${CLASH_PROXY_PORT}. See /tmp/clash.log if needed."
            return
        fi
        log "Clash started successfully."
    fi

    export_clash_proxy
}

log_section "Start installation for ${ENV_NAME}"
log "Script directory: ${SCRIPT_DIR}"
log "Repository root: ${REPO_ROOT}"
log "PrE-Text directory: ${PRETEXT_DIR}"
log "Target Python version: ${PYTHON_VERSION}"
log "Data root: ${DATA_ROOT}"
log "Conda environment prefix: ${CONDA_ENV_PREFIX}"
log "pip cache dir: ${PIP_CACHE_DIR}"
log "HF cache dir: ${HF_HOME}"
log "Clash directory: ${CLASH_DIR}"
log "Clash proxy target: ${CLASH_HOST}:${CLASH_PROXY_PORT}"
log "Install vLLM: ${INSTALL_VLLM}"
log "Install xformers: ${INSTALL_XFORMERS}"
log "Install flash-attn: ${INSTALL_FLASH_ATTN}"
log "requirements file: ${REQUIREMENTS_FILE}"
log "Log file: ${LOG_FILE}"

[[ -d "${REPO_ROOT}" ]] || fail "Repository root does not exist: ${REPO_ROOT}"
[[ -d "${PRETEXT_DIR}" ]] || fail "PrE-Text directory does not exist: ${PRETEXT_DIR}"
[[ -f "${REQUIREMENTS_FILE}" ]] || fail "requirements file does not exist: ${REQUIREMENTS_FILE}"
mkdir -p "$(dirname "${LOG_FILE}")" 2>/dev/null || true
mkdir -p "$(dirname "${CONDA_ENV_PREFIX}")" "${PIP_CACHE_DIR}" "${TRANSFORMERS_CACHE}" 2>/dev/null || true

export PIP_CACHE_DIR
export HF_HOME
export TRANSFORMERS_CACHE

if [[ ! -x "${CONDA_BIN}" ]]; then
    ensure_command conda
    CONDA_BIN="$(command -v conda)"
fi
[[ -f "${CONDA_SH}" ]] || fail "Conda activation script not found: ${CONDA_SH}"

if command -v nvidia-smi >/dev/null 2>&1; then
    log_section "GPU probe"
    run_logged nvidia-smi
fi

remove_existing_conda_env
maybe_start_clash

log_section "Step 2: Recreate conda environment"

cd "${REPO_ROOT}"

run_logged "${CONDA_BIN}" create -p "${CONDA_ENV_PREFIX}" "python=${PYTHON_VERSION}" -y

source "${CONDA_SH}"
conda activate "${CONDA_ENV_PREFIX}"

PYTHON_BIN="${CONDA_ENV_PREFIX}/bin/python"
PIP_BIN="${CONDA_ENV_PREFIX}/bin/pip"

[[ -x "${PYTHON_BIN}" ]] || fail "Python binary not found: ${PYTHON_BIN}"
[[ -x "${PIP_BIN}" ]] || fail "pip binary not found: ${PIP_BIN}"
export PATH="${CONDA_ENV_PREFIX}/bin:${PATH}"

log_section "Step 3: Upgrade packaging toolchain"
run_logged "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
log "Active python: $(${PYTHON_BIN} --version)"
log "Active pip: $(${PIP_BIN} --version)"

log_section "Step 4: Install PyTorch with CUDA 12.4 support"

# PrE-Text requires torch>=2.6 with CUDA 12.4 (cu124)
if python_has_module "torch"; then
    log "Torch already available: $(resolve_package_version "torch")"
else
    log "Torch is missing; installing ${TORCH_SPEC} from ${TORCH_INDEX_URL}"
    run_logged "${PYTHON_BIN}" -m pip install "${TORCH_SPEC}" --index-url "${TORCH_INDEX_URL}"
fi

log_section "Step 5: Install PrE-Text direct dependencies (excluding torch and cuda libs - installed in Step 4)"

# Create a filtered requirements file that excludes torch and nvidia-cuda-*
# (these must come from the official torch wheel to match CUDA version)
FILTERED_REQS="/tmp/requirements_filtered.txt"
grep -vE "^(torch|nvidia-cuda-|nvidia-cublas|nvidia-cudnn|nvidia-cufft|nvidia-curand|nvidia-cusolver|nvidia-cusparse|nvidia-cusparselt|nvidia-nccl|nvidia-nvjitlink|nvidia-nvtx|nvidia-cupti|nvidia-cuda-nvrtc|nvidia-cuda-runtime|triton|xformers|vllm)==" \
    "${REQUIREMENTS_FILE}" > "${FILTERED_REQS}"

run_logged "${PYTHON_BIN}" -m pip install -r "${FILTERED_REQS}"

# Re-install torch to ensure correct version (in case requirements.txt had torch pinned)
log "Reinstalling torch to guarantee CUDA 12.4 build..."
run_logged "${PYTHON_BIN}" -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-deps

log_section "Step 6: Install optional acceleration packages"

OS_NAME="$(uname -s)"

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

if [[ "${INSTALL_VLLM}" == "true" ]]; then
    if [[ "${OS_NAME}" == "Linux" ]]; then
        log "Installing ${VLLM_SPEC} (optional vLLM for accelerated inference)."
        run_logged "${PYTHON_BIN}" -m pip install "${VLLM_SPEC}"
    else
        warn "vllm skipped because host OS is ${OS_NAME}."
    fi
else
    log "INSTALL_VLLM=false. Skipping vLLM installation."
fi

log_section "Step 7: Verify environment"

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
    ("vllm", "vllm"),
    ("wandb", "wandb"),
    ("trl", "trl"),
    ("cupy", "cupy-cuda12x"),
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
log "Environment prefix: ${CONDA_ENV_PREFIX}"
log "Activation command: conda activate ${CONDA_ENV_PREFIX}"
log "Verification command: cd ${PRETEXT_DIR} && python -m pretext_platform.scripts.run_pipeline --help"