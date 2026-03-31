#!/usr/bin/env bash
# =============================================================================
# 下载所有模型（不含大模型，失败不中断）
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
THESIS_DIR="${REPO_ROOT}/thesis_platform"
OUTPUT_DIR="${REPO_ROOT}/outputs"
LOG_DIR="${OUTPUT_DIR}/download_logs"
HF_TOKEN="${HF_TOKEN:-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}"
PIP_CACHE_DIR="${DATA_ROOT:-/root/autodl-tmp}/.cache/pip"
HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 生成日志文件名
LOG_FILE="${LOG_DIR}/models_$(date '+%Y%m%d_%H%M%S').log"
PID_FILE="${LOG_DIR}/models_download.pid"

echo "========================================"
echo "  下载所有模型（不含大模型）"
echo "========================================"
echo ""
echo "日志文件: ${LOG_FILE}"
echo "PID 文件: ${PID_FILE}"
echo ""

# 检查 conda 环境
if [[ "${CONDA_DEFAULT_ENV:-}" != "caiqiyue" ]]; then
    echo "[提示] 建议先激活 caiqiyue 环境: conda activate caiqiyue"
    echo ""
fi

# 设置环境变量
export HF_HOME="${HF_HOME}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR}"
export HF_TOKEN="${HF_TOKEN}"
export PYTHONPATH="${THESIS_DIR}:${PYTHONPATH:-}"

echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "HF_HOME: ${HF_HOME}"
echo "HF_TOKEN: 已设置"
echo "Python: $(which python)"
echo "Python 版本: $(python --version)"
echo ""

# 启动后台下载
nohup bash -c "
    source \"${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}\"
    conda activate caiqiyue

    export HF_HOME=\"${HF_HOME}\"
    export PIP_CACHE_DIR=\"${PIP_CACHE_DIR}\"
    export HF_TOKEN=\"${HF_TOKEN}\"
    export PYTHONPATH=\"${THESIS_DIR}:\${PYTHONPATH:-}\"

    echo '开始下载所有模型...'
    echo ''

    python -m thesis_platform.scripts.download_models \
        --include-optional \
        2>&1

    echo ''
    echo '结束时间: \$(date '+%Y-%m-%d %H:%M:%S')'
" > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${PID_FILE}"

echo "后台进程已启动"
echo "PID: ${PID}"
echo ""
echo "========================================"
echo "  下载已在后台运行"
echo "========================================"
echo ""
echo "查看日志:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "查看进度（实时）:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "停止下载:"
echo "  kill \$(cat ${PID_FILE})"
echo ""