#!/usr/bin/env bash
# =============================================================================
# 9. 下载指定的数据集或模型
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
THESIS_DIR="${REPO_ROOT}/thesis_platform"
OUTPUT_DIR="${REPO_ROOT}/outputs"
LOG_DIR="${OUTPUT_DIR}/download_logs"

mkdir -p "${LOG_DIR}"

# 检查参数
if [[ $# -lt 2 ]]; then
    echo "========================================"
    echo "  9. 下载指定的数据集或模型"
    echo "========================================"
    echo ""
    echo "用法:"
    echo "  bash start/9_download_specific.sh dataset <数据集名称> [数据集名称2 ...]"
    echo "  bash start/9_download_specific.sh model <模型名称> [模型名称2 ...]"
    echo ""
    echo "示例:"
    echo "  bash start/9_download_specific.sh dataset gsm8k imdb"
    echo "  bash start/9_download_specific.sh model opt-125m distilgpt2"
    echo ""
    echo "查看所有可用名称: bash start/6_list_available.sh"
    exit 1
fi

TYPE="$1"
shift

if [[ "${TYPE}" != "dataset" && "${TYPE}" != "model" ]]; then
    echo "错误: 类型必须是 'dataset' 或 'model'"
    exit 1
fi

NAMES=("$@")
NAMES_STR=$(IFS=' '; echo "${NAMES[*]}")

LOG_FILE="${LOG_DIR}/${TYPE}s_$(date '+%Y%m%d_%H%M%S').log"

echo "========================================"
echo "  下载指定 ${TYPE}(s)"
echo "========================================"
echo ""
echo "类型: ${TYPE}"
echo "名称: ${NAMES_STR}"
echo "日志: ${LOG_FILE}"
echo ""

# 检查 conda 环境
if [[ "${CONDA_DEFAULT_ENV:-}" != "caiqiyue" ]]; then
    echo "[提示] 建议先激活 caiqiyue 环境: conda activate caiqiyue"
    echo ""
fi

# 启动后台下载
echo "开始时间: $(date)" > "${LOG_FILE}"
echo "类型: ${TYPE}" >> "${LOG_FILE}"
echo "名称: ${NAMES_STR}" >> "${LOG_FILE}"
echo "========================================" >> "${LOG_FILE}"

if [[ "${TYPE}" == "dataset" ]]; then
    nohup bash -c "
        source \"${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}\"
        conda activate caiqiyue
        export PYTHONPATH=\"${THESIS_DIR}:\${PYTHONPATH:-}\"

        python -c \"
import sys
sys.path.insert(0, '${THESIS_DIR}')
from thesis_platform.dataset_downloaders.controller import download_datasets

print('开始下载数据集: ${NAMES_STR}')
report = download_datasets(names=${NAMES}, include_optional=True)
print('')
print('下载完成!')
print(f'总计: {report[\"counts\"][\"total\"]}')
print(f'成功: {report[\"counts\"][\"downloaded\"]}')
print(f'跳过: {report[\"counts\"][\"skipped\"]}')
print(f'失败: {report[\"counts\"][\"failed\"]}')
\" >> \"${LOG_FILE}\" 2>&1
" > /dev/null 2>&1 &
else
    nohup bash -c "
        source \"${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}\"
        conda activate caiqiyue
        export PYTHONPATH=\"${THESIS_DIR}:\${PYTHONPATH:-}\"
        export HF_HOME=\"${HF_HOME:-/root/autodl-tmp/.cache/huggingface}\"

        python -c \"
import sys
sys.path.insert(0, '${THESIS_DIR}')
from thesis_platform.model_downloaders.controller import download_models

print('开始下载模型: ${NAMES_STR}')
report = download_models(names=${NAMES}, include_optional=True, include_large=False)
print('')
print('下载完成!')
print(f'总计: {report[\"counts\"][\"total\"]}')
print(f'成功: {report[\"counts\"][\"downloaded\"]}')
print(f'跳过: {report[\"counts\"][\"skipped\"]}')
print(f'失败: {report[\"counts\"][\"failed\"]}')
\" >> \"${LOG_FILE}\" 2>&1
" > /dev/null 2>&1 &
fi

PID=$!
echo "后台进程 PID: ${PID}"
echo "请使用 'tail -f ${LOG_FILE}' 查看下载进度"
echo ""
echo "========================================"
echo "  下载已在后台启动"
echo "========================================"