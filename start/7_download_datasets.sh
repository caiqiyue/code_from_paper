#!/usr/bin/env bash
# =============================================================================
# 7. 下载所有数据集（后台运行）
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
THESIS_DIR="${REPO_ROOT}/thesis_platform"
OUTPUT_DIR="${REPO_ROOT}/outputs"
LOG_DIR="${OUTPUT_DIR}/download_logs"
DATASET_LOG="${LOG_DIR}/datasets_$(date '+%Y%m%d_%H%M%S').log"

# 创建日志目录
mkdir -p "${LOG_DIR}"

echo "========================================"
echo "  7. 下载所有数据集"
echo "========================================"
echo ""
echo "日志将保存到: ${DATASET_LOG}"
echo "实时查看日志: tail -f ${DATASET_LOG}"
echo ""

# 检查 conda 环境
if [[ "${CONDA_DEFAULT_ENV:-}" != "caiqiyue" ]]; then
    echo "[提示] 建议先激活 caiqiyue 环境: conda activate caiqiyue"
    echo ""
fi

# 启动后台下载
echo "正在后台启动数据集下载..."
echo "开始时间: $(date)" > "${DATASET_LOG}"
echo "========================================" >> "${DATASET_LOG}"

nohup bash -c "
    source \"${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}\"
    conda activate caiqiyue

    export PYTHONPATH=\"${THESIS_DIR}:\${PYTHONPATH:-}\"

    python -c \"
import sys
sys.path.insert(0, '${THESIS_DIR}')
from thesis_platform.dataset_downloaders.controller import download_datasets

print('开始下载所有数据集...')
report = download_datasets(include_optional=True)
print('')
print('下载完成!')
print(f'总计: {report[\"counts\"][\"total\"]}')
print(f'成功: {report[\"counts\"][\"downloaded\"]}')
print(f'跳过: {report[\"counts\"][\"skipped\"]}')
print(f'失败: {report[\"counts\"][\"failed\"]}')
print('')
print('详细报告已保存到:')
print('  thesis_platform/datasets/download_report.json')
\" >> \"${DATASET_LOG}\" 2>&1

    echo \"结束时间: \$(date)\" >> \"${DATASET_LOG}\"
" > /dev/null 2>&1 &

PID=$!
echo "后台进程 PID: ${PID}"
echo "请使用 'tail -f ${DATASET_LOG}' 查看下载进度"
echo ""
echo "========================================"
echo "  数据集下载已在后台启动"
echo "========================================"