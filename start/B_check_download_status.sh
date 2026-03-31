#!/usr/bin/env bash
# =============================================================================
# B. 检查下载状态和日志
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
THESIS_DIR="${REPO_ROOT}/thesis_platform"
OUTPUT_DIR="${REPO_ROOT}/outputs"
LOG_DIR="${OUTPUT_DIR}/download_logs"

echo "========================================"
echo "  B. 检查下载状态和日志"
echo "========================================"

mkdir -p "${LOG_DIR}"

# 检查后台进程
echo ""
echo "----------------------------------------"
echo "  后台下载进程:"
echo "----------------------------------------"
if pgrep -f "download_datasets\|download_models" > /dev/null 2>&1; then
    echo "发现正在运行的下载进程:"
    ps aux | grep -E "download_datasets|download_models" | grep -v grep | head -5
else
    echo "没有正在运行的下载进程"
fi

# 查看下载日志
echo ""
echo "----------------------------------------"
echo "  下载日志文件:"
echo "----------------------------------------"
if [[ -d "${LOG_DIR}" ]]; then
    ls -lah "${LOG_DIR}"/*.log 2>/dev/null || echo "没有日志文件"
else
    echo "日志目录不存在: ${LOG_DIR}"
fi

# 查看最新日志内容
echo ""
echo "----------------------------------------"
echo "  最新日志内容 (最后 30 行):"
echo "----------------------------------------"
LATEST_LOG=$(ls -t "${LOG_DIR}"/*.log 2>/dev/null | head -1)
if [[ -n "${LATEST_LOG}" && -f "${LATEST_LOG}" ]]; then
    echo "文件: ${LATEST_LOG}"
    echo ""
    tail -30 "${LATEST_LOG}"
else
    echo "没有日志文件"
fi

# 检查数据集下载报告
echo ""
echo "----------------------------------------"
echo "  数据集下载报告:"
echo "----------------------------------------"
DATASET_REPORT="${THESIS_DIR}/datasets/download_report.json"
if [[ -f "${DATASET_REPORT}" ]]; then
    python -c "
import json
with open('${DATASET_REPORT}') as f:
    report = json.load(f)
print(f'生成时间: {report.get(\"generated_at\", \"N/A\")}')
print(f'总计: {report[\"counts\"][\"total\"]}')
print(f'成功: {report[\"counts\"][\"downloaded\"]}')
print(f'跳过: {report[\"counts\"][\"skipped\"]}')
print(f'失败: {report[\"counts\"][\"failed\"]}')
"
else
    echo "数据集下载报告不存在"
fi

# 检查模型下载报告
echo ""
echo "----------------------------------------"
echo "  模型下载报告:"
echo "----------------------------------------"
MODEL_REPORT="${THESIS_DIR}/models/download_report.json"
if [[ -f "${MODEL_REPORT}" ]]; then
    python -c "
import json
with open('${MODEL_REPORT}') as f:
    report = json.load(f)
print(f'生成时间: {report.get(\"generated_at\", \"N/A\")}')
print(f'总计: {report[\"counts\"][\"total\"]}')
print(f'成功: {report[\"counts\"][\"downloaded\"]}')
print(f'跳过: {report[\"counts\"][\"skipped\"]}')
print(f'失败: {report[\"counts\"][\"failed\"]}')
"
else
    echo "模型下载报告不存在"
fi

echo ""
echo "========================================"
echo "  状态检查完成"
echo "========================================"