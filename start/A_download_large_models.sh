#!/usr/bin/env bash
# =============================================================================
# A. 下载大模型 (>15B参数，需要更多磁盘空间和内存)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
THESIS_DIR="${REPO_ROOT}/thesis_platform"
OUTPUT_DIR="${REPO_ROOT}/outputs"
LOG_DIR="${OUTPUT_DIR}/download_logs"
MODEL_LOG="${LOG_DIR}/large_models_$(date '+%Y%m%d_%H%M%S').log"

# 创建日志目录
mkdir -p "${LOG_DIR}"

echo "========================================"
echo "  A. 下载大模型 (>15B参数)"
echo "========================================"
echo ""
echo "注意: 大模型需要更多下载时间和磁盘空间!"
echo "日志将保存到: ${MODEL_LOG}"
echo ""

# 检查 conda 环境
if [[ "${CONDA_DEFAULT_ENV:-}" != "caiqiyue" ]]; then
    echo "[提示] 建议先激活 caiqiyue 环境: conda activate caiqiyue"
    echo ""
fi

# 先列出大模型
echo "----------------------------------------"
echo "  将要下载的大模型:"
echo "----------------------------------------"
python -c "
import sys
sys.path.insert(0, '${THESIS_DIR}')
from thesis_platform.model_downloaders.controller import list_model_downloaders

models = list_model_downloaders()
large_models = [m for m in models if m['large_model']]
print(f'共 {len(large_models)} 个大模型:\n')
for m in large_models:
    print(f'  - {m[\"name\"]} ({m[\"parameter_count_billions\"]}B)')
    print(f'    {m.get(\"default_repo_id\", \"N/A\")}')
    print()
"

read -p "确认下载这些大模型? (y/N): " -r
if [[ ! "${REPLY}" =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "正在后台启动大模型下载..."
echo "开始时间: $(date)" > "${MODEL_LOG}"
echo "========================================" >> "${MODEL_LOG}"

nohup bash -c "
    source \"${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}\"
    conda activate caiqiyue

    export PYTHONPATH=\"${THESIS_DIR}:\${PYTHONPATH:-}\"
    export HF_HOME=\"${HF_HOME:-/root/autodl-tmp/.cache/huggingface}\"

    python -c \"
import sys
sys.path.insert(0, '${THESIS_DIR}')
from thesis_platform.model_downloaders.controller import download_models

print('开始下载所有大模型...')
report = download_models(include_optional=True, include_large=True)
print('')
print('下载完成!')
print(f'总计: {report[\"counts\"][\"total\"]}')
print(f'成功: {report[\"counts\"][\"downloaded\"]}')
print(f'跳过: {report[\"counts\"][\"skipped\"]}')
print(f'失败: {report[\"counts\"][\"failed\"]}')
\" >> \"${MODEL_LOG}\" 2>&1

    echo \"结束时间: \$(date)\" >> \"${MODEL_LOG}\"
" > /dev/null 2>&1 &

PID=$!
echo "后台进程 PID: ${PID}"
echo "请使用 'tail -f ${MODEL_LOG}' 查看下载进度"
echo ""
echo "========================================"
echo "  大模型下载已在后台启动"
echo "========================================"