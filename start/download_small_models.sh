#!/usr/bin/env bash
# =============================================================================
# 下载指定的小模型（all_minilm_l6_v2, distilgpt2, qwen_2_0_5b_instruct, roberta_large, flan_t5_3b）
# 失败不中断，继续下载后续模型
# =============================================================================

# 不使用 set -e，让脚本能够处理失败并继续

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
THESIS_DIR="${REPO_ROOT}/thesis_platform"
OUTPUT_DIR="${REPO_ROOT}/outputs"
LOG_DIR="${OUTPUT_DIR}/download_logs"
HF_TOKEN="${HF_TOKEN:-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}"
PIP_CACHE_DIR="${DATA_ROOT:-/root/autodl-tmp}/.cache/pip"
HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"

# 目标模型列表
TARGET_MODELS=("all_minilm_l6_v2" "distilgpt2" "qwen_2_0_5b_instruct" "roberta_large" "flan_t5_3b")

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 生成日志文件名
LOG_FILE="${LOG_DIR}/small_models_$(date '+%Y%m%d_%H%M%S').log"

echo "========================================"
echo "  下载指定小模型"
echo "========================================"
echo ""
echo "目标模型: ${TARGET_MODELS[*]}"
echo "日志文件: ${LOG_FILE}"
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

# 记录开始时间到日志
{
    echo "========================================"
    echo "下载开始: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
} > "${LOG_FILE}"

# 逐个下载模型，失败跳过
for MODEL_NAME in "${TARGET_MODELS[@]}"; do
    echo "----------------------------------------"
    echo "  开始下载: ${MODEL_NAME}"
    echo "----------------------------------------"

    {
        echo ""
        echo "========================================"
        echo "开始下载: ${MODEL_NAME}"
        echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================"
    } | tee -a "${LOG_FILE}"

    # 使用 Python 下载，带完整的错误处理
    python -c "
import sys
import traceback
sys.path.insert(0, '${THESIS_DIR}')

try:
    from thesis_platform.model_downloaders.controller import download_models

    print('正在下载: ${MODEL_NAME}')
    print('')

    report = download_models(names=['${MODEL_NAME}'], include_optional=False, include_large=False)

    counts = report['counts']
    print('')
    print('=== 下载结果 ===')
    print(f'模型: ${MODEL_NAME}')
    print(f'总计: {counts[\"total\"]}')
    print(f'成功: {counts[\"downloaded\"]}')
    print(f'跳过: {counts[\"skipped\"]}')
    print(f'失败: {counts[\"failed\"]}')

    if report['results']:
        for r in report['results']:
            status = r.get('status', 'unknown')
            if status == 'downloaded':
                print('')
                print('状态: ✅ 下载成功')
                print(f'路径: {r.get(\"target_path\", \"N/A\")}')
                if r.get('disk_usage_bytes'):
                    size_mb = r['disk_usage_bytes'] / 1024 / 1024
                    print(f'大小: {size_mb:.2f} MiB')
            elif status == 'skipped':
                print('')
                print('状态: ⏭️  已存在，跳过')
            elif status == 'failed':
                print('')
                print('状态: ❌ 下载失败')
                if r.get('message'):
                    print(f'原因: {r[\"message\"]}')
                if r.get('error'):
                    print(f'错误: {r[\"error\"]}')
            else:
                print(f'状态: {status}')

    sys.exit(0) if counts['downloaded'] > 0 or counts['skipped'] > 0 else sys.exit(1)

except Exception as e:
    print('')
    print('=== 下载异常 ===')
    print(f'模型: ${MODEL_NAME}')
    print('状态: ❌ 下载失败')
    print(f'错误: {str(e)}')
    traceback.print_exc()
    sys.exit(1)
" 2>&1 | tee -a "${LOG_FILE}"

    DOWNLOAD_EXIT=${PIPESTATUS[0]}

    {
        echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "退出码: ${DOWNLOAD_EXIT}"
    } | tee -a "${LOG_FILE}"

    if [[ ${DOWNLOAD_EXIT} -eq 0 ]]; then
        echo "✅ ${MODEL_NAME} 处理完成"
    else
        echo "⚠️  ${MODEL_NAME} 失败，继续下一个..."
    fi

    echo ""
done

echo "========================================"
echo "  全部模型处理完成"
echo "========================================"
echo ""
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "日志文件: ${LOG_FILE}"

{
    echo "========================================"
    echo "下载结束: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
} >> "${LOG_FILE}"
