#!/usr/bin/env bash
# =============================================================================
# 下载指定的小模型（all_minilm_l6_v2, distilgpt2, qwen_2_0_5b_instruct, roberta_large, flan_t5_3b）
# 失败不中断，继续下载后续模型
# =============================================================================

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

# 代理配置
PROXY_HOST="127.0.0.1"
PROXY_PORT="7890"

# HuggingFace 官方站
HF_OFFICIAL="https://huggingface.co"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/small_models_$(date '+%Y%m%d_%H%M%S').log"

# =============================================================================
# 函数定义
# =============================================================================

check_hf_access() {
    echo "----------------------------------------"
    echo "  检测 HuggingFace 官方站连接状态..."
    echo "----------------------------------------"

    local try_url="${HF_OFFICIAL}"
    local max_retries=3
    local retry_delay=3

    for ((i=1; i<=max_retries; i++)); do
        echo "[$i/$max_retries] 尝试连接: ${try_url}"

        if curl -s -o /dev/null -w "%{http_code}" --max-time 10 "${try_url}" | grep -q "200\|301\|302"; then
            echo ""
            echo "✅ 成功连接到 HuggingFace 官方站"
            return 0
        fi

        if [[ $i -lt $max_retries ]]; then
            echo "⚠️  连接失败，${retry_delay}秒后重试..."
            sleep $retry_delay
        fi
    done

    echo ""
    echo "❌ 无法连接到 HuggingFace 官方站 (${try_url})"
    echo ""
    echo "可能原因："
    echo "  1. 网络不通"
    echo "  2. 代理未开启或配置错误"
    echo "  3. 需要使用代理"
    echo ""
    echo "建议解决方案："
    echo "  - 开启 Clash 代理"
    echo "  - 设置代理: export https_proxy=\"http://${PROXY_HOST}:${PROXY_PORT}\""
    echo "  - 或使用镜像站: export HF_ENDPOINT=\"https://hf-mirror.com\""
    echo ""
    return 1
}

check_proxy() {
    timeout 1 bash -c "echo >/dev/tcp/${PROXY_HOST}/${PROXY_PORT}" 2>/dev/null
}

start_background_download() {
    echo ""
    echo "========================================"
    echo "  开始后台下载"
    echo "========================================"
    echo ""

    # 检查 conda 环境
    if [[ "${CONDA_DEFAULT_ENV:-}" != "caiqiyue" ]]; then
        echo "[提示] 建议先激活 caiqiyue 环境: conda activate caiqiyue"
        echo ""
    fi

    # 检查代理
    if check_proxy; then
        echo "✅ 检测到代理已开启: ${PROXY_HOST}:${PROXY_PORT}"
        PROXY_ENABLED=true
    else
        echo "⚠️  未检测到代理: ${PROXY_HOST}:${PROXY_PORT}"
        echo "   如果下载慢或失败，请确保 Clash 代理已开启"
        PROXY_ENABLED=false
    fi

    # 设置环境变量
    export HF_HOME="${HF_HOME}"
    export PIP_CACHE_DIR="${PIP_CACHE_DIR}"
    export HF_TOKEN="${HF_TOKEN}"
    export HF_ENDPOINT="${HF_ENDPOINT:-${HF_OFFICIAL}}"
    export PYTHONPATH="${THESIS_DIR}:${PYTHONPATH:-}"

    # 设置代理（如果可用）
    if [[ "${PROXY_ENABLED}" == "true" ]]; then
        export HTTP_PROXY="http://${PROXY_HOST}:${PROXY_PORT}"
        export HTTPS_PROXY="http://${PROXY_HOST}:${PROXY_PORT}"
        echo "代理已设置: ${HTTP_PROXY}"
    fi

    echo "HF_HOME: ${HF_HOME}"
    echo "HF_ENDPOINT: ${HF_ENDPOINT}"
    echo "Python: $(which python)"
    echo ""

    # 写入启动信息到日志
    {
        echo "========================================"
        echo "后台下载开始: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "HF_ENDPOINT: ${HF_ENDPOINT}"
        echo "HF_TOKEN: 已设置"
        echo "目标模型: ${TARGET_MODELS[*]}"
        echo "========================================"
    } > "${LOG_FILE}"

    # 使用 nohup 后台执行
    nohup bash -c "
        source \"\${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}\"
        conda activate caiqiyue

        export HF_HOME=\"${HF_HOME}\"
        export PIP_CACHE_DIR=\"${PIP_CACHE_DIR}\"
        export HF_TOKEN=\"${HF_TOKEN}\"
        export HF_ENDPOINT=\"${HF_ENDPOINT}\"
        export PYTHONPATH=\"${THESIS_DIR}:\${PYTHONPATH:-}\"

        if [[ \"${PROXY_ENABLED}\" == \"true\" ]]; then
            export HTTP_PROXY=\"http://${PROXY_HOST}:${PROXY_PORT}\"
            export HTTPS_PROXY=\"http://${PROXY_HOST}:${PROXY_PORT}\"
        fi

        echo \"环境初始化完成，PID: \$\$\"
        echo \"开始下载...\"
        echo \"\"

        for MODEL_NAME in \"\${MODELS[@]}\"; do
            echo \"----------------------------------------\"
            echo \"开始下载: \${MODEL_NAME}\"
            echo \"时间: \$(date '+%Y-%m-%d %H:%M:%S')\"
            echo \"----------------------------------------\"

            {
                echo \"开始下载: \${MODEL_NAME} - \$(date '+%Y-%m-%d %H:%M:%S')\"
            } >> \"${LOG_FILE}\"

            python -c \"
import sys
import traceback
sys.path.insert(0, '${THESIS_DIR}')

try:
    from thesis_platform.model_downloaders.controller import download_models

    print('正在下载: \${MODEL_NAME}')
    print('')

    report = download_models(names=['\${MODEL_NAME}'], include_optional=False, include_large=False)

    counts = report['counts']
    print('')
    print('=== 下载结果 ===')
    print(f'模型: \${MODEL_NAME}')
    print(f'总计: {counts['total']}')
    print(f'成功: {counts['downloaded']}')
    print(f'跳过: {counts['skipped']}')
    print(f'失败: {counts['failed']}')

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
    print(f'模型: \${MODEL_NAME}')
    print('状态: ❌ 下载失败')
    print(f'错误: {str(e)}')
    traceback.print_exc()
    sys.exit(1)
\" 2>&1 | tee -a \"${LOG_FILE}\"

            EXIT_CODE=\${PIPESTATUS[0]}
            echo \"\${MODEL_NAME} 退出码: \${EXIT_CODE}\" >> \"${LOG_FILE}\"

            if [[ \${EXIT_CODE} -eq 0 ]]; then
                echo \"✅ \${MODEL_NAME} 处理完成\"
            else
                echo \"⚠️  \${MODEL_NAME} 失败，继续下一个...\"
            fi
            echo \"\"
        done

        echo \"\"
        echo \"========================================\"
        echo \"  全部模型处理完成\"
        echo \"结束时间: \$(date '+%Y-%m-%d %H:%M:%S')\"
        echo \"========================================\"
        echo \"日志文件: ${LOG_FILE}\"
    " > "${LOG_FILE}.stdout" 2>&1 &

    local PID=$!
    echo "${PID}" > "${LOG_DIR}/small_models.pid"

    echo "✅ 后台下载已启动"
    echo "PID: ${PID}"
    echo "日志: ${LOG_FILE}.stdout"
    echo "模型日志: ${LOG_FILE}"
    echo ""
    echo "查看进度: tail -f ${LOG_FILE}"
    echo "查看完整输出: tail -f ${LOG_FILE}.stdout"
    echo "停止下载: kill ${PID}"
}

# =============================================================================
# 主流程
# =============================================================================

echo "========================================"
echo "  下载指定小模型"
echo "========================================"
echo ""
echo "目标模型: ${TARGET_MODELS[*]}"
echo "日志文件: ${LOG_FILE}"
echo ""

# 1. 先测试 HuggingFace 连接
if ! check_hf_access; then
    echo "❌ HuggingFace 连接测试失败，停止下载"
    echo ""
    echo "========================================"
    echo "  下载已取消"
    echo "========================================"
    exit 1
fi

# 2. 连接正常，开始后台下载
start_background_download

exit 0
