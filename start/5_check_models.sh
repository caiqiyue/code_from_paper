#!/usr/bin/env bash
# =============================================================================
# 5. 检查模型文件是否已准备就绪
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
THESIS_DIR="${REPO_ROOT}/thesis_platform"

echo "========================================"
echo "  5. 检查模型文件"
echo "========================================"

# 检查 models 目录
MODELS_DIR="${REPO_ROOT}/thesis_platform/models"
if [[ -d "${MODELS_DIR}" ]]; then
    echo "模型目录: ${MODELS_DIR}"
    echo "目录内容:"
    for item in "${MODELS_DIR}"/*; do
        if [[ -d "${item}" ]]; then
            size=$(du -sh "${item}" 2>/dev/null | cut -f1 || echo "unknown")
            echo "  - $(basename "${item}") (${size})"
        elif [[ -f "${item}" ]]; then
            size=$(du -sh "${item}" 2>/dev/null | cut -f1 || echo "unknown")
            echo "  - $(basename "${item}") (${size})"
        fi
    done
else
    echo "模型目录不存在: ${MODELS_DIR}"
fi

echo ""
echo "检查模型元数据..."
python -c "
import sys
import os
sys.path.insert(0, '${THESIS_DIR}')

from thesis_platform.core import model_downloaders

print('已注册模型下载器:')
for name, cls in model_downloaders.MODEL_DOWNLOADERS.items():
    status = 'ready'
    if hasattr(cls, 'optional') and cls.optional:
        status = 'optional'
    print(f'  - {name}: {status}')
"

echo ""
echo "========================================"
echo "  模型检查完成"
echo "========================================"