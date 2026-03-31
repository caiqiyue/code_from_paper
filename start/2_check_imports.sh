#!/usr/bin/env bash
# =============================================================================
# 2. 检查 thesis_platform 模块是否可以正常导入
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
THESIS_DIR="${REPO_ROOT}/thesis_platform"

echo "========================================"
echo "  2. 检查 thesis_platform 导入"
echo "========================================"

export PYTHONPATH="${THESIS_DIR}:${PYTHONPATH:-}"

echo "THESIS_DIR: ${THESIS_DIR}"
echo "PYTHONPATH: ${PYTHONPATH}"

# 逐个检查核心模块
echo ""
echo "检查核心模块..."

python -c "
import sys
sys.path.insert(0, '${THESIS_DIR}')

# 检查基础模块
print('  - testing import thesis_platform.core.logging_utils')
from thesis_platform.core import logging_utils
print('    ✓ logging_utils')

print('  - testing import thesis_platform.core.experiment_runner')
from thesis_platform.core import experiment_runner
print('    ✓ experiment_runner')

print('  - testing import thesis_platform.core.model_downloaders')
from thesis_platform.core import model_downloaders
print('    ✓ model_downloaders')

print('  - testing import thesis_platform.core.privacy')
from thesis_platform.core import privacy
print('    ✓ privacy')

print('  - testing import thesis_platform.core.training')
from thesis_platform.core import training
print('    ✓ training')
"

echo ""
echo "检查模型下载器注册..."
python -c "
import sys
sys.path.insert(0, '${THESIS_DIR}')
from thesis_platform.core.model_downloaders import MODEL_DOWNLOADERS
print('✓ 已注册模型下载器数量:', len(MODEL_DOWNLOADERS))
for name, cls in list(MODEL_DOWNLOADERS.items())[:5]:
    print(f'  - {name}')
if len(MODEL_DOWNLOADERS) > 5:
    print(f'  ... 还有 {len(MODEL_DOWNLOADERS) - 5} 个')
"

echo ""
echo "========================================"
echo "  thesis_platform 导入检查完成"
echo "========================================"