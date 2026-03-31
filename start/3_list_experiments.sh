#!/usr/bin/env bash
# =============================================================================
# 3. 列出所有可用实验
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
THESIS_DIR="${REPO_ROOT}/thesis_platform"

echo "========================================"
echo "  3. 列出所有可用实验"
echo "========================================"

export PYTHONPATH="${THESIS_DIR}:${PYTHONPATH:-}"

python -c "
import sys
sys.path.insert(0, '${THESIS_DIR}')

from thesis_platform.core import experiment_runner

print('可用实验:')
for name, cls in experiment_runner.EXPERIMENTS.items():
    print(f'  - {name}')
    if hasattr(cls, '__doc__') and cls.__doc__:
        doc = cls.__doc__.strip().split('\n')[0]
        print(f'    {doc}')
"

echo ""
echo "========================================"
echo "  实验列表获取完成"
echo "========================================"