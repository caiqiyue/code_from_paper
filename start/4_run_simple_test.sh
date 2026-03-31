#!/usr/bin/env bash
# =============================================================================
# 4. 运行简单测试实验（不下载模型，不训练，仅验证 runner）
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
THESIS_DIR="${REPO_ROOT}/thesis_platform"

echo "========================================"
echo "  4. 运行简单测试实验"
echo "========================================"

export PYTHONPATH="${THESIS_DIR}:${PYTHONPATH:-}"

# 创建临时测试目录
TEST_OUTPUT_DIR="${REPO_ROOT}/outputs/test_simple_run"
mkdir -p "${TEST_OUTPUT_DIR}"

echo "测试输出目录: ${TEST_OUTPUT_DIR}"

python -c "
import sys
import os
sys.path.insert(0, '${THESIS_DIR}')
os.environ['OUTPUT_ROOT'] = '${TEST_OUTPUT_DIR}'

from thesis_platform.core import experiment_runner

print('测试 ExperimentRunner 初始化...')

# 创建 runner（不指定 experiment_id，使用默认测试）
runner = experiment_runner.ExperimentRunner(
    output_root='${TEST_OUTPUT_DIR}',
    experiment_id='test_simple'
)

print('✓ ExperimentRunner 创建成功')
print(f'✓ 实验输出目录: {runner.experiment_dir}')

# 检查目录是否创建
import os
if os.path.exists(runner.experiment_dir):
    print('✓ 实验目录已创建')
else:
    print('✗ 实验目录未创建')

# 列出实验目录内容
print('实验目录内容:')
for item in os.listdir(runner.experiment_dir):
    print(f'  - {item}')

print('')
print('========================================')
print('  简单测试实验完成')
print('========================================')
"