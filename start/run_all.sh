#!/usr/bin/env bash
# =============================================================================
# 运行所有测试脚本
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo ""
echo "########################################"
echo "  开始运行所有测试"
echo "########################################"
echo ""

# 激活 conda 环境（如果尚未激活）
if [[ "${CONDA_DEFAULT_ENV:-}" != "caiqiyue" ]]; then
    echo "[提示] 建议先运行: conda activate caiqiyue"
    echo ""
fi

# 按顺序运行测试脚本
for script in 1_check_env.sh 2_check_imports.sh 3_list_experiments.sh 4_run_simple_test.sh 5_check_models.sh; do
    echo ""
    echo ">>> 运行: ${script}"
    echo "========================================"
    bash "${SCRIPT_DIR}/${script}"
    echo ""
done

echo ""
echo "########################################"
echo "  所有测试完成!"
echo "########################################"
echo ""
echo "如果所有测试都显示 ✓，说明项目在 Linux 服务器上可以正常运行。"
echo ""
echo "下一步："
echo "  1. 准备数据集"
echo "  2. 运行实际实验"
echo ""