#!/usr/bin/env bash
# =============================================================================
# 1. 检查虚拟环境是否正确配置
# =============================================================================

set -e

echo "========================================"
echo "  1. 检查虚拟环境"
echo "========================================"

# 确保 conda 环境已激活
if [[ "${CONDA_DEFAULT_ENV:-}" != "caiqiyue" ]]; then
    echo "请先激活 caiqiyue 环境："
    echo "  conda activate caiqiyue"
    exit 1
fi

echo "✓ conda 环境: ${CONDA_DEFAULT_ENV}"
echo "✓ Python 路径: $(which python)"
echo "✓ Python 版本: $(python --version)"

# 检查 PyTorch 和 CUDA
python -c "
import torch
print('✓ PyTorch 版本:', torch.__version__)
print('✓ CUDA 可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✓ GPU 数量:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f'  - GPU {i}:', torch.cuda.get_device_name(i))
"

# 检查关键包
echo ""
echo "========================================"
echo "  检查关键依赖包"
echo "========================================"

for pkg in transformers datasets huggingface_hub peft opacus bitsandbytes accelerate sentence-transformers faiss-cpu tiktoken; do
    if python -c "import ${pkg//-/_}" 2>/dev/null; then
        ver=$(python -c "import ${pkg//-/_}; print(${pkg//-/_}.__version__)" 2>/dev/null || echo "unknown")
        echo "✓ ${pkg}: ${ver}"
    else
        echo "✗ ${pkg}: 未安装"
    fi
done

echo ""
echo "========================================"
echo "  环境检查完成"
echo "========================================"