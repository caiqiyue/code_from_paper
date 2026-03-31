#!/usr/bin/env bash
# =============================================================================
# 6. 列出所有可用的数据集和模型
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
THESIS_DIR="${REPO_ROOT}/thesis_platform"

echo "========================================"
echo "  6. 列出所有可用数据集和模型"
echo "========================================"

export PYTHONPATH="${THESIS_DIR}:${PYTHONPATH:-}"

# 列出所有数据集
echo ""
echo "----------------------------------------"
echo "  可用数据集:"
echo "----------------------------------------"
python -c "
import sys
sys.path.insert(0, '${THESIS_DIR}')
from thesis_platform.dataset_downloaders.controller import list_dataset_downloaders

datasets = list_dataset_downloaders()
print(f'共 {len(datasets)} 个数据集:\n')
for ds in datasets:
    optional_tag = ' [optional]' if ds['optional'] else ''
    print(f'  - {ds[\"name\"]}{optional_tag}')
    print(f'    {ds[\"description\"]}')
    print()
"

# 列出所有模型
echo ""
echo "----------------------------------------"
echo "  可用模型:"
echo "----------------------------------------"
python -c "
import sys
sys.path.insert(0, '${THESIS_DIR}')
from thesis_platform.model_downloaders.controller import list_model_downloaders

models = list_model_downloaders()
print(f'共 {len(models)} 个模型:\n')
for m in models:
    size_info = f' ({m[\"parameter_count_billions\"]}B)' if m['parameter_count_billions'] else ''
    optional_tag = ' [optional]' if m['optional'] else ''
    large_tag = ' [large]' if m['large_model'] else ''
    mirror_tag = ' [mirror only]' if m.get('community_mirror_only') else ''
    print(f'  - {m[\"name\"]}{size_info}{optional_tag}{large_tag}{mirror_tag}')
    if m.get('default_repo_id'):
        print(f'    repo: {m[\"default_repo_id\"]}')
    print()
"

echo ""
echo "========================================"
echo "  列表获取完成"
echo "========================================"
echo ""
echo "说明:"
echo "  [optional]  - 可选数据集/模型"
echo "  [large]     - 大模型 (>15B参数)"
echo "  [mirror only] - 只能从社区镜像下载"