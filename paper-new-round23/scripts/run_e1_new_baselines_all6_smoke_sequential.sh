#!/usr/bin/env bash
# E1 新增基线 all6 smoke 验证脚本 — C4-only / Aug-PE / DP-Prompt
# 6 数据集 × 1 seed × 3 方法，共 18 个实验
# GPU: A6000 (index 1), 环境: pretext

BASE="/mnt/public/caiqiyue_file/code_from_paper"
PYTHON_BIN="/home/k8smaster/anaconda3/envs/pretext/bin/python"
RUNNER="$BASE/paper-new-round23/scripts/round23_dynamic_experiment_runner.py"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

cd "$BASE"

COMMON_ARGS="--target-gpu-index 1 --min-free-gb-for-vllm 26 --gpu-wait-poll-seconds 60 --gpu-wait-timeout-seconds 43200 --max-attempts 3 --retry-delay-seconds 10"

run_mode() {
    local mode=$1
    echo ""
    echo "========================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动 mode: $mode"
    echo "========================================"
    "$PYTHON_BIN" "$RUNNER" --mode "$mode" $COMMON_ARGS
    local exit_code=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] mode $mode 完成，exit_code=$exit_code"
    return $exit_code
}

# 生成 all6 smoke configs
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 生成 all6 smoke experiment configs..."
"$PYTHON_BIN" "$BASE/paper-new-round23/scripts/generate_e1_dpprompt_experiment_configs.py" --mode e1_dpprompt_all6_smoke
"$PYTHON_BIN" "$BASE/paper-new-round23/scripts/generate_e1_c4only_experiment_configs.py" --mode e1_c4only_all6_smoke
"$PYTHON_BIN" "$BASE/paper-new-round23/scripts/generate_e1_augpe_experiment_configs.py" --mode e1_augpe_all6_smoke

echo "[$(date '+%Y-%m-%d %H:%M:%S')] E1 新基线 all6 smoke 验证启动，共 18 个实验（3 方法 × 6 数据集 × 1 seed）"

run_mode e1_dpprompt_all6_smoke
run_mode e1_c4only_all6_smoke
run_mode e1_augpe_all6_smoke

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] E1 新基线 all6 smoke 验证完成！"
