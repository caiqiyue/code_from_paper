#!/usr/bin/env bash
# E7 消融实验顺序启动脚本 — 全 4 个 mode，共 100 个实验
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

echo "[$(date '+%Y-%m-%d %H:%M:%S')] E7 实验启动，共 100 个实验（4 个 mode）"

run_mode e7_no_dataset_seen_repeat10
run_mode e7_no_dataset_unseen_repeat10
run_mode e7_no_coverage_seen_repeat5
run_mode e7_no_redundancy_seen_repeat5

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 全部 E7 实验完成！"
