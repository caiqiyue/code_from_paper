#!/usr/bin/env bash
# E1 新增基线正式实验脚本 — C4-only / Aug-PE / DP-Prompt（各 4 数据集 × 30 seeds，共 360 个）
# GPU: A6000 (index 1), 环境: pretext
# 前提：已通过 run_e1_new_baselines_smoke_sequential.sh 的 smoke 验证

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

# 生成 repeat30 configs
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 生成 repeat30 experiment configs..."
"$PYTHON_BIN" "$BASE/paper-new-round23/scripts/generate_e1_dpprompt_experiment_configs.py" --mode e1_dpprompt_seen_repeat30
"$PYTHON_BIN" "$BASE/paper-new-round23/scripts/generate_e1_c4only_experiment_configs.py" --mode e1_c4only_seen_repeat30
"$PYTHON_BIN" "$BASE/paper-new-round23/scripts/generate_e1_augpe_experiment_configs.py" --mode e1_augpe_seen_repeat30

echo "[$(date '+%Y-%m-%d %H:%M:%S')] E1 新基线正式实验启动，共 360 个实验（3 方法 × 4 数据集 × 30 seeds）"

run_mode e1_dpprompt_seen_repeat30
run_mode e1_c4only_seen_repeat30
run_mode e1_augpe_seen_repeat30

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] E1 新基线正式实验全部完成！"
echo ""
echo "结果汇总："
echo "  DP-Prompt: $BASE/paper-new-round23/logs/round23_e1_dpprompt_seen_repeat30_summary.tsv"
echo "  C4-only:   $BASE/paper-new-round23/logs/round23_e1_c4only_seen_repeat30_summary.tsv"
echo "  Aug-PE:    $BASE/paper-new-round23/logs/round23_e1_augpe_seen_repeat30_summary.tsv"
echo ""
echo "合并 E1 扩展对比表："
echo "  $PYTHON_BIN $BASE/paper-new-round23/scripts/merge_thesis_e1_extended_results.py --scale repeat30"
