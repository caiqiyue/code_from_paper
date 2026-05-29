#!/usr/bin/env bash
# 等待 E6 主跑结束，自动重跑 30 个 congressional 实验

BASE="/mnt/public/caiqiyue_file/code_from_paper"
PYTHON_BIN="${PYTHON_BIN:-/home/k8smaster/anaconda3/envs/pretext/bin/python}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

FULL_MANIFEST="$BASE/paper-new-round-14/configs/experiments/single_node_tuning_e6_round14_budget_sweep/repeat2_180/round14_lineage_e6_repeat2_180_manifest.tsv"
RERUN_MANIFEST="$BASE/paper-new-round-14/configs/experiments/single_node_tuning_e6_round14_budget_sweep/repeat2_180/round14_lineage_e6_congressional_rerun_manifest.tsv"
RERUN_SUMMARY="$BASE/paper-new-round-14/logs/round14_lineage_e6_congressional_rerun_summary.tsv"
RERUN_LOGDIR="$BASE/paper-new-round-14/logs/round14_lineage_e6_congressional_rerun_logs"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 监控 E6 进程..."

# 等待 E6 主进程退出
while pgrep -f 'run_round14_lineage_e6_budget_sweep_repeat2_180_sequential' > /dev/null 2>&1; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] E6 仍在运行，60s 后再检查..."
    sleep 60
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] E6 主跑已结束，构建 congressional 专用 manifest..."

# 提取 congressional 行（header + 30行数据）
head -1 "$FULL_MANIFEST" > "$RERUN_MANIFEST"
grep 'congressional' "$FULL_MANIFEST" >> "$RERUN_MANIFEST"

RERUN_COUNT=$(grep -c 'congressional' "$RERUN_MANIFEST")
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 已提取 $RERUN_COUNT 个 congressional 实验，开始重跑..."

cd "$BASE"

"$PYTHON_BIN" paper-new-round-14/scripts/run_round14_lineage_manifest.py \
  --manifest-path "$RERUN_MANIFEST" \
  --summary-path "$RERUN_SUMMARY" \
  --log-dir "$RERUN_LOGDIR" \
  --python-executable "$PYTHON_BIN" \
  --max-attempts 3 \
  --retry-delay-seconds 10 \
  --min-free-gb-for-vllm 26 \
  --gpu-wait-poll-seconds 60 \
  --gpu-wait-timeout-seconds 43200 \
  --target-gpu-index "${CUDA_VISIBLE_DEVICES}" \
  --reset-summary

echo "[$(date '+%Y-%m-%d %H:%M:%S')] congressional 重跑完成！"
echo "结果在: $RERUN_SUMMARY"
