#!/bin/bash
set -u
base=/mnt/public/caiqiyue_file/code_from_paper/outputs/thesis_platform/smoke_single_node_tiny
echo '===== GPU ====='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo '===== PROC ====='
pgrep -af 'smoke_single_node_tiny|thesis_platform.scripts.run_experiment|pretext_platform.scripts.run_pipeline|run_eval_small' || true
echo '===== RUN STATE ====='
if [ -f "$base/run_state.json" ]; then
  cat "$base/run_state.json"
else
  echo missing_run_state
fi
echo '===== METRICS ====='
if [ -f "$base/metrics_summary.json" ]; then
  cat "$base/metrics_summary.json"
else
  echo missing_metrics_summary
fi
echo '===== FILES ====='
find "$base" -maxdepth 3 -type f | sort
