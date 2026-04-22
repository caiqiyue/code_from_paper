#!/bin/bash
set -euo pipefail
pkill -f 'sn_c5_jobs_eps05' || true
pkill -f 'python -m thesis_platform.scripts.run_experiment --config thesis_platform/configs/experiments/single_node_formal/sn_c5_jobs_eps05.yaml' || true
sleep 2
echo AFTER_KILL
pgrep -af 'sn_c5_jobs_eps05|thesis_platform.scripts.run_experiment|pretext_platform.scripts.run_pipeline|run_eval_small' || true
echo GPU
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
