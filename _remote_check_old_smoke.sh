#!/bin/bash
set -u
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo '--- RUNNING ---'
pgrep -af 'smoke_single_node_tiny|thesis_platform.scripts.run_experiment|pretext_platform.scripts.run_pipeline|run_eval_small' || true
