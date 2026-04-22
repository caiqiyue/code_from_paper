#!/usr/bin/env bash
set -euo pipefail
pgrep -af 'python -m thesis_platform.scripts.run_experiment|python -m pretext_platform.scripts.run_pipeline|python -m pretext_platform.scripts.run_eval_small|sn_c5_jobs_eps05|sp_c5_jobs_eps05' || true
pkill -f 'python -m thesis_platform.scripts.run_experiment' || true
pkill -f 'python -m pretext_platform.scripts.run_pipeline' || true
pkill -f 'python -m pretext_platform.scripts.run_eval_small' || true
pkill -f 'sn_c5_jobs_eps05' || true
pkill -f 'sp_c5_jobs_eps05' || true
sleep 2
echo AFTER_KILL
pgrep -af 'python -m thesis_platform.scripts.run_experiment|python -m pretext_platform.scripts.run_pipeline|python -m pretext_platform.scripts.run_eval_small|sn_c5_jobs_eps05|sp_c5_jobs_eps05' || true
echo GPU
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
