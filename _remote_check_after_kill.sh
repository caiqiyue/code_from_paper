#!/bin/bash
pgrep -af 'python -m thesis_platform.scripts.run_experiment|python -m pretext_platform.scripts.run_pipeline|python -m pretext_platform.scripts.run_eval_small|sn_c5_jobs_eps05|sp_c5_jobs_eps05' || true
echo GPU_PIDS
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader || true
