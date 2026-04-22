#!/bin/bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af 'sn_c5_jobs_eps05|thesis_platform.scripts.run_experiment|pretext_platform.scripts.run_pipeline|run_eval_small' || true
