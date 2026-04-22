#!/bin/bash
base=/mnt/public/caiqiyue_file/code_from_paper/PrE-Text/outputs/pretext_platform/sp_c5_jobs_eps05
echo 'FILES'
find "$base" -maxdepth 1 -type f | sort
if [ -f "$base/failure_summary.json" ]; then
  echo '===== FAILURE SUMMARY ====='
  cat "$base/failure_summary.json"
fi
if [ -f "$base/metrics_summary.json" ]; then
  echo '===== METRICS SUMMARY ====='
  cat "$base/metrics_summary.json"
fi
echo '===== PROCESS CHECK ====='
pgrep -af 'sp_c5_jobs_eps05|pretext_platform.scripts.run_pipeline|pretext_platform.scripts.run_eval_small' || true
