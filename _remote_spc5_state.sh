#!/bin/bash
set -u
base=/mnt/public/caiqiyue_file/code_from_paper/PrE-Text/outputs/pretext_platform/sp_c5_jobs_eps05
echo '===== SP-C5 RUN STATE ====='
if [ -f "$base/run_state.json" ]; then
  cat "$base/run_state.json"
else
  echo missing run_state.json
fi
echo '===== SP-C5 LOG TAIL ====='
if [ -f /mnt/public/caiqiyue_file/code_from_paper/old_automation/SP-C5.remote.log ]; then
  tail -n 120 /mnt/public/caiqiyue_file/code_from_paper/old_automation/SP-C5.remote.log | egrep -i 'error|traceback|failed|exception|terminate|killed|stage2|eval_small|metrics_summary|warning|running|complete|success' || tail -n 120 /mnt/public/caiqiyue_file/code_from_paper/old_automation/SP-C5.remote.log
else
  echo missing log
fi
