#!/bin/bash
set -u
base=/mnt/public/caiqiyue_file/code_from_paper/outputs/thesis_platform/smoke_single_node_tiny
echo '===== RUN STATE ====='
if [ -f "$base/run_state.json" ]; then
  cat "$base/run_state.json"
else
  echo missing run_state.json
fi
echo '===== FILES ====='
find "$base" -maxdepth 3 -type f | sort
echo '===== LOG TAIL ====='
if [ -f /mnt/public/caiqiyue_file/code_from_paper/old_automation/smoke_single_node_tiny.remote.log ]; then
  tail -n 120 /mnt/public/caiqiyue_file/code_from_paper/old_automation/smoke_single_node_tiny.remote.log
else
  echo missing log
fi
