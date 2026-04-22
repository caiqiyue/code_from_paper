#!/bin/bash
set -u
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader || true
echo '--- PS ---'
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk -F',' '{gsub(/ /, "", $1); print $1}'); do
  ps -p "$pid" -o pid=,etime=,cmd=
 done
