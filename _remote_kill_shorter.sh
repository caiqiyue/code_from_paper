#!/bin/bash
kill -9 3569
sleep 2
echo '--- AFTER ---'
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader || true
ps -p 3569,16962 -o pid=,etime=,cmd= || true
