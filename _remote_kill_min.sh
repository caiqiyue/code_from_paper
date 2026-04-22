#!/bin/bash
kill -9 15794 15820 4132 4255 2>/dev/null || true
sleep 2
echo AFTER_KILL
ps -p 15794,15820,4132,4255 -o pid=,cmd= || true
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
