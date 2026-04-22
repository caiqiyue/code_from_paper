#!/bin/bash
sleep 90
echo '--- STATUS ---'
ps -p 24704 -o pid=,etime=,cmd=
echo '--- STAGE A FILES ---'
find /mnt/public/caiqiyue_file/code_from_paper/outputs/thesis_platform/smoke_single_node_tiny/stage_a -maxdepth 3 -type f | sort
echo '--- ROUND DIRS ---'
find /mnt/public/caiqiyue_file/code_from_paper/outputs/thesis_platform/smoke_single_node_tiny/rounds -maxdepth 2 -type d | sort 2>/dev/null || true
