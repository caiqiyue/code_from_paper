#!/bin/bash
sleep 120
echo '--- STATUS ---'
ps -p 24704 -o pid=,etime=,cmd=
echo '--- ROUND DIRS ---'
find /mnt/public/caiqiyue_file/code_from_paper/outputs/thesis_platform/smoke_single_node_tiny/rounds -maxdepth 2 -type d | sort 2>/dev/null || true
echo '--- SUMMARY FLAGS ---'
egrep -n 'Stage A completed|Stage B/C completed|best_round_index|evaluation_rounds|metrics_summary' /mnt/public/caiqiyue_file/code_from_paper/old_automation/smoke_single_node_tiny.remote.log || true
