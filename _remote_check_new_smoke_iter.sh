#!/bin/bash
sleep 30
echo '--- STATUS ---'
ps -p 24704 -o pid=,etime=,cmd=
echo '--- ITER FLAGS ---'
egrep -n 'Stage A iteration [123]/3|Prompt updated|Generated [0-9]+ critiques|Selected top|best_round_index|Stage B/C completed|metrics_summary' /mnt/public/caiqiyue_file/code_from_paper/old_automation/smoke_single_node_tiny.remote.log || true
