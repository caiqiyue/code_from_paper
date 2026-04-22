#!/bin/bash
sleep 60
echo '--- STATUS ---'
ps -p 24704 -o pid=,etime=,cmd=
echo '--- LOG TAIL ---'
tail -n 80 /mnt/public/caiqiyue_file/code_from_paper/old_automation/smoke_single_node_tiny.remote.log
