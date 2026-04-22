#!/bin/bash
sleep 20
echo '--- STATUS ---'
ps -p 24704 -o pid=,etime=,cmd=
echo '--- TAIL ---'
tail -n 160 /mnt/public/caiqiyue_file/code_from_paper/old_automation/smoke_single_node_tiny.remote.log
