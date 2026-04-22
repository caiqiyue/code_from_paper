#!/bin/bash
sleep 15
ps -p 14324 -o pid=,etime=,cmd=
echo '--- LOG TAIL ---'
tail -n 80 /mnt/public/caiqiyue_file/code_from_paper/old_automation/smoke_single_node_tiny.remote.log
