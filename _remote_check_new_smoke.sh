#!/bin/bash
sleep 20
ps -p 24704 -o pid=,etime=,cmd=
echo '--- LOG ---'
tail -n 120 /mnt/public/caiqiyue_file/code_from_paper/old_automation/smoke_single_node_tiny.remote.log
