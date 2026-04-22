#!/bin/bash
set -u
ps -p 507 -o pid=,etime=,cmd=
 echo '--- LOG TAIL ---'
 tail -n 40 /mnt/public/caiqiyue_file/code_from_paper/old_automation/smoke_single_node_tiny.remote.log
