#!/bin/bash
ps -p 14324 -o pid=,etime=,cmd=
 echo '--- LOG HEAD ---'
 sed -n '1,80p' /mnt/public/caiqiyue_file/code_from_paper/old_automation/smoke_single_node_tiny.remote.log
