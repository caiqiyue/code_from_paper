#!/bin/bash
egrep -n 'Stage A completed|Stage B/C completed|round_index|best_round_index|evaluation_rounds|metrics_summary|Prompt updated|Stage A iteration [123]/3' /mnt/public/caiqiyue_file/code_from_paper/old_automation/smoke_single_node_tiny.remote.log || true
