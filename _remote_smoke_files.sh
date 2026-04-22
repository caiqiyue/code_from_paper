#!/bin/bash
base=/mnt/public/caiqiyue_file/code_from_paper/outputs/thesis_platform/smoke_single_node_tiny
echo '===== STAGE A FILES ====='
find "$base/stage_a" -maxdepth 3 -type f | sort
echo '===== ROUND DIRS ====='
find "$base/rounds" -maxdepth 2 -type d | sort
