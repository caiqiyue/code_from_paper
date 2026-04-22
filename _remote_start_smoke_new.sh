#!/bin/bash
set -euo pipefail
rm -rf /mnt/public/caiqiyue_file/code_from_paper/outputs/thesis_platform/smoke_single_node_tiny
rm -f /mnt/public/caiqiyue_file/code_from_paper/old_automation/smoke_single_node_tiny.remote.log
source /home/k8smaster/anaconda3/etc/profile.d/conda.sh
conda activate caiqiyue-vllm
cd /mnt/public/caiqiyue_file/code_from_paper
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
nohup python -m thesis_platform.scripts.run_experiment --config thesis_platform/configs/experiments/smoke/smoke_single_node_tiny.yaml > old_automation/smoke_single_node_tiny.remote.log 2>&1 &
echo $!
