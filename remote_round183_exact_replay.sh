#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/public/caiqiyue_file/code_from_paper/paper-new-round-16
LOGDIR="$ROOT/logs"
SUMMARY="$LOGDIR/round183_exact_replay_summary.tsv"
MASTER="$LOGDIR/round183_exact_replay_master.log"

source /home/k8smaster/anaconda3/etc/profile.d/conda.sh
conda activate pretext
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$LOGDIR"
cd "$ROOT"

echo -e "experiment\tstatus\tresolved_seed_top_k\trunner_up_seed_top_k\tbest_top1\tbest_top3\tbest_top5\tbest_top10" > "$SUMMARY"
: > "$MASTER"

echo "$(date '+%F %T') ENV conda_env=pretext CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF" | tee -a "$MASTER"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader | tee -a "$MASTER"
echo "---" | tee -a "$MASTER"

rm -rf \
  /mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/r182_forums_fixed22 \
  /mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/r181_congressional_g1

run_one() {
  local exp="$1"
  local cfg="$2"
  local log="$LOGDIR/${exp}_replay.log"

  echo "$(date '+%F %T') START $exp cfg=$cfg" | tee -a "$MASTER"
  python -m paper_new_selector.run_selector_single_node --config "$cfg" > "$log" 2>&1
  local status=$?

  if [ "$status" -eq 0 ]; then
    python - "$SUMMARY" "$exp" "$cfg" <<'PY'
import json
import sys
from pathlib import Path

from paper_new_selector.thesis_bridge import resolve_output_root

summary, exp, cfg = sys.argv[1:4]
outdir = resolve_output_root(Path(cfg))
stage1 = json.loads((outdir / "stage1_summary.json").read_text())
evaluation = json.loads((outdir / "eval" / "downstream_eval_summary.json").read_text())
seed = stage1.get("seed_budget") or {}
metrics = evaluation.get("metrics") or {}
row = [
    exp,
    "0",
    str(seed.get("resolved_seed_top_k", "NA")),
    str(seed.get("runner_up_seed_top_k", "NA")),
    str(metrics.get("best_top1", "NA")),
    str(metrics.get("best_top3", "NA")),
    str(metrics.get("best_top5", "NA")),
    str(metrics.get("best_top10", "NA")),
]
with open(summary, "a", encoding="utf-8") as handle:
    handle.write("\t".join(row) + "\n")
PY
  else
    echo -e "$exp\t$status\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR" >> "$SUMMARY"
  fi

  echo "$(date '+%F %T') END $exp status=$status" | tee -a "$MASTER"
}

run_one r182_forums_fixed22 configs/experiments/single_node_tuning_round182/fixed_budget/r182_forums_fixed22.yaml
run_one r181_congressional_g1 configs/experiments/single_node_tuning_round181/diagnostics/r181_congressional_g1.yaml

echo "$(date '+%F %T') exact replay batch done" | tee -a "$MASTER"
