#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/public/caiqiyue_file/code_from_paper/paper-new-round-16
LOGDIR="$ROOT/logs"
SUMMARY="$LOGDIR/round182_forums_fixed_budget_summary.tsv"
MASTER="$LOGDIR/round182_forums_fixed_budget_master.log"

source /home/k8smaster/anaconda3/etc/profile.d/conda.sh
conda activate pretext
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

cd "$ROOT"
mkdir -p "$LOGDIR"

echo -e "experiment\tstatus\tconfigured_seed_top_k\tresolved_seed_top_k\tseed_budget_mode\tbootstrap_max_tokens\tbest_top1\tbest_top3\tbest_top5\tbest_top10" > "$SUMMARY"
: > "$MASTER"

INCLUDE_OPTIONAL20="${INCLUDE_OPTIONAL20:-0}"
had_failure=0

declare -a CONFIGS=(
  "configs/experiments/single_node_tuning_round182/fixed_budget/r182_forums_fixed21.yaml"
  "configs/experiments/single_node_tuning_round182/fixed_budget/r182_forums_fixed22.yaml"
)

declare -a EXPS=(
  "r182_forums_fixed21"
  "r182_forums_fixed22"
)

if [ "$INCLUDE_OPTIONAL20" = "1" ]; then
  CONFIGS+=("configs/experiments/single_node_tuning_round182/fixed_budget/r182_forums_fixed20.yaml")
  EXPS+=("r182_forums_fixed20")
fi

for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  exp="${EXPS[$i]}"
  outdir="$(python - <<'PY' "$cfg"
from pathlib import Path
import sys
from paper_new_selector.thesis_bridge import resolve_output_root
print(resolve_output_root(Path(sys.argv[1])))
PY
)"
  log="$LOGDIR/${exp}.log"

  echo "$(date '+%F %T') START $exp cfg=$cfg" | tee -a "$MASTER"
  rm -rf "$outdir"

  set +e
  python -m paper_new_selector.run_selector_single_node --config "$cfg" > "$log" 2>&1
  status=$?
  set -e

  if [ "$status" -eq 0 ]; then
    python scripts/append_round182_summary.py "$SUMMARY" "$exp" "$status" "$cfg" "$outdir"
  else
    echo -e "${exp}\t${status}\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR" >> "$SUMMARY"
    had_failure=1
  fi
  echo "$(date '+%F %T') END $exp status=$status" | tee -a "$MASTER"
done

echo "$(date '+%F %T') Round18.2 forums fixed-budget sweep done" | tee -a "$MASTER"
exit "$had_failure"
