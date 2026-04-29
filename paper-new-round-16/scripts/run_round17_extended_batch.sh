#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/public/caiqiyue_file/code_from_paper/paper-new-round-16
OUTROOT=/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs
LOGDIR="$ROOT/logs"
SUMMARY="$LOGDIR/round17_extended_batch_summary.tsv"
MASTER="$LOGDIR/round17_extended_batch_master.log"

source /home/k8smaster/anaconda3/etc/profile.d/conda.sh
conda activate pretext
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

cd "$ROOT"
mkdir -p "$LOGDIR"

echo -e "experiment\tstatus\tresolved_seed_top_k\trunner_up_seed_top_k\tutility_gap\tfeasible_budgets\tcoverage_threshold\tbest_coverage_p25\tselection_stage\tfallback_used\tbest_top1\tbest_top3\tbest_top5\tbest_top10" > "$SUMMARY"
: > "$MASTER"

declare -a CONFIGS=(
  "configs/experiments/single_node_tuning_round17/ratio_sweep/r17_microblog_r099.yaml"
  "configs/experiments/single_node_tuning_round17/ratio_sweep/r17_microblog_r098.yaml"
  "configs/experiments/single_node_tuning_round17/ratio_sweep/r17_microblog_r097.yaml"
  "configs/experiments/single_node_tuning_round17/ratio_sweep/r17_jobs_r098.yaml"
  "configs/experiments/single_node_tuning_round17/ratio_sweep/r17_congressional_r098.yaml"
)

declare -a EXPS=(
  "r17_microblog_r099"
  "r17_microblog_r098"
  "r17_microblog_r097"
  "r17_jobs_r098"
  "r17_congressional_r098"
)

for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  exp="${EXPS[$i]}"
  outdir="$OUTROOT/$exp"
  log="$LOGDIR/${exp}.log"

  echo "$(date '+%F %T') START $exp cfg=$cfg" | tee -a "$MASTER"
  rm -rf "$outdir"

  set +e
  python -m paper_new_selector.run_selector_single_node --config "$cfg" > "$log" 2>&1
  status=$?
  set -e

  python scripts/append_round17_summary.py "$SUMMARY" "$exp" "$status" "$outdir"
  echo "$(date '+%F %T') END $exp status=$status" | tee -a "$MASTER"
done

echo "$(date '+%F %T') Round17 extended batch done" | tee -a "$MASTER"
