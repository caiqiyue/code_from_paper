#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/public/caiqiyue_file/code_from_paper/paper-new-round-16
OUTROOT=/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs
LOGDIR="$ROOT/logs"
SUMMARY="$LOGDIR/round18_probe_batch_summary.tsv"
MASTER="$LOGDIR/round18_probe_batch_master.log"

source /home/k8smaster/anaconda3/etc/profile.d/conda.sh
conda activate pretext
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

cd "$ROOT"
mkdir -p "$LOGDIR"

echo -e "experiment\tstatus\tresolved_seed_top_k\trunner_up_seed_top_k\tutility_gap\tcoverage_mode\tfeasible_budgets\tcoverage_metrics\tfamily_score_by_budget\trecheck_passed\tpromoted_budget\tbest_top1\tbest_top3\tbest_top5\tbest_top10" > "$SUMMARY"
: > "$MASTER"

declare -a CONFIGS=(
  "configs/experiments/single_node_tuning_round18/probes/r18_probe_congressional_f1.yaml"
  "configs/experiments/single_node_tuning_round18/probes/r18_probe_congressional_f2.yaml"
  "configs/experiments/single_node_tuning_round18/probes/r18_probe_forums_f1.yaml"
  "configs/experiments/single_node_tuning_round18/focus/r18_congressional_f1.yaml"
  "configs/experiments/single_node_tuning_round18/focus/r18_congressional_f2.yaml"
  "configs/experiments/single_node_tuning_round18/focus/r18_congressional_f3.yaml"
  "configs/experiments/single_node_tuning_round18/seed_sanity/r18_forums_seed123.yaml"
  "configs/experiments/single_node_tuning_round18/seed_sanity/r18_forums_seed456.yaml"
)

declare -a EXPS=(
  "r18_probe_congressional_f1"
  "r18_probe_congressional_f2"
  "r18_probe_forums_f1"
  "r18_congressional_f1"
  "r18_congressional_f2"
  "r18_congressional_f3"
  "r18_forums_seed123"
  "r18_forums_seed456"
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

  python scripts/append_round18_summary.py "$SUMMARY" "$exp" "$status" "$outdir"
  echo "$(date '+%F %T') END $exp status=$status" | tee -a "$MASTER"
done

echo "$(date '+%F %T') Round18 probe batch done" | tee -a "$MASTER"
