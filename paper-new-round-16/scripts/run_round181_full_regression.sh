#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/public/caiqiyue_file/code_from_paper/paper-new-round-16
OUTROOT=/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs
LOGDIR="$ROOT/logs"
SUMMARY="$LOGDIR/round181_full_regression_summary.tsv"
MASTER="$LOGDIR/round181_full_regression_master.log"

source /home/k8smaster/anaconda3/etc/profile.d/conda.sh
conda activate pretext
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

cd "$ROOT"
mkdir -p "$LOGDIR"

echo -e "experiment\tstatus\tresolved_seed_top_k\trunner_up_seed_top_k\tutility_gap\tfeasible_budgets\tcandidate_budget\tpromoted_budget\trecheck_passed\tsupport_drop\tsupport_drop_normalized\tcoverage_mean_gain\tcoverage_p25_gain\tcoverage_min_gain\tfamily_score_gain\tbest_top1\tbest_top3\tbest_top5\tbest_top10" > "$SUMMARY"
: > "$MASTER"

declare -a CONFIGS=(
  "configs/experiments/single_node_tuning_round181/full_run/r181_full_forums.yaml"
  "configs/experiments/single_node_tuning_round181/full_run/r181_full_microblog.yaml"
  "configs/experiments/single_node_tuning_round181/full_run/r181_full_jobs.yaml"
  "configs/experiments/single_node_tuning_round181/full_run/r181_full_congressional.yaml"
)

declare -a EXPS=(
  "r181_full_forums"
  "r181_full_microblog"
  "r181_full_jobs"
  "r181_full_congressional"
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

  python scripts/append_round181_summary.py "$SUMMARY" "$exp" "$status" "$outdir"
  echo "$(date '+%F %T') END $exp status=$status" | tee -a "$MASTER"
done

echo "$(date '+%F %T') Round18.1 full regression done" | tee -a "$MASTER"
