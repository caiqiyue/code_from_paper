#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="$ROOT/logs"
SUMMARY="$LOGDIR/round21_quick_compare_summary.tsv"
MASTER="$LOGDIR/round21_quick_compare_master.log"

cd "$ROOT"
mkdir -p "$LOGDIR"

echo -e "experiment\tstatus\tregime\tresolved_seed_top_k\tselection_stage\tarbitration_triggered\tarbitration_winner_policy\tarbitration_reason\tarbitration_broad_stability\tarbitration_compact_stability\tbest_top1\tvalidation_status" > "$SUMMARY"
: > "$MASTER"

declare -a CONFIGS=(
  "configs/experiments/single_node_tuning_round21/quick_compare/r21_jobs_tau175_seed42_fallback.yaml"
  "configs/experiments/single_node_tuning_round21/quick_compare/r21_jobs_tau175_seed42_arbitration.yaml"
  "configs/experiments/single_node_tuning_round21/quick_compare/r21_jobs_tau175_seed123_fallback.yaml"
  "configs/experiments/single_node_tuning_round21/quick_compare/r21_jobs_tau175_seed123_arbitration.yaml"
  "configs/experiments/single_node_tuning_round21/quick_compare/r21_microblog_tau300_seed42_fallback.yaml"
  "configs/experiments/single_node_tuning_round21/quick_compare/r21_microblog_tau300_seed42_arbitration.yaml"
  "configs/experiments/single_node_tuning_round21/quick_compare/r21_microblog_tau300_seed123_fallback.yaml"
  "configs/experiments/single_node_tuning_round21/quick_compare/r21_microblog_tau300_seed123_arbitration.yaml"
)

had_failure=0

for cfg in "${CONFIGS[@]}"; do
  exp="$(python - <<'PY' "$cfg"
from pathlib import Path
import sys
from paper_new_selector.thesis_bridge import load_yaml_config
print(load_yaml_config(Path(sys.argv[1]))["meta"]["experiment_id"])
PY
)"
  expected_stage="uncertain_fallback_policy"
  expected_triggered="false"
  if [[ "$exp" == *"_arbitration" ]]; then
    expected_stage="uncertainty_policy_arbitration"
    expected_triggered="true"
  fi
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
    set +e
    python scripts/append_round21_summary.py \
      "$SUMMARY" "$exp" "$status" "$outdir" \
      "uncertain" "$expected_stage" "$expected_triggered"
    summary_status=$?
    set -e
    if [ "$summary_status" -ne 0 ]; then
      echo "$(date '+%F %T') VALIDATION_FAIL $exp summary_status=$summary_status" | tee -a "$MASTER"
      had_failure=1
    fi
  else
    echo -e "${exp}\t${status}\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tRUN_FAILED" >> "$SUMMARY"
    had_failure=1
  fi

  echo "$(date '+%F %T') END $exp status=$status" | tee -a "$MASTER"
done

exit "$had_failure"
