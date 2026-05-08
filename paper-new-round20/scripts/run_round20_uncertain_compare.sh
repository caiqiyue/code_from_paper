#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="$ROOT/logs"
SUMMARY="$LOGDIR/round20_uncertain_compare_summary.tsv"
MASTER="$LOGDIR/round20_uncertain_compare_master.log"

cd "$ROOT"
mkdir -p "$LOGDIR"

echo -e "experiment\tstatus\tregime\tresolved_seed_top_k\tselection_stage\tarbitration_triggered\tarbitration_winner_policy\tarbitration_reason\tbest_top1\tvalidation_status" > "$SUMMARY"
: > "$MASTER"

declare -a CONFIGS=(
  "configs/experiments/single_node_tuning_round20/r20_jobs_baseline_fallback.yaml"
  "configs/experiments/single_node_tuning_round20/r20_jobs_arbitration.yaml"
  "configs/experiments/single_node_tuning_round20/r20_microblog_baseline_fallback.yaml"
  "configs/experiments/single_node_tuning_round20/r20_microblog_arbitration.yaml"
)

declare -a EXPS=(
  "r20_jobs_baseline_fallback"
  "r20_jobs_arbitration"
  "r20_microblog_baseline_fallback"
  "r20_microblog_arbitration"
)

declare -a EXPECTED_REGIMES=(
  "uncertain"
  "uncertain"
  "uncertain"
  "uncertain"
)

declare -a EXPECTED_STAGES=(
  "uncertain_fallback_policy"
  "uncertainty_policy_arbitration"
  "uncertain_fallback_policy"
  "uncertainty_policy_arbitration"
)

declare -a EXPECTED_TRIGGERED=(
  "false"
  "true"
  "false"
  "true"
)

had_failure=0

for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  exp="${EXPS[$i]}"
  expected_regime="${EXPECTED_REGIMES[$i]}"
  expected_stage="${EXPECTED_STAGES[$i]}"
  expected_triggered="${EXPECTED_TRIGGERED[$i]}"
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
    python scripts/append_round20_uncertain_summary.py \
      "$SUMMARY" "$exp" "$status" "$outdir" \
      "$expected_regime" "$expected_stage" "$expected_triggered"
    summary_status=$?
    set -e
    if [ "$summary_status" -ne 0 ]; then
      echo "$(date '+%F %T') VALIDATION_FAIL $exp summary_status=$summary_status expected_regime=$expected_regime expected_stage=$expected_stage expected_triggered=$expected_triggered" | tee -a "$MASTER"
      had_failure=1
    fi
  else
    echo -e "${exp}\t${status}\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tRUN_FAILED" >> "$SUMMARY"
    had_failure=1
  fi

  echo "$(date '+%F %T') END $exp status=$status" | tee -a "$MASTER"
done

exit "$had_failure"
