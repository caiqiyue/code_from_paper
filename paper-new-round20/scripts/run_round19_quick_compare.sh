#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="$ROOT/logs"
SUMMARY="$LOGDIR/round19_quick_compare_summary.tsv"
MASTER="$LOGDIR/round19_quick_compare_master.log"

cd "$ROOT"
mkdir -p "$LOGDIR"

echo -e "experiment\tstatus\tmode\tregime\tconfigured_seed_top_k\tresolved_seed_top_k\trunner_up_seed_top_k\tselection_stage\tfallback_used\tpolicy_fallback_used\tfeasible_budgets\tshape_score\tbest_top1\tbest_top3\tbest_top5\tbest_top10" > "$SUMMARY"
: > "$MASTER"

declare -a CONFIGS=(
  "configs/experiments/single_node_tuning_round19/guard/r19_guard_forums.yaml"
  "configs/experiments/single_node_tuning_round19/guard/r19_guard_congressional.yaml"
  "configs/experiments/single_node_tuning_round19/full_run/r19_full_jobs.yaml"
  "configs/experiments/single_node_tuning_round19/full_run/r19_full_congressional.yaml"
  "configs/experiments/single_node_tuning_round19/full_run/r19_full_forums.yaml"
  "configs/experiments/single_node_tuning_round19/full_run/r19_full_microblog.yaml"
  "configs/experiments/single_node_tuning_round19/ablations/r19_ablate_no_router_forums.yaml"
  "configs/experiments/single_node_tuning_round19/ablations/r19_ablate_no_router_congressional.yaml"
  "configs/experiments/single_node_tuning_round19/ablations/r19_ablate_force_compact_forums.yaml"
  "configs/experiments/single_node_tuning_round19/ablations/r19_ablate_force_broad_congressional.yaml"
)

declare -a EXPS=(
  "r19_guard_forums"
  "r19_guard_congressional"
  "r19_full_jobs"
  "r19_full_congressional"
  "r19_full_forums"
  "r19_full_microblog"
  "r19_ablate_no_router_forums"
  "r19_ablate_no_router_congressional"
  "r19_ablate_force_compact_forums"
  "r19_ablate_force_broad_congressional"
)

had_failure=0

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
  final_status="$status"

  if [ "$status" -eq 0 ]; then
    set +e
    python scripts/append_round19_summary.py "$SUMMARY" "$exp" "$status" "$outdir"
    summary_status=$?
    set -e
    if [ "$summary_status" -ne 0 ]; then
      echo -e "${exp}\tSUMMARY_ERROR_${summary_status}\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR" >> "$SUMMARY"
      had_failure=1
      final_status="summary_error_${summary_status}"
    fi
  else
    echo -e "${exp}\t${status}\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR" >> "$SUMMARY"
    had_failure=1
  fi

  echo "$(date '+%F %T') END $exp status=$final_status" | tee -a "$MASTER"
done

exit "$had_failure"
