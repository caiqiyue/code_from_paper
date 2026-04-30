#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/public/caiqiyue_file/code_from_paper/paper-new-round-16
LOGDIR="$ROOT/logs"
SUMMARY="$LOGDIR/round183_guard_batch_summary.tsv"
MASTER="$LOGDIR/round183_guard_batch_master.log"

source /home/k8smaster/anaconda3/etc/profile.d/conda.sh
conda activate pretext
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES
fi

cd "$ROOT"
mkdir -p "$LOGDIR"

echo -e "experiment\tstatus\tselection_source\tconfigured_seed_top_k\tresolved_seed_top_k\tlength_family_resolved_seed_top_k\trunner_up_seed_top_k\tselection_stage\tfallback_used\tfeasible_budgets\tbest_top1\tbest_top3\tbest_top5\tbest_top10" > "$SUMMARY"
: > "$MASTER"

had_failure=0

declare -a CONFIGS=(
  "configs/experiments/single_node_tuning_round183/guard/r183_guard_forums.yaml"
  "configs/experiments/single_node_tuning_round183/guard/r183_guard_congressional.yaml"
)

declare -a EXPS=(
  "r183_guard_forums"
  "r183_guard_congressional"
)

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
    python scripts/append_round183_summary.py "$SUMMARY" "$exp" "$status" "$outdir"
    summary_status=$?
    set -e
    if [ "$summary_status" -ne 0 ]; then
      echo -e "${exp}\tSUMMARY_ERROR_${summary_status}\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR" >> "$SUMMARY"
      had_failure=1
      final_status="summary_error_${summary_status}"
    fi
  else
    echo -e "${exp}\t${status}\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR" >> "$SUMMARY"
    had_failure=1
  fi
  echo "$(date '+%F %T') END $exp status=$final_status" | tee -a "$MASTER"
done

echo "$(date '+%F %T') Round18.3 guard batch done" | tee -a "$MASTER"
exit "$had_failure"
