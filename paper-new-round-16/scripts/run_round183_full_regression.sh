#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/public/caiqiyue_file/code_from_paper/paper-new-round-16
LOGDIR="$ROOT/logs"
SUMMARY="$LOGDIR/round183_full_regression_summary.tsv"
MASTER="$LOGDIR/round183_full_regression_master.log"

source /home/k8smaster/anaconda3/etc/profile.d/conda.sh
conda activate pretext
export CUDA_DEVICE_ORDER=PCI_BUS_ID
GPU_SLOT="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES="$GPU_SLOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$ROOT"
mkdir -p "$LOGDIR"

echo -e "experiment\tstatus\tselection_source\tconfigured_seed_top_k\tresolved_seed_top_k\tlength_family_resolved_seed_top_k\trunner_up_seed_top_k\tselection_stage\tfallback_used\tfeasible_budgets\tbest_top1\tbest_top3\tbest_top5\tbest_top10" > "$SUMMARY"
: > "$MASTER"

echo "$(date '+%F %T') ENV conda_env=pretext CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF" | tee -a "$MASTER"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader | tee -a "$MASTER"
echo "---" | tee -a "$MASTER"

had_failure=0

declare -a CONFIGS=(
  "configs/experiments/single_node_tuning_round183/full_run/r183_full_forums.yaml"
  "configs/experiments/single_node_tuning_round183/full_run/r183_full_congressional.yaml"
  "configs/experiments/single_node_tuning_round183/full_run/r183_full_jobs.yaml"
  "configs/experiments/single_node_tuning_round183/full_run/r183_full_microblog.yaml"
)

declare -a EXPS=(
  "r183_full_forums"
  "r183_full_congressional"
  "r183_full_jobs"
  "r183_full_microblog"
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
  python - <<'PY' | tee -a "$MASTER"
import os
print(f"runtime_cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
PY
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

echo "$(date '+%F %T') Round18.3 full regression done" | tee -a "$MASTER"
exit "$had_failure"
