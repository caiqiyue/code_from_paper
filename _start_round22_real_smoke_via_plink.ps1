ROOT=/mnt/public/caiqiyue_file/code_from_paper/paper-new-round22
LOGDIR=$ROOT/logs
MANIFEST=$ROOT/configs/experiments/single_node_tuning_round22_bandit/real_smoke/round22_real_smoke_manifest.tsv
MODEL_DIR=$ROOT/artifacts/learned_budget_policy/round22_lgbm_full500_v2
MASTER=$LOGDIR/round22_real_smoke_master.log
NOHUP=$LOGDIR/round22_real_smoke_nohup.log
RUNNER=/tmp/round22_real_smoke_runner.sh
WATCHER=/tmp/round22_real_smoke_watch.sh
mkdir -p $LOGDIR
cat > $RUNNER <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/public/caiqiyue_file/code_from_paper/paper-new-round22
LOGDIR=$ROOT/logs
MANIFEST=$ROOT/configs/experiments/single_node_tuning_round22_bandit/real_smoke/round22_real_smoke_manifest.tsv
MODEL_DIR=$ROOT/artifacts/learned_budget_policy/round22_lgbm_full500_v2
MASTER=$LOGDIR/round22_real_smoke_master.log
mkdir -p "$LOGDIR"
cd "$ROOT"
source /home/k8smaster/anaconda3/etc/profile.d/conda.sh
conda activate pretext
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
awk 'NR>1 {print $1"\t"$4"\t"$5}' "$MANIFEST" | while IFS=$'\t' read -r exp cfg outroot; do
  logfile="$LOGDIR/${exp}.log"
  echo "$(date '+%F %T') START $exp config=$cfg output=$outroot" | tee -a "$MASTER"
  if python scripts/run_round22_with_learned_policy.py --config "$ROOT/$cfg" --model-dir "$MODEL_DIR" --output-root "/mnt/public/caiqiyue_file/code_from_paper/$outroot" > "$logfile" 2>&1; then
    echo "$(date '+%F %T') END $exp status=0" | tee -a "$MASTER"
  else
    status=$?
    echo "$(date '+%F %T') END $exp status=$status" | tee -a "$MASTER"
    exit $status
  fi
  sleep 5
done
EOF
chmod +x $RUNNER
cat > $WATCHER <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
while ps -p 4133 -o pid= >/dev/null 2>&1; do
  sleep 20
done
exec /tmp/round22_real_smoke_runner.sh
EOF
chmod +x $WATCHER
cd "$ROOT"
nohup bash /tmp/round22_real_smoke_watch.sh > "$NOHUP" 2>&1 &
echo $!
