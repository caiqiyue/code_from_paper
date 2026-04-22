#!/bin/bash
set -u
base=/mnt/public/caiqiyue_file/code_from_paper/old_automation
for label in SN-C5 SP-C5; do
  echo "===== $label LOG ====="
  if [ -f "$base/$label.remote.log" ]; then
    tail -n 120 "$base/$label.remote.log"
  else
    echo "missing log: $base/$label.remote.log"
  fi
  echo "===== $label OUT ====="
  case $label in
    SN-C5) out=/mnt/public/caiqiyue_file/code_from_paper/outputs/thesis_platform/sn_c5_jobs_eps05 ;;
    SP-C5) out=/mnt/public/caiqiyue_file/code_from_paper/PrE-Text/outputs/pretext_platform/sp_c5_jobs_eps05 ;;
  esac
  if [ -d "$out" ]; then
    find "$out" -maxdepth 2 -type f | sort
  else
    echo "missing output dir: $out"
  fi
  echo
 done
