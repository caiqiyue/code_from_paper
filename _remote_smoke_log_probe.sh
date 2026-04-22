#!/bin/bash
set -u
log=/mnt/public/caiqiyue_file/code_from_paper/old_automation/smoke_single_node_tiny.remote.log
if [ -f "$log" ]; then
  echo '===== KEYWORDS ====='
  egrep -n 'Traceback|ImportError|RuntimeError|sentence_transformers|transformers|peft|retriever|critic|aggregator|Stage A|DataInf|vLLM generation memory precheck' "$log" | tail -n 120
  echo '===== TAIL ====='
  tail -n 120 "$log"
else
  echo missing_log
fi
