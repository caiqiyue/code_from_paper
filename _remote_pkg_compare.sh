#!/bin/bash
source /home/k8smaster/anaconda3/etc/profile.d/conda.sh
for env in caiqiyue-vllm caiqiyue; do
  echo "===== $env ====="
  conda activate "$env" 2>/dev/null || { echo "missing env"; continue; }
  python - <<'PY'
import importlib.metadata as md
for pkg in ['transformers','peft','sentence-transformers']:
    try:
        print(f'{pkg}=' + md.version(pkg))
    except Exception as exc:
        print(f'{pkg}=ERROR:{exc}')
PY
done
