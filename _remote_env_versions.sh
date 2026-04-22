#!/bin/bash
source /home/k8smaster/anaconda3/etc/profile.d/conda.sh
conda activate caiqiyue-vllm
python - <<'PY'
import os, sys
print('CONDA_DEFAULT_ENV=' + os.environ.get('CONDA_DEFAULT_ENV', ''))
print('CONDA_PREFIX=' + os.environ.get('CONDA_PREFIX', ''))
print('PYTHON=' + sys.executable)
try:
    import transformers, peft, sentence_transformers
    print('transformers=' + transformers.__version__)
    print('peft=' + peft.__version__)
    print('sentence_transformers=' + sentence_transformers.__version__)
except Exception as exc:
    print(type(exc).__name__ + ': ' + str(exc))
PY
