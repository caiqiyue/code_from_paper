#!/bin/bash
source /home/k8smaster/anaconda3/etc/profile.d/conda.sh
conda activate caiqiyue
python - <<'PY'
import importlib.metadata as md
print('transformers=' + md.version('transformers'))
print('peft=' + md.version('peft'))
print('sentence-transformers=' + md.version('sentence-transformers'))
import transformers
print('EncoderDecoderCache=' + str(hasattr(transformers, 'EncoderDecoderCache')))
import sentence_transformers
print('sentence_transformers_import=OK')
PY
