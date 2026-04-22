#!/bin/bash
source /home/k8smaster/anaconda3/etc/profile.d/conda.sh
conda activate caiqiyue-vllm
python - <<'PY'
import transformers
print('before', hasattr(transformers, 'EncoderDecoderCache'))
if not hasattr(transformers, 'EncoderDecoderCache') and hasattr(transformers, 'Cache'):
    transformers.EncoderDecoderCache = transformers.Cache
print('after', hasattr(transformers, 'EncoderDecoderCache'))
import sentence_transformers
print('import_ok')
PY
