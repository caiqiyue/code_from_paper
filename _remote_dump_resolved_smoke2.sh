#!/bin/bash
base=/mnt/public/caiqiyue_file/code_from_paper/outputs/thesis_platform/smoke_single_node_tiny
if [ -f "$base/resolved_config.json" ]; then
  python - <<'PY'
import json
from pathlib import Path
p = Path('/mnt/public/caiqiyue_file/code_from_paper/outputs/thesis_platform/smoke_single_node_tiny/resolved_config.json')
obj = json.loads(p.read_text())
for key in ['scorer','retriever','critic','aggregator','stage_a','stage_b','generator']:
    print(f'== {key} ==')
    print(json.dumps(obj.get(key, {}), ensure_ascii=False, indent=2))
PY
else
  echo missing_resolved_config
fi
