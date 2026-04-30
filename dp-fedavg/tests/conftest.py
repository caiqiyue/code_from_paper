from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent

for candidate in [str(PROJECT_ROOT), str(REPO_ROOT)]:
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
