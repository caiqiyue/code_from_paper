#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "$#" -ne 1 ]; then
  echo "Usage: bash scripts/run_dp_prompt.sh <config-yaml>" >&2
  exit 1
fi

cd "$PROJECT_ROOT"
python -m dp_prompt.cli --config "$1"
