#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: bash scripts/run_dp_prompt.sh <config-yaml>" >&2
  exit 1
fi

python -m dp_prompt.cli --config "$1"
