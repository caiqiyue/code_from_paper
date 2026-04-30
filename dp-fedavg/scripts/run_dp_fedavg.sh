#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:?usage: run_dp_fedavg.sh <config.yaml>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHONPATH="$ROOT/..:${PYTHONPATH:-}" python -m dp_fedavg.run_experiment --config "$CONFIG_PATH"
