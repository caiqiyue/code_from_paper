#!/bin/bash

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

PYTHON="${PYTHON:-python3}"
PARALLEL=1
RUN_GLUE=0
DRY_RUN=0
RESUME=0
SMOKE_TEST=0
declare -a SUITES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)
            SMOKE_TEST=1
            shift
            ;;
        --jobs|--forums|--microblog|--congressional)
            SUITES+=("${1#--}")
            shift
            ;;
        --glue)
            RUN_GLUE=1
            shift
            ;;
        --parallel)
            PARALLEL="${2:-2}"
            shift 2
            ;;
        --resume)
            RESUME=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ "$SMOKE_TEST" -eq 1 ]]; then
    CMD=("$PYTHON" -m pretext_platform.scripts.run_pipeline --config configs/experiments/validate_tiny_complete_test.yaml)
else
    CMD=("$PYTHON" -m pretext_platform.scripts.run_experiments)
    if [[ ${#SUITES[@]} -gt 0 ]]; then
        CMD+=(--suite "${SUITES[@]}")
    else
        CMD+=(--all)
    fi
    if [[ "$RUN_GLUE" -eq 1 ]]; then
        CMD+=(--with-glue)
    fi
    if [[ "$PARALLEL" -gt 1 ]]; then
        CMD+=(--parallel "$PARALLEL")
    fi
    if [[ "$RESUME" -eq 1 ]]; then
        CMD+=(--resume)
    fi
fi

echo "Command: ${CMD[*]}"
if [[ "$DRY_RUN" -eq 1 ]]; then
    exit 0
fi

"${CMD[@]}"
