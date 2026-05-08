#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PARENT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
WASP_ROOT="${PARENT_ROOT}/WASP"
DPGA_ROOT="${PARENT_ROOT}/DPGA-TextSyn"
PRETEXT_ENV="${PRETEXT_ENV:-pretext}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH" >&2
  exit 1
fi

if [[ ! -d "${WASP_ROOT}" ]]; then
  echo "Missing sibling repo: ${WASP_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${DPGA_ROOT}" ]]; then
  echo "Missing sibling repo: ${DPGA_ROOT}" >&2
  exit 1
fi

cd "${REPO_ROOT}"

conda run --no-capture-output -n "${PRETEXT_ENV}" python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_screening/c4_s_jobs_screening.yaml --validate-only
conda run --no-capture-output -n "${PRETEXT_ENV}" python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_screening/eo_s_jobs_screening.yaml --validate-only
conda run --no-capture-output -n "${PRETEXT_ENV}" python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_screening/ep_s_jobs_screening.yaml --validate-only
conda run --no-capture-output -n "${PRETEXT_ENV}" python "${WASP_ROOT}/src/run_paper_new_screening.py" --generated-jsonl "${WASP_ROOT}/src/data_accumulate_start/imdb/chatglm3-6b-base/100_20/train.jsonl" --output-json outputs/wasp_stage1_summary.json --budget 100
conda run --no-capture-output -n "${PRETEXT_ENV}" python -m paper_new_selector.run_external_baseline_eval --summary-json outputs/wasp_stage1_summary.json --config configs/experiments/single_node_screening/wasp_s_jobs_screening.yaml --output-dir outputs/wasp_s_jobs_screening
conda run --no-capture-output -n "${PRETEXT_ENV}" python -c "import json; from pathlib import Path; Path('outputs').mkdir(exist_ok=True); Path('outputs/dpga_epoch_all.json').write_text(json.dumps([{'text': f'dpga synthetic sample {i} with enough words for screening'} for i in range(24)]), encoding='utf-8')"
conda run --no-capture-output -n "${PRETEXT_ENV}" python "${DPGA_ROOT}/main/run_paper_new_screening.py" --epoch-all-json outputs/dpga_epoch_all.json --output-json outputs/dpga_stage1_summary.json --budget 100
conda run --no-capture-output -n "${PRETEXT_ENV}" python -m paper_new_selector.run_external_baseline_eval --summary-json outputs/dpga_stage1_summary.json --config configs/experiments/single_node_screening/dpga_s_jobs_screening.yaml --output-dir outputs/dpga_s_jobs_screening
