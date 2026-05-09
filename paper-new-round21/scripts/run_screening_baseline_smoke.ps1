$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$ParentRoot = Split-Path -Parent $RepoRoot
$WaspRoot = Join-Path $ParentRoot "WASP"
$DpgaRoot = Join-Path $ParentRoot "DPGA-TextSyn"
$PretextEnv = if ($env:PRETEXT_ENV) { $env:PRETEXT_ENV } else { "pretext" }

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda not found in PATH"
}

if (-not (Test-Path -LiteralPath $WaspRoot -PathType Container)) {
    throw "Missing sibling repo: $WaspRoot"
}

if (-not (Test-Path -LiteralPath $DpgaRoot -PathType Container)) {
    throw "Missing sibling repo: $DpgaRoot"
}

Push-Location $RepoRoot
try {
    conda run --no-capture-output -n $PretextEnv python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_screening/c4_s_jobs_screening.yaml --validate-only
    conda run --no-capture-output -n $PretextEnv python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_screening/eo_s_jobs_screening.yaml --validate-only
    conda run --no-capture-output -n $PretextEnv python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_screening/ep_s_jobs_screening.yaml --validate-only

    conda run --no-capture-output -n $PretextEnv python (Join-Path $WaspRoot "src/run_paper_new_screening.py") --generated-jsonl (Join-Path $WaspRoot "src/data_accumulate_start/imdb/chatglm3-6b-base/100_20/train.jsonl") --output-json outputs/wasp_stage1_summary.json --budget 100
    conda run --no-capture-output -n $PretextEnv python -m paper_new_selector.run_external_baseline_eval --summary-json outputs/wasp_stage1_summary.json --config configs/experiments/single_node_screening/wasp_s_jobs_screening.yaml --output-dir outputs/wasp_s_jobs_screening

    @'
import json
from pathlib import Path
Path("outputs").mkdir(exist_ok=True)
payload = [{"text": f"dpga synthetic sample {i} with enough words for screening"} for i in range(24)]
Path("outputs/dpga_epoch_all.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
'@ | conda run --no-capture-output -n $PretextEnv python -

    conda run --no-capture-output -n $PretextEnv python (Join-Path $DpgaRoot "main/run_paper_new_screening.py") --epoch-all-json outputs/dpga_epoch_all.json --output-json outputs/dpga_stage1_summary.json --budget 100
    conda run --no-capture-output -n $PretextEnv python -m paper_new_selector.run_external_baseline_eval --summary-json outputs/dpga_stage1_summary.json --config configs/experiments/single_node_screening/dpga_s_jobs_screening.yaml --output-dir outputs/dpga_s_jobs_screening
}
finally {
    Pop-Location
}
