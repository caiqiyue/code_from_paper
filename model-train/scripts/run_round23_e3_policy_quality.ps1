param(
  [string]$Config = "configs/round23_e3_policy_quality_1200_all6.json",
  [string]$Python = $(if ($env:PYTHON) { $env:PYTHON } else { "python" }),
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ModelTrainRoot = Split-Path -Parent $ScriptDir
$ConfigPath = Join-Path $ModelTrainRoot $Config
$Cfg = Get-Content -Raw -Encoding UTF8 $ConfigPath | ConvertFrom-Json

$ArgsList = @(
  "eval_round23_e3_policy_quality.py",
  "--controller-context-table", $Cfg.controller_context_table,
  "--round19-replay-table", $Cfg.round19_replay_table,
  "--model-dir", $Cfg.model_dir,
  "--model-family", $Cfg.model_family,
  "--feature-version", $Cfg.feature_version,
  "--config", $Cfg.model_config,
  "--output-dir", $Cfg.output_dir,
  "--scope", $Cfg.scope
)

Push-Location $ModelTrainRoot
try {
  if ($DryRun) {
    Write-Host "Working directory: $ModelTrainRoot"
    Write-Host "Command: $Python $($ArgsList -join ' ')"
    exit 0
  }
  & $Python @ArgsList
  exit $LASTEXITCODE
}
finally {
  Pop-Location
}
