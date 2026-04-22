$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $root
$log = Join-Path $root 'old_automation\old_experiment_queue.scheduler.log'
while ($true) {
    & 'D:\python311\python.exe' (Join-Path $root 'old_automation\old_experiment_queue.py') *>> $log
    Start-Sleep -Seconds 1800
}
