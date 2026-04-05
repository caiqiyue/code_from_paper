#!/usr/bin/env python3
"""Upload fixed llama2_eval.py and run PT-2 test via PowerShell with SSH_ASKPASS."""
import subprocess
import os
import tempfile

ps_script = r'''
$src = @'
using System;
public static class Program {
    public static void Main() {
        Console.Write("banpFV2mtSjO");
    }
}
'@
$asmPath = "C:\temp\ssh_askpass.exe"
Add-Type -TypeDefinition $src -Language CSharp -OutputAssembly $asmPath -OutputType ConsoleApplication

$env:SSH_ASKPASS = $asmPath
$env:SSH_ASKPASS_REQUIRE = 'force'
$env:DISPLAY = 'dummy:0'

Write-Host "=== Step 1: Verify local file exists ==="
$localFile = "C:\temp\llama2_eval.py"
if (Test-Path $localFile) {
    $lines = (Get-Content $localFile -Raw).Split("`n").Count
    Write-Host "Local file exists, lines: $lines"
} else {
    Write-Host "ERROR: Local file not found at $localFile"
    exit 1
}

Write-Host ""
Write-Host "=== Step 2: Encode local file as base64 ==="
$bytes = [System.IO.File]::ReadAllBytes($localFile)
$b64 = [Convert]::ToBase64String($bytes)
$b64File = Join-Path $env:TEMP "llama2_eval_b64.txt"
[System.IO.File]::WriteAllText($b64File, $b64)
Write-Host "Base64 written to: $b64File"

Write-Host ""
Write-Host "=== Step 3: Transfer base64 via SCP ==="
$scpCmd = "scp -P 26732 -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no `"$b64File`" root@connect.nmb1.seetacloud.com:/tmp/llama2_eval_b64.txt 2>&1"
Write-Host "Running SCP..."
$scpResult = Invoke-Expression $scpCmd
Write-Host $scpResult

Write-Host ""
Write-Host "=== Step 4: Decode on server ==="
$remoteFile = "/root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.py"
$decodeCmd = "ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com `"cat /tmp/llama2_eval_b64.txt | base64 -d > $remoteFile && rm /tmp/llama2_eval_b64.txt`" 2>&1"
Write-Host "Decoding on server..."
$decodeResult = Invoke-Expression $decodeCmd
Write-Host $decodeResult

Write-Host ""
Write-Host "=== Step 5: Verify upload ==="
$verifyCmd = "ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com `"wc -l $remoteFile && grep -n 'device_map.*accelerator.device' $remoteFile | head -5`" 2>&1"
$verifyResult = Invoke-Expression $verifyCmd
Write-Host $verifyResult

Write-Host ""
Write-Host "=== Step 6: Modify config to load_in_8bit=False ==="
$modCmd = "ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com `"sed -i 's/load_in_8bit: true/load_in_8bit: false/' /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/configs/experiments/validate_jobs_tiny_thesis_platform.yaml`" 2>&1"
$modResult = Invoke-Expression $modCmd
Write-Host $modResult

$grepCmd = "ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com `"grep load_in_8bit /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/configs/experiments/validate_jobs_tiny_thesis_platform.yaml`" 2>&1"
$grepResult = Invoke-Expression $grepCmd
Write-Host "load_in_8bit config: $grepResult"

Write-Host ""
Write-Host "=== Step 7: Run PT-2 with load_in_8bit=False ==="
$runCmd = "source /root/miniconda3/etc/profile.d/conda.sh && cd /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text && conda activate pretext && python -m pretext_platform.scripts.run_pipeline --config configs/experiments/validate_jobs_tiny_thesis_platform.yaml 2>&1"
$runFullCmd = "ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com `"$runCmd`" 2>&1"
$runResult = Invoke-Expression $runFullCmd
Write-Host $runResult

if ($runResult -match "metrics_summary.json") {
    Write-Host ""
    Write-Host "=== PT-2 (no 8-bit): SUCCESS ==="
} elseif ($runResult -match "Error|Traceback|CUDA|OOM|OutOfMemory|Exception") {
    Write-Host ""
    Write-Host "=== PT-2 (no 8-bit): FAILED ==="
} else {
    Write-Host ""
    Write-Host "=== PT-2 (no 8-bit): Check output manually ==="
}

Write-Host ""
Write-Host "=== Step 8: Restore config ==="
$restoreCmd = "ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com `"sed -i 's/load_in_8bit: false/load_in_8bit: true/' /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/configs/experiments/validate_jobs_tiny_thesis_platform.yaml`" 2>&1"
$restoreResult = Invoke-Expression $restoreCmd
Write-Host $restoreResult
Write-Host "Done."
'''

ps1_path = os.path.join(tempfile.gettempdir(), 'run_pt2_fix5.ps1')
with open(ps1_path, 'w', encoding='utf-8') as f:
    f.write(ps_script)

print("Running PT-2 fix script via PowerShell...")
result = subprocess.run(
    ['powershell.exe', '-ExecutionPolicy', 'Bypass', '-File', ps1_path],
    capture_output=True, text=True, timeout=900
)
print(result.stdout)
if result.stderr:
    lines = result.stderr.strip().split('\n')
    errors = [l for l in lines if l.strip() and 'debug1' not in l and 'Invoke-Expression' not in l]
    if errors:
        print('STDERR:', errors[:10])

try:
    os.remove(ps1_path)
except:
    pass