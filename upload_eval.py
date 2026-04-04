#!/usr/bin/env python3
"""Upload fixed llama2_eval.py to server."""
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

Write-Host "=== Upload llama2_eval.py via SCP ==="
$cmd = "scp -P 26732 -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no 'D:\学习记录\导师项目\研究\caiqiyue_file\PrE-Text\pretext_platform\evaluation\llama2_eval.py' root@connect.nmb1.seetacloud.com:/root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.py 2>&1"
$result = Invoke-Expression $cmd
Write-Host $result

Write-Host ""
Write-Host "=== Verify ==="
$verifyCmd = "ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com 'wc -l /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.py && head -5 /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.py''"
$verifyResult = Invoke-Expression $verifyCmd
Write-Host $verifyResult

Write-Host ""
Write-Host "=== Setting load_in_8bit=False ==="
$cmdMod = "ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com 'sed -i \"s/load_in_8bit: true/load_in_8bit: false/\" /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/configs/experiments/validate_jobs_tiny_thesis_platform.yaml' 2>&1"
ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $cmdMod 2>&1

Write-Host ""
Write-Host "=== Verify load_in_8bit ==="
$cmdGrep = "ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com 'grep load_in_8bit /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/configs/experiments/validate_jobs_tiny_thesis_platform.yaml' 2>&1"
ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $cmdGrep 2>&1

Write-Host ""
Write-Host "=== Running PT-2 with load_in_8bit=False ==="
$runCmd = "source /root/miniconda3/etc/profile.d/conda.sh && cd /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text && conda activate pretext && python -m pretext_platform.scripts.run_pipeline --config configs/experiments/validate_jobs_tiny_thesis_platform.yaml 2>&1"
$sshResult = ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $runCmd 2>&1
Write-Host $sshResult

if ($sshResult -match "metrics_summary.json") {
    Write-Host ""
    Write-Host "=== PT-2 (no 8-bit): SUCCESS ==="
} elseif ($sshResult -match "Error|Traceback|CUDA|OOM|OutOfMemory|Exception") {
    Write-Host ""
    Write-Host "=== PT-2 (no 8-bit): FAILED ==="
} else {
    Write-Host ""
    Write-Host "=== PT-2 (no 8-bit): Check output manually ==="
}

# Restore config
$cmdRestore = "ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com 'sed -i \"s/load_in_8bit: false/load_in_8bit: true/\" /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/configs/experiments/validate_jobs_tiny_thesis_platform.yaml' 2>&1"
ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $cmdRestore 2>&1
'''

ps1_path = os.path.join(tempfile.gettempdir(), 'upload_eval.ps1')
with open(ps1_path, 'w', encoding='utf-8') as f:
    f.write(ps_script)

print("Uploading fixed llama2_eval.py...")
result = subprocess.run(
    ['powershell.exe', '-ExecutionPolicy', 'Bypass', '-File', ps1_path],
    capture_output=True, text=True, timeout=600
)
print(result.stdout)
if result.stderr:
    lines = result.stderr.strip().split('\n')
    errors = [l for l in lines if l.strip() and 'debug1' not in l]
    if errors:
        print('STDERR:', errors[:5])

try:
    os.remove(ps1_path)
except:
    pass