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

Write-Host "=== Upload fixed llama2_eval.py ==="
$localFile = "D:\学习记录\导师项目\研究\caiqiyue_file\PrE-Text\pretext_platform\evaluation\llama2_eval.py"
$remoteFile = "/root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.py"
$tempB64 = "D:\temp_llama2_eval_b64.txt"

# Base64 encode locally
$content = Get-Content $localFile -Raw -Encoding UTF8
$bytes = [System.Text.Encoding]::UTF8.GetBytes($content)
$encoded = [Convert]::ToBase64String($bytes)
$encoded | Out-File -FilePath $tempB64 -Encoding ASCII

# Transfer via SSH
$cmdDecode = "echo `"$(Get-Content $tempB64 -Raw)`" | base64 -d > $remoteFile"
ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $cmdDecode 2>&1

Write-Host "=== Verify upload ==="
$verifyCmd = "wc -l /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.py"
ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $verifyCmd 2>&1

Write-Host ""
Write-Host "=== Setting load_in_8bit=False in config ==="
$cmdMod = "sed -i 's/load_in_8bit: true/load_in_8bit: false/' /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/configs/experiments/validate_jobs_tiny_thesis_platform.yaml"
ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $cmdMod 2>&1

Write-Host ""
Write-Host "=== Verify the change ==="
$cmdGrep = "grep 'load_in_8bit' /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/configs/experiments/validate_jobs_tiny_thesis_platform.yaml"
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
$cmdRestore = "sed -i 's/load_in_8bit: false/load_in_8bit: true/' /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/configs/experiments/validate_jobs_tiny_thesis_platform.yaml"
ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $cmdRestore 2>&1

Remove-Item $tempB64 -ErrorAction SilentlyContinue