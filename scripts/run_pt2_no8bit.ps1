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

$cmd = @"
source /root/miniconda3/etc/profile.d/conda.sh && cd /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text && conda activate pretext && python -c 'from pretext_platform.evaluation.llama2_eval import run_llama2_eval; from pretext_platform.core.config import load_config; config = load_config(\"configs/experiments/validate_jobs_tiny_thesis_platform.yaml\"); config.eval_large[\"load_in_8bit\"] = False; print(\"Will run with load_in_8bit=False\")' 2>&1
"@

Write-Host "=== Setting load_in_8bit=False in config ==="
ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $cmd 2>&1

# Modify config to set load_in_8bit=False
$cmd2 = "sed -i 's/load_in_8bit: true/load_in_8bit: false/' /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/configs/experiments/validate_jobs_tiny_thesis_platform.yaml"
ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $cmd2 2>&1

Write-Host ""
Write-Host "=== Verify the change ==="
ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com "grep 'load_in_8bit' /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/configs/experiments/validate_jobs_tiny_thesis_platform.yaml" 2>&1

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
$cmd3 = "sed -i 's/load_in_8bit: false/load_in_8bit: true/' /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/configs/experiments/validate_jobs_tiny_thesis_platform.yaml"
ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $cmd3 2>&1