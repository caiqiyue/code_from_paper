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

function Run-SSH {
    param($cmd)
    ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $cmd 2>&1
}

Write-Host "=== PT-MinFull: PrE-Text minimal full pipeline test ==="
Write-Host ""

$cmd1 = "source /root/miniconda3/etc/profile.d/conda.sh && cd /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text && conda activate pretext && python -m pretext_platform.scripts.run_pipeline --config configs/experiments/validate_tiny_complete_test.yaml 2>&1"
$result1 = Run-SSH $cmd1
Write-Host $result1

# Check if the result contains success indicators
if ($result1 -match "metrics_summary.json" -or $result1 -match "eval_large" -and $result1 -notmatch "Error" -and $result1 -notmatch "Traceback") {
    Write-Host ""
    Write-Host "=== PT-MinFull: SUCCESS ==="
} elseif ($result1 -match "Error|Traceback|CUDA|OOM|OutOfMemory") {
    Write-Host ""
    Write-Host "=== PT-MinFull: FAILED - See output above ==="
    Write-Host "Error details captured."
} else {
    Write-Host ""
    Write-Host "=== PT-MinFull: Output unclear, check manually ==="
}

Write-Host ""
Write-Host "=== Test run complete ==="