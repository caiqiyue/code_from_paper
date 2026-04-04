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

Write-Host "=== Check validate_jobs_tiny_thesis_platform.yaml ==="
Run-SSH "cat /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/configs/experiments/validate_jobs_tiny_thesis_platform.yaml"

Write-Host ""
Write-Host "=== Check if eval_large is enabled in PT-2 config ==="
Run-SSH "grep -A5 'eval_large' /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/configs/experiments/validate_jobs_tiny_thesis_platform.yaml"