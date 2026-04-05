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

Write-Host "=== Check validate_tiny_complete_test.yaml ==="
Run-SSH "cat /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/configs/experiments/validate_tiny_complete_test.yaml"

Write-Host ""
Write-Host "=== Check model_paths.llama2_7b in the config ==="
Run-SSH "grep -r 'llama2_7b\|llama_2_7b' /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/core/types.py 2>&1 | head -20"