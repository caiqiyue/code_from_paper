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

Write-Host "=== Check file size and lines ==="
$cmd = "wc -l /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.py"
$result = ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $cmd 2>&1
Write-Host "Lines: $result"

Write-Host ""
Write-Host "=== Check for load_in_8bit ==="
$cmd2 = "grep -c 'load_in_8bit' /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.py"
$result2 = ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $cmd2 2>&1
Write-Host "Count: $result2"

Write-Host ""
Write-Host "=== Check lines around 148-155 ==="
$cmd3 = "sed -n '148,155p' /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.py"
$result3 = ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $cmd3 2>&1
Write-Host $result3