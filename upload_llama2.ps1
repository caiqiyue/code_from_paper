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

# Use Unix-style path since PowerShell is running in Git Bash context
$localFile = "/d/学习记录/导师项目/研究/caiqiyue_file/PrE-Text/pretext_platform/evaluation/llama2_eval.py"
$bytes = [System.IO.File]::ReadAllBytes($localFile)
$b64 = [Convert]::ToBase64String($bytes)

Write-Host "=== Local file: $($bytes.Length) bytes ==="
Write-Host "=== Base64 length: $($b64.Length) chars ==="

# Upload via SSH with base64 decode
$uploadCmd = "echo '$b64' | base64 -d > /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.py && wc -l /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.py"
Write-Host "=== Uploading (this may take a moment) ==="
ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $uploadCmd 2>&1

Write-Host ""
Write-Host "=== Verify the fix is in place ==="
ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com "grep -n 'load_in_8bit\|quantization_config' /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.py | head -10" 2>&1