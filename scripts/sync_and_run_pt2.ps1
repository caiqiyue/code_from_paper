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

# Read local llama2_eval.py and base64 encode it
$localFile = "D:\学习记录\导师项目\研究\caiqiyue_file\PrE-Text\pretext_platform\evaluation\llama2_eval.py"
$bytes = [System.Text.Encoding]::UTF8.GetBytes((Get-Content $localFile -Raw))
$b64 = [Convert]::ToBase64String($bytes)

# Upload via SSH: decode base64 to file
$uploadCmd = "echo '$b64' | base64 -d > /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.py"

Write-Host "=== Uploading llama2_eval.py ==="
ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com "$uploadCmd" 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "Upload failed"
    exit 1
}

Write-Host "=== Running PT-2 pipeline ==="

$runCmd = "source /root/miniconda3/etc/profile.d/conda.sh && cd /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text && conda activate pretext && python -m pretext_platform.scripts.run_pipeline --config configs/experiments/validate_jobs_tiny_thesis_platform.yaml 2>&1"

ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com "$runCmd" 2>&1

Write-Host "=== Done, exit code: $LASTEXITCODE ==="