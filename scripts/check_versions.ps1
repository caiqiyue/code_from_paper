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

Write-Host "=== Check accelerate version in pretext env ==="
ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com "source /root/miniconda3/etc/profile.d/conda.sh && conda activate pretext && python -c 'import accelerate; print(accelerate.__version__)'" 2>&1

Write-Host ""
Write-Host "=== Check transformers version ==="
ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com "source /root/miniconda3/etc/profile.d/conda.sh && conda activate pretext && python -c 'import transformers; print(transformers.__version__)'" 2>&1

Write-Host ""
Write-Host "=== Check bitsandbytes version ==="
ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com "source /root/miniconda3/etc/profile.d/conda.sh && conda activate pretext && python -c 'import bitsandbytes; print(bitsandbytes.__version__)'" 2>&1