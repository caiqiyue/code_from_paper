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
source /root/miniconda3/etc/profile.d/conda.sh && cd /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text && conda activate pretext && python -m pretext_platform.scripts.run_pipeline --config configs/experiments/validate_jobs_tiny_thesis_platform.yaml 2>&1
"@

ssh -v `
  -o PreferredAuthentications=password `
  -o PubkeyAuthentication=no `
  -o NumberOfPasswordPrompts=1 `
  -o ConnectTimeout=30 `
  -o StrictHostKeyChecking=no `
  -p 26732 `
  root@connect.nmb1.seetacloud.com `
  $cmd