#!/usr/bin/env python3
"""Check accelerate dispatch_model source."""
import subprocess
import os
import tempfile

ps_script = r'''
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

$cmd = "source /root/miniconda3/etc/profile.d/conda.sh && cd /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text && conda activate pretext && python -c 'from accelerate.big_modeling import dispatch_model; import inspect; lines = inspect.getsource(dispatch_model).split(chr(10)); print(chr(10).join(lines[:80]))'" 2>&1

$result = ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $cmd 2>&1
Write-Host $result
'''

ps1_path = os.path.join(tempfile.gettempdir(), 'check_acc.ps1')
with open(ps1_path, 'w', encoding='utf-8') as f:
    f.write(ps_script)

result = subprocess.run(
    ['powershell.exe', '-ExecutionPolicy', 'Bypass', '-File', ps1_path],
    capture_output=True, text=True, timeout=120
)
print(result.stdout)
if result.stderr:
    lines = result.stderr.strip().split('\n')
    errors = [l for l in lines if l.strip() and 'debug1' not in l]
    if errors:
        print('STDERR:', errors[:5])

try:
    os.remove(ps1_path)
except:
    pass