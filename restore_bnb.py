#!/usr/bin/env python3
"""Restore bitsandbytes to 0.49.2 and prepare monkey-patch approach."""
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

$cmd = "source /root/miniconda3/etc/profile.d/conda.sh && conda activate pretext && pip install 'bitsandbytes==0.49.2' 2>&1"

Write-Host "=== Restoring bitsandbytes ==="
$sshResult = ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $cmd 2>&1
Write-Host $sshResult

Write-Host ""
Write-Host "=== Check bitsandbytes version ==="
$cmd2 = "source /root/miniconda3/etc/profile.d/conda.sh && conda activate pretext && python -c 'import bitsandbytes; print(bitsandbytes.__version__)'"
$sshResult2 = ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $cmd2 2>&1
Write-Host $sshResult2
'''

ps1_path = os.path.join(tempfile.gettempdir(), 'restore_bnb.ps1')
with open(ps1_path, 'w', encoding='utf-8') as f:
    f.write(ps_script)

print("Restoring bitsandbytes...")
result = subprocess.run(
    ['powershell.exe', '-ExecutionPolicy', 'Bypass', '-File', ps1_path],
    capture_output=True, text=True, timeout=300
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