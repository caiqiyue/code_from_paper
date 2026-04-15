#!/usr/bin/env python3
"""Upload llama2_eval.py to server using SSH_ASKPASS via PowerShell."""
import subprocess
import base64
import os
import tempfile
import shutil

# Read local file
local_path = r"D:\学习记录\导师项目\研究\caiqiyue_file\PrE-Text\pretext_platform\evaluation\llama2_eval.py"
with open(local_path, 'rb') as f:
    content = f.read()

print(f"Local file: {len(content)} bytes")

# Base64 encode
b64 = base64.b64encode(content).decode('ascii')
print(f"Base64: {len(b64)} chars")

# Write a PowerShell script to a temp file
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

$uploadCmd = 'echo "' + b64 + '" | base64 -d > /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.py'
$null = ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $uploadCmd 2>&1

$checkCmd = 'wc -l /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.py'
$result = ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $checkCmd 2>&1
print($result)

$verifyCmd = 'grep -c "load_in_8bit" /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.py'
$verify = ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $verifyCmd 2>&1
print("load_in_8bit count:", $verify)
'''

# Write PS1 to temp file
ps1_path = os.path.join(tempfile.gettempdir(), 'upload_llama2.ps1')
with open(ps1_path, 'w', encoding='utf-8') as f:
    f.write(ps_script)

print(f"PowerShell script written to: {ps1_path}")

# Run PowerShell script
result = subprocess.run(
    ['powershell.exe', '-ExecutionPolicy', 'Bypass', '-File', ps1_path],
    capture_output=True, text=True
)
print(result.stdout)
if result.stderr:
    lines = result.stderr.strip().split('\n')
    errors = [l for l in lines if 'debug1' not in l and l.strip()]
    if errors:
        print('STDERR:', errors[:3])

# Cleanup
try:
    os.remove(ps1_path)
except:
    pass