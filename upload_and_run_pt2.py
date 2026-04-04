#!/usr/bin/env python3
"""Write llama2_eval.py content directly via SSH command."""
import subprocess
import os
import tempfile

# The file content
LOCAL_PATH = r"D:\学习记录\导师项目\研究\caiqiyue_file\PrE-Text\pretext_platform\evaluation\llama2_eval.py"
with open(LOCAL_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

print(f"File size: {len(content)} chars")

# Escape the content for safe shell insertion - use here-doc approach
# We'll write it using Python on the server side
# First, write the content to a temp file on local, then copy via base64

import base64
b64 = base64.b64encode(content.encode('utf-8')).decode('ascii')
print(f"Base64 size: {len(b64)} chars")

# Create a PS1 script that:
# 1. Writes the base64 to a temp file on server
# 2. Decodes it on server
# 3. Verifies

ps_script = f'''
$src = @'
using System;
public static class Program {{
    public static void Main() {{
        Console.Write("banpFV2mtSjO");
    }}
}}
'@
$asmPath = "C:\\temp\\ssh_askpass.exe"
Add-Type -TypeDefinition $src -Language CSharp -OutputAssembly $asmPath -OutputType ConsoleApplication

$env:SSH_ASKPASS = $asmPath
$env:SSH_ASKPASS_REQUIRE = 'force'
$env:DISPLAY = 'dummy:0'

# Step 1: Write base64 to temp file on server
$cmd1 = 'cat > /tmp/llama2_eval.b64 << '+"'"+'_B64EOF_'+"'"+'\n{b64}\n'+"'"+'_B64EOF_'+"'"+''
Write-Host "Step 1: Writing base64 to server..."
$null = ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $cmd1 2>&1

# Step 2: Decode on server
Write-Host "Step 2: Decoding..."
$cmd2 = 'base64 -d < /tmp/llama2_eval.b64 > /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.py && wc -l /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.py'
$null = ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $cmd2 2>&1

# Step 3: Verify
Write-Host "Step 3: Verifying..."
$cmd3 = 'grep -c "load_in_8bit" /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.py'
$result = ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $cmd3 2>&1
Write-Host "load_in_8bit count:" $result

$cmd4 = 'head -5 /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.py'
$null = ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $cmd4 2>&1
'''

ps1_path = os.path.join(tempfile.gettempdir(), 'upload_llama2.ps1')
with open(ps1_path, 'w', encoding='utf-8') as f:
    f.write(ps_script)

print(f"PS1 script: {ps1_path}")
result = subprocess.run(
    ['powershell.exe', '-ExecutionPolicy', 'Bypass', '-File', ps1_path],
    capture_output=True, text=True, timeout=60
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