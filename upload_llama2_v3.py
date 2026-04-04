#!/usr/bin/env python3
"""Upload llama2_eval.py using chunked base64 transfer."""
import subprocess
import base64
import os
import tempfile

# Read local file
LOCAL_PATH = r"D:\学习记录\导师项目\研究\caiqiyue_file\PrE-Text\pretext_platform\evaluation\llama2_eval.py"
with open(LOCAL_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

print(f"File size: {len(content)} chars")

# Base64 encode
b64 = base64.b64encode(content.encode('utf-8')).decode('ascii')
print(f"Base64 size: {len(b64)} chars")

# Split into chunks of 2000 chars
CHUNK_SIZE = 2000
chunks = [b64[i:i+CHUNK_SIZE] for i in range(0, len(b64), CHUNK_SIZE)]
print(f"Split into {len(chunks)} chunks")

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

# Clear existing file and start fresh
$null = ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com "echo -n '' > /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.py" 2>&1

Write-Host "Transferring base64 in chunks..."
'''

ps1_path = os.path.join(tempfile.gettempdir(), 'upload_llama2_v3.ps1')
with open(ps1_path, 'w', encoding='utf-8') as f:
    f.write(ps_script)

result = subprocess.run(
    ['powershell.exe', '-ExecutionPolicy', 'Bypass', '-File', ps1_path],
    capture_output=True, text=True, timeout=60
)
print(result.stdout)

# Now upload each chunk
for i, chunk in enumerate(chunks):
    # Use Python subprocess to call SSH with the chunk
    chunk_escaped = chunk.replace("'", "'\"'\"'")
    cmd = f"ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com \"echo -n '{chunk_escaped}' >> /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.b64\" 2>&1"

    # Build a PS1 for this chunk
    chunk_ps = rf'''
$env:SSH_ASKPASS = "C:\temp\ssh_askpass.exe"
$env:SSH_ASKPASS_REQUIRE = 'force'
$env:DISPLAY = 'dummy:0'
$cmd = 'echo -n ''{chunk_escaped}'' >> /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.b64'
$null = ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $cmd 2>&1
Write-Host "Chunk {i+1}/{len(chunks)} done"
'''
    chunk_ps_path = os.path.join(tempfile.gettempdir(), f'chunk_{i}.ps1')
    with open(chunk_ps_path, 'w', encoding='utf-8') as f:
        f.write(chunk_ps)

    result = subprocess.run(
        ['powershell.exe', '-ExecutionPolicy', 'Bypass', '-File', chunk_ps_path],
        capture_output=True, text=True, timeout=30
    )
    if result.stdout.strip():
        print(result.stdout.strip())

    try:
        os.remove(chunk_ps_path)
    except:
        pass

# Decode
decode_ps = r'''
$env:SSH_ASKPASS = "C:\temp\ssh_askpass.exe"
$env:SSH_ASKPASS_REQUIRE = 'force'
$env:DISPLAY = 'dummy:0'
$cmd = 'base64 -d < /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.b64 > /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.py && wc -l /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.py'
$null = ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $cmd 2>&1
Write-Host "Decode result:" $null
$cmd2 = 'wc -l /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.py'
$null = ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $cmd2 2>&1
Write-Host "Lines:" $null
$cmd3 = 'grep -c "load_in_8bit" /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.py'
$null = ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $cmd3 2>&1
Write-Host "load_in_8bit count:" $null
$cmd4 = 'sed -n '148,155p' /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.py'
$null = ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $cmd4 2>&1
Write-Host "Lines 148-155:" $null
'''
decode_path = os.path.join(tempfile.gettempdir(), 'decode.ps1')
with open(decode_path, 'w', encoding='utf-8') as f:
    f.write(decode_ps)

print("\nDecoding...")
result = subprocess.run(
    ['powershell.exe', '-ExecutionPolicy', 'Bypass', '-File', decode_path],
    capture_output=True, text=True, timeout=60
)
print(result.stdout)
if result.stderr:
    lines = result.stderr.strip().split('\n')
    errors = [l for l in lines if l.strip() and 'debug1' not in l]
    if errors:
        print('STDERR:', errors[:3])

try:
    os.remove(decode_path)
    os.remove(ps1_path)
except:
    pass