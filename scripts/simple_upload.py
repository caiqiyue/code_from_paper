#!/usr/bin/env python3
"""Simple file upload via SCP."""
import subprocess
import base64
import os
import tempfile
import shutil

# Read local file
LOCAL_PATH = r"D:\学习记录\导师项目\研究\caiqiyue_file\PrE-Text\pretext_platform\evaluation\llama2_eval.py"
REMOTE_PATH = "/root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.py"

with open(LOCAL_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

print(f"File size: {len(content)} chars")

# Base64 encode
b64 = base64.b64encode(content.encode('utf-8')).decode('ascii')
print(f"Base64 size: {len(b64)} chars")

# Write base64 to temp file
tmp_dir = tempfile.gettempdir()
b64_file = os.path.join(tmp_dir, 'llama2_eval.b64')
with open(b64_file, 'w') as f:
    f.write(b64)
print(f"Written base64 to: {b64_file}")

# Create password helper
cs_code = r'''using System;
public static class Program {
    public static void Main() {
        Console.Write("banpFV2mtSjO");
    }
}
'''
asm_path = os.path.join(tmp_dir, 'ssh_askpass.exe')

ps_compile = f'''
$src = @'
{cs_code}
'@
Add-Type -TypeDefinition $src -Language CSharp -OutputAssembly "{asm_path}" -OutputType ConsoleApplication
'''
comp_result = subprocess.run(['powershell.exe', '-ExecutionPolicy', 'Bypass', '-Command', ps_compile],
                            capture_output=True, text=True)
if comp_result.stderr:
    print("Compile stderr:", comp_result.stderr[:200])

# Set environment and use SCP
env = os.environ.copy()
env['SSH_ASKPASS'] = asm_path
env['SSH_ASKPASS_REQUIRE'] = 'force'
env['DISPLAY'] = 'dummy:0'

# SCP upload
scp_cmd = [
    'scp',
    '-o', 'PreferredAuthentications=password',
    '-o', 'PubkeyAuthentication=no',
    '-o', 'NumberOfPasswordPrompts=1',
    '-o', 'ConnectTimeout=30',
    '-o', 'StrictHostKeyChecking=no',
    '-P', '26732',
    b64_file,
    f'root@connect.nmb1.seetacloud.com:{REMOTE_PATH}.b64'
]

print(f"Running SCP: {' '.join(scp_cmd[:3])} ... {b64_file.split(os.sep)[-1]} ...")
result = subprocess.run(scp_cmd, env=env, capture_output=True, text=True, timeout=60)
print("SCP stdout:", result.stdout)
print("SCP stderr:", result.stderr[:500] if result.stderr else "none")
print("SCP returncode:", result.returncode)

# Now decode on server
decode_cmd = [
    'ssh',
    '-o', 'PreferredAuthentications=password',
    '-o', 'PubkeyAuthentication=no',
    '-o', 'NumberOfPasswordPrompts=1',
    '-o', 'ConnectTimeout=30',
    '-o', 'StrictHostKeyChecking=no',
    '-p', '26732',
    'root@connect.nmb1.seetacloud.com',
    f'base64 -d < {REMOTE_PATH}.b64 > {REMOTE_PATH} && echo OK'
]
print(f"Decoding on server...")
result2 = subprocess.run(decode_cmd, env=env, capture_output=True, text=True, timeout=60)
print("Decode stdout:", result2.stdout)
print("Decode stderr:", result2.stderr[:500] if result2.stderr else "none")

# Verify
verify_cmd = [
    'ssh',
    '-o', 'PreferredAuthentications=password',
    '-o', 'PubkeyAuthentication=no',
    '-o', 'NumberOfPasswordPrompts=1',
    '-o', 'ConnectTimeout=30',
    '-o', 'StrictHostKeyChecking=no',
    '-p', '26732',
    'root@connect.nmb1.seetacloud.com',
    f'wc -l {REMOTE_PATH} && grep -c load_in_8bit {REMOTE_PATH}'
]
print("Verifying...")
result3 = subprocess.run(verify_cmd, env=env, capture_output=True, text=True, timeout=30)
print("Verify:", result3.stdout)
if result3.stderr:
    print("Verify stderr:", result3.stderr[:300])

# Cleanup
try:
    os.remove(b64_file)
    os.remove(asm_path)
except:
    pass