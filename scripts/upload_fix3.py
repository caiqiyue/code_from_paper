#!/usr/bin/env python3
"""Upload fixed llama2_eval.py via SCP."""
import subprocess
import base64
import os
import tempfile

LOCAL_PATH = r"D:\学习记录\导师项目\研究\caiqiyue_file\PrE-Text\pretext_platform\evaluation\llama2_eval.py"
REMOTE_PATH = "/root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.py"

with open(LOCAL_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

print(f"File size: {len(content)} chars")
b64 = base64.b64encode(content.encode('utf-8')).decode('ascii')
print(f"Base64 size: {len(b64)} chars")

tmp_dir = tempfile.gettempdir()
b64_file = os.path.join(tmp_dir, 'llama2_eval.b64')
with open(b64_file, 'w') as f:
    f.write(b64)

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
subprocess.run(['powershell.exe', '-ExecutionPolicy', 'Bypass', '-Command', ps_compile],
             capture_output=True, text=True)

env = os.environ.copy()
env['SSH_ASKPASS'] = asm_path
env['SSH_ASKPASS_REQUIRE'] = 'force'
env['DISPLAY'] = 'dummy:0'

scp_cmd = [
    'scp', '-o', 'PreferredAuthentications=password', '-o', 'PubkeyAuthentication=no',
    '-o', 'NumberOfPasswordPrompts=1', '-o', 'ConnectTimeout=30',
    '-o', 'StrictHostKeyChecking=no', '-P', '26732',
    b64_file, f'root@connect.nmb1.seetacloud.com:{REMOTE_PATH}.b64'
]
print("Uploading...")
result = subprocess.run(scp_cmd, env=env, capture_output=True, text=True, timeout=60)
print("SCP:", result.returncode, result.stderr[:200] if result.stderr else "OK")

decode_cmd = [
    'ssh', '-o', 'PreferredAuthentications=password', '-o', 'PubkeyAuthentication=no',
    '-o', 'NumberOfPasswordPrompts=1', '-o', 'ConnectTimeout=30',
    '-o', 'StrictHostKeyChecking=no', '-p', '26732',
    'root@connect.nmb1.seetacloud.com',
    f'base64 -d < {REMOTE_PATH}.b64 > {REMOTE_PATH} && wc -l {REMOTE_PATH}'
]
result2 = subprocess.run(decode_cmd, env=env, capture_output=True, text=True, timeout=60)
print("Decode:", result2.stdout.strip())

# Quick syntax check
verify_cmd = [
    'ssh', '-o', 'PreferredAuthentications=password', '-o', 'PubkeyAuthentication=no',
    '-o', 'NumberOfPasswordPrompts=1', '-o', 'ConnectTimeout=30',
    '-o', 'StrictHostKeyChecking=no', '-p', '26732',
    'root@connect.nmb1.seetacloud.com',
    f'source /root/miniconda3/etc/profile.d/conda.sh && conda activate pretext && python3 -c "import ast; ast.parse(open(\'{REMOTE_PATH}\').read())" && echo SYNTAX_OK'
]
result3 = subprocess.run(verify_cmd, env=env, capture_output=True, text=True, timeout=60)
print("Syntax:", result3.stdout.strip())

try:
    os.remove(b64_file)
    os.remove(asm_path)
except:
    pass