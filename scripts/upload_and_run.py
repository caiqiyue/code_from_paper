#!/usr/bin/env python3
"""Upload test script to server and run tokenizer tests."""
import subprocess
import base64
import sys

# The test script content
TEST_SCRIPT = b'''#!/usr/bin/env python3
import sys
sys.path.insert(0, "/root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text")

from pathlib import Path
from transformers import AutoTokenizer

models = {
    "llama_2_7b_hf": "/root/autodl-tmp/caiqiyue/code_from_paper/thesis_platform/open_model/llama_2_7b_hf",
    "llama_3_2_3b_instruct": "/root/autodl-tmp/caiqiyue/code_from_paper/thesis_platform/open_model/llama_3_2_3b_instruct",
    "Meta-Llama-2-7b-chat-hf": "/root/autodl-tmp/caiqiyue/code_from_paper/thesis_platform/open_model/Meta-Llama-2-7b-chat-hf",
}

for name, path in models.items():
    p = Path(path)
    if not p.exists():
        print(f"{name}: DIR NOT FOUND")
        continue
    tok_file = p / "tokenizer.json"
    if not tok_file.exists():
        print(f"{name}: tokenizer.json NOT FOUND")
        continue
    size = tok_file.stat().st_size
    print(f"{name}: tokenizer.json size = {size} bytes")
    try:
        tok = AutoTokenizer.from_pretrained(str(p), local_files_only=True)
        print(f"  -> Load OK: {type(tok).__name__}")
    except Exception as e:
        print(f"  -> Load FAILED: {type(e).__name__}: {e}")
    print()
'''

# Base64 encode the script
script_b64 = base64.b64encode(TEST_SCRIPT).decode('ascii')

# Build the SSH command
ssh_cmd = [
    'ssh',
    '-o', 'PreferredAuthentications=password',
    '-o', 'PubkeyAuthentication=no',
    '-o', 'NumberOfPasswordPrompts=1',
    '-o', 'ConnectTimeout=30',
    '-o', 'StrictHostKeyChecking=no',
    '-p', '26732',
    'root@connect.nmb1.seetacloud.com',
    f'echo "{script_b64}" | base64 -d > /tmp/test_tokenizers.py && '
    'source /root/miniconda3/etc/profile.d/conda.sh && '
    'conda activate pretext && '
    'python3 /tmp/test_tokenizers.py'
]

# For password, we need to use SSH_ASKPASS approach
# Let's use the C# compiled approach
import os
import tempfile

# Write the C# source
cs_code = r'''using System;
public static class Program {
    public static void Main() {
        Console.Write("banpFV2mtSjO");
    }
}
'''

tmp_dir = tempfile.gettempdir()
asm_path = os.path.join(tmp_dir, 'ssh_askpass.exe')

# Compile C# (requires csc.exe or mcs)
# On Windows, we can use Add-Type via PowerShell
ps_script = f'''
$src = @'
{cs_code}
'@
$asmPath = "{asm_path}"
Add-Type -TypeDefinition $src -Language CSharp -OutputAssembly $asmPath -OutputType ConsoleApplication
'''

# Run PowerShell to compile and set environment
subprocess.run(['powershell.exe', '-ExecutionPolicy', 'Bypass', '-Command', ps_script],
               capture_output=True)

os.environ['SSH_ASKPASS'] = asm_path
os.environ['SSH_ASKPASS_REQUIRE'] = 'force'
os.environ['DISPLAY'] = 'dummy:0'

print("Running tokenizer tests on server...")
result = subprocess.run(ssh_cmd, capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    # Filter out SSH debug info
    stderr_lines = result.stderr.split('\n')
    real_errors = [l for l in stderr_lines if 'debug1' not in l and l.strip()]
    if real_errors:
        print('STDERR:', '\n'.join(real_errors[:5]))