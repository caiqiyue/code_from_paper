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

# Create the test script content
$scriptContent = @'
#!/usr/bin/env python3
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
'@

# Write script to local temp file first
$localScript = "C:\temp\test_tokenizers.py"
$scriptContent | Out-File -FilePath $localScript -Encoding UTF8

# Read and base64 encode
$bytes = [System.IO.File]::ReadAllBytes($localScript)
$b64 = [Convert]::ToBase64String($bytes)

Write-Host "=== Uploading test script (base64, $($bytes.Length) bytes) ==="

# Upload via SSH
$uploadCmd = "echo '$b64' | base64 -d > /tmp/test_tokenizers.py && echo 'Upload OK'"
ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $uploadCmd 2>&1

Write-Host ""
Write-Host "=== Running tokenizer tests ==="
$runCmd = "source /root/miniconda3/etc/profile.d/conda.sh && conda activate pretext && python3 /tmp/test_tokenizers.py"
ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $runCmd 2>&1

# Cleanup
Remove-Item $localScript -ErrorAction SilentlyContinue