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

function Run-SSH {
    param($cmd)
    ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $cmd 2>&1
}

Write-Host "=== Create test script on server ==="
$testScript = @'
#!/usr/bin/env python3
import sys
sys.path.insert(0, "/root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text")

from pathlib import Path
from transformers import AutoTokenizer

models = {
    "llama_2_7b_hf": "/root/autodl-tmp/caiqiyue/code_from_paper/thesis_platform/open_model/llama_2_7b_hf",
    "llama_3_2_3b_instruct": "/root/autodl-tmp/caiqiyue/code_from_paper/thesis_platform/open_model/llama_3_2_3b_instruct",
    "llama_2_7b_chat": "/root/autodl-tmp/caiqiyue/code_from_paper/thesis_platform/open_model/Meta-Llama-2-7b-chat-hf",
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
    print(f"{name}: tokenizer.json size = {tok_file.stat().st_size} bytes")
    try:
        tok = AutoTokenizer.from_pretrained(str(p), local_files_only=True)
        print(f"  -> Load OK: {type(tok).__name__}")
    except Exception as e:
        print(f"  -> Load FAILED: {e}")
    print()
'@

# Upload the script using Python on server side
$uploadCmd = @"
python3 -c `
import base64, sys
script = `$(${testScript} | Out-String).Replace('\`n','').Replace('`r','')
# Actually let me just write it directly
print('test')
"
"@

# Simpler approach - just echo the script content to a file on server
$createScript = "cat > /tmp/test_tokenizers.py << 'ENDOFSCRIPT'
" + $testScript.Replace("'", "'\''").Replace("`n", "\n").Replace("`r", "") + "
ENDOFSCRIPT"

Run-SSH $createScript

Write-Host "=== Run test script ==="
Run-SSH "source /root/miniconda3/etc/profile.d/conda.sh && conda activate pretext && python3 /tmp/test_tokenizers.py"