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

Write-Host "=== Test llama_2_7b_hf tokenizer ==="
$cmd = @"
source /root/miniconda3/etc/profile.d/conda.sh && conda activate pretext && cd /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text && python -c 'from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained("/root/autodl-tmp/caiqiyue/code_from_paper/thesis_platform/open_model/llama_2_7b_hf", local_files_only=True); print("OK:", type(tok).__name__)'
"@
Run-SSH $cmd

Write-Host ""
Write-Host "=== Test llama_3_2_3b_instruct tokenizer ==="
$cmd2 = @"
source /root/miniconda3/etc/profile.d/conda.sh && conda activate pretext && cd /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text && python -c 'from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained("/root/autodl-tmp/caiqiyue/code_from_paper/thesis_platform/open_model/llama_3_2_3b_instruct", local_files_only=True); print("OK:", type(tok).__name__)'
"@
Run-SSH $cmd2

Write-Host ""
Write-Host "=== Check tokenizer.json size ==="
Run-SSH "ls -la /root/autodl-tmp/caiqiyue/code_from_paper/thesis_platform/open_model/llama_2_7b_hf/tokenizer.json"