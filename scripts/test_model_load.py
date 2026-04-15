#!/usr/bin/env python3
"""Test model loading with device_map, without PEFT."""
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

Write-Host "=== Test 1: Load model without device_map, then move to GPU ==="
$testCmd1 = @"
source /root/miniconda3/etc/profile.d/conda.sh && cd /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text && conda activate pretext && python -c '
import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: cuda:0")

model_path = \"/root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/models/llama_2_7b_hf\"
print(f\"Loading model from {model_path}...\")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    use_safetensors=True,
    torch_dtype=torch.float16,
)
print(f"Model loaded. Moving to GPU...")
model = model.to(\"cuda:0\")
print(f"Model is now on: {next(model.parameters()).device}")

# Test forward pass
tokenizer = LlamaTokenizer.from_pretrained(model_path, local_files_only=True)
inputs = tokenizer("Hello world", return_tensors=\"pt\").to(\"cuda:0\")
print(f"Inputs device: {inputs[\"input_ids\"].device}")
with torch.no_grad():
    outputs = model(**inputs)
print(f"Forward pass successful! Output device: {outputs.logits.device}")
' 2>&1
"@

$testResult1 = ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $testCmd1 2>&1
Write-Host $testResult1

Write-Host ""
Write-Host "=== Test 2: Load model with device_map ==="
$testCmd2 = @"
source /root/miniconda3/etc/profile.d/conda.sh && cd /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text && conda activate pretext && python -c '
import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer
print(f\"CUDA available: {torch.cuda.is_available()}\")

model_path = \"/root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/models/llama_2_7b_hf\"
print(f\"Loading model with device_map...\")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    use_safetensors=True,
    torch_dtype=torch.float16,
    device_map={\"\": \"cuda:0\"},
)
print(f\"Model is on: {next(model.parameters()).device}\")

# Check rotary embedding device
rotary = model.model.rotary_emb
print(f\"Rotary emb state_dict keys: {list(rotary.state_dict().keys())}\")
for k, v in rotary.state_dict().items():
    print(f\"  {k}: {v.device}\")

# Test forward pass
tokenizer = LlamaTokenizer.from_pretrained(model_path, local_files_only=True)
inputs = tokenizer(\"Hello world\", return_tensors=\"pt\").to(\"cuda:0\")
print(f\"Inputs device: {inputs[\\"input_ids\\"].device}\")
with torch.no_grad():
    outputs = model(**inputs)
print(f\"Forward pass successful! Output device: {outputs.logits.device}\")
' 2>&1
"@

$testResult2 = ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $testCmd2 2>&1
Write-Host $testResult2

Write-Host ""
Write-Host "=== Test 3: Load with device_map, then apply PEFT ==="
$testCmd3 = @"
source /root/miniconda3/etc/profile.d/conda.sh && cd /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text && conda activate pretext && python -c '
import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer
from peft import LoraConfig, get_peft_model
print(f\"CUDA available: {torch.cuda.is_available()}\")

model_path = \"/root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/models/llama_2_7b_hf\"
print(f\"Loading model with device_map...\")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    use_safetensors=True,
    torch_dtype=torch.float16,
    device_map={\"\": \"cuda:0\"},
)
print(f\"Model is on: {next(model.parameters()).device}\")

# Apply PEFT
peft_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.0,
    target_modules=[\"q_proj\", \"o_proj\", \"v_proj\", \"k_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"],
    bias=\"none\",
    task_type=\"CAUSAL_LM\",
)
print(\"Applying PEFT...\")
model = get_peft_model(model, peft_config)
print(f\"PEFT model is on: {next(model.parameters()).device}\")

# Check all devices in model
devices = set()
for p in model.parameters():
    devices.add(p.device)
print(f\"All parameter devices: {devices}\")

# Test forward pass
tokenizer = LlamaTokenizer.from_pretrained(model_path, local_files_only=True)
inputs = tokenizer(\"Hello world\", return_tensors=\"pt\").to(\"cuda:0\")
print(f\"Inputs device: {inputs[\\"input_ids\\"].device}\")
with torch.no_grad():
    outputs = model(**inputs)
print(f\"Forward pass successful! Output device: {outputs.logits.device}\")
' 2>&1
"@

$testResult3 = ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p 26732 root@connect.nmb1.seetacloud.com $testCmd3 2>&1
Write-Host $testResult3
'''

ps1_path = os.path.join(tempfile.gettempdir(), 'test_model_load.ps1')
with open(ps1_path, 'w', encoding='utf-8') as f:
    f.write(ps_script)

print("Testing model loading...")
result = subprocess.run(
    ['powershell.exe', '-ExecutionPolicy', 'Bypass', '-File', ps1_path],
    capture_output=True, text=True, timeout=300
)
print(result.stdout)
if result.stderr:
    lines = result.stderr.strip().split('\n')
    errors = [l for l in lines if l.strip() and 'debug1' not in l and 'Invoke-Expression' not in l]
    if errors:
        print('STDERR:', errors[:10])

try:
    os.remove(ps1_path)
except:
    pass