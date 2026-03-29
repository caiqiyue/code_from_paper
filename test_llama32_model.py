"""Test llama_3_2_3b model loading."""
import sys
sys.path.insert(0, ".")
import os
os.chdir(r"D:\学习记录\导师项目\研究\caiqiyue_file")

print("Testing llama_3_2_3b model loading...", flush=True)

import torch
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}", flush=True)
print(f"torch.version.cuda: {torch.version.cuda}", flush=True)

model_path = "thesis_platform/open_model/llama_3_2_3b_instruct"

# Check safetensors files
import os
files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
print(f"Safetensors files: {files}", flush=True)
for f in files:
    size = os.path.getsize(os.path.join(model_path, f))
    print(f"  {f}: {size:,} bytes", flush=True)

print("\nLoading with AutoModelForCausalLM...", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained(model_path)
print("Tokenizer loaded", flush=True)

print("Loading model with device_map='auto'...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
print("Model loaded!", flush=True)
print(f"Model device: {next(model.parameters()).device}", flush=True)

# Test generation
print("\nTesting generation...", flush=True)
text = "Hello, this is a test."
inputs = tok(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
result = tok.decode(output[0], skip_special_tokens=True)
print(f"Generated: {repr(result)}", flush=True)
print("\nALL CHECKS PASSED", flush=True)
