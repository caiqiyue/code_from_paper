"""Test GPU model loading fresh (after clearing GPU state)."""
import sys
import os
os.chdir(r"D:\学习记录\导师项目\研究\caiqiyue_file")
sys.path.insert(0, r"D:\学习记录\导师项目\研究\caiqiyue_file")

import gc
import torch

print(f"CUDA: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print(f"VRAM after cleanup: {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

# Clear any cached model state
import thesis_platform.models.backends as bb
bb._MODEL_CACHE.clear()
print("Model cache cleared", flush=True)

print("Loading model with device='cuda' (not auto)...", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "thesis_platform/open_model/llama_3_2_3b_instruct"
tok = AutoTokenizer.from_pretrained(model_path)
print("Tokenizer loaded", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="cuda:0",
    low_cpu_mem_usage=True,
)
print(f"Model loaded, device: {next(model.parameters()).device}", flush=True)

# Test generation
print("Testing generation...", flush=True)
inputs = tok("Hello world", return_tensors="pt").to("cuda:0")
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=5, do_sample=False)
result = tok.decode(output[0], skip_special_tokens=True)
print(f"Generated: {repr(result)}", flush=True)
print("ALL PASSED", flush=True)
