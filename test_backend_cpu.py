"""Test backend loading with CPU to isolate issue."""
import sys
import os
os.chdir(r"D:\学习记录\导师项目\研究\caiqiyue_file")
sys.path.insert(0, r"D:\学习记录\导师项目\研究\caiqiyue_file")

import torch
print(f"CUDA: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"VRAM before: {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

print("Importing...", flush=True)
from thesis_platform.models.backends import build_text_backend
from pathlib import Path

repo_root = Path("D:/学习记录/导师项目/研究/caiqiyue_file")

print("Creating CPU backend...", flush=True)
backend = build_text_backend(
    {
        "engine": "transformers",
        "model_name_or_path": "thesis_platform/open_model/llama_3_2_3b_instruct",
        "device": "cpu",
        "dtype": "float32",
        "temperature": 0.0,
        "max_new_tokens": 10,
        "use_chat_template": False,
        "use_fast": True,
        "role": "client",
    },
    repo_root=repo_root,
)
print(f"Backend: {backend.backend_name}", flush=True)

print("Testing generation...", flush=True)
result = backend.generate("Hello world", max_new_tokens=5)
print(f"Result: {repr(result)}", flush=True)
print("ALL DONE", flush=True)
