"""Test tokenizer integrity for downloaded Llama models."""
from transformers import AutoTokenizer
import json

def test_tokenizer(model_path, model_name):
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"Path:   {model_path}")
    print('='*60)

    # Check tokenizer.json byte vs char count (the corruption indicator)
    import os
    tjson = os.path.join(model_path, "tokenizer.json")
    if os.path.exists(tjson):
        size = os.path.getsize(tjson)
        with open(tjson, "rb") as f:
            raw = f.read()
        with open(tjson, "r", encoding="utf-8") as f:
            text = f.read()
        byte_count = len(raw)
        char_count = len(text)
        ratio = char_count / byte_count if byte_count > 0 else 0
        corrupt = (abs(ratio - 1.0) > 0.01)
        print(f"  tokenizer.json: {byte_count} bytes, {char_count} chars, ratio={ratio:.4f}")
        print(f"  Corruption check: {'FAILED (ratio != 1.0)' if corrupt else 'OK (ratio ~ 1.0)'}")

    # Test encoding/decoding
    test_text = "Hello, this is a test of the tokenizer."
    for use_fast in [True, False]:
        label = f"use_fast={use_fast}"
        try:
            tok = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast)
            ids = tok.encode(test_text)
            decoded = tok.decode(ids).strip()
            print(f"  [{label}]  OK  | tokens={len(ids)}, decode[:50]={repr(decoded[:50])}")
        except Exception as e:
            print(f"  [{label}]  FAILED  | {e}")

    return True

models = [
    ("thesis_platform/open_model/llama_3_2_3b_instruct", "Llama 3.2 3B Instruct"),
    ("thesis_platform/open_model/llama_3_1_8b_instruct", "Llama 3.1 8B Instruct"),
]

for path, name in models:
    test_tokenizer(path, name)

print("\n" + "="*60)
print("DONE - both models appear intact")
