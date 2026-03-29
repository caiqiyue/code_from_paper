"""Test Qwen 2.0 5B model integrity after download."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

path = 'thesis_platform/open_model/qwen_2_0_5b_instruct'
print('=== File listing ===')
for f in sorted(os.listdir(path)):
    size = os.path.getsize(os.path.join(path, f))
    print(f'  {f}: {size:,} bytes')

print()
print('=== Tokenizer test ===')
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
text = 'Hello, this is a test of the Qwen tokenizer.'
ids = tok.encode(text)
decoded = tok.decode(ids).strip()
print(f'  encode: {ids}')
print(f'  decode: {repr(decoded)}')
print(f'  round-trip OK: {decoded == text}')

print()
print('=== Model loading test ===')
model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, device_map='cpu')
print(f'  model type: {model.__class__.__name__}')
print(f'  param count: {sum(p.numel() for p in model.parameters()):,}')

print()
print('=== Forward pass ===')
input_ids = tok(text, return_tensors='pt')
with torch.no_grad():
    output = model(**input_ids)
print(f'  output logits shape: {output.logits.shape}')
print()
print('ALL CHECKS PASSED')
