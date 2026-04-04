#!/usr/bin/env python3
"""Test each model on remote server by actually loading and generating."""

import paramiko
import time

HOST = "connect.nmb1.seetacloud.com"
PORT = 26732
USERNAME = "root"
PASSWORD = "banpFV2mtSjO"

MODEL_PATHS = {
    "llama_3_2_3b_instruct": "/root/autodl-tmp/caiqiyue/code_from_paper/thesis_platform/open_model/llama_3_2_3b_instruct",
    "llama_3_1_8b_instruct": "/root/autodl-tmp/caiqiyue/code_from_paper/thesis_platform/open_model/llama_3_1_8b_instruct",
    "Meta-Llama-3-8B": "/root/autodl-tmp/caiqiyue/code_from_paper/thesis_platform/open_model/Meta-Llama-3-8B",
    "llama_2_7b_hf": "/root/autodl-tmp/caiqiyue/code_from_paper/thesis_platform/open_model/llama_2_7b_hf",
    "qwen_2_0_5b_instruct": "/root/autodl-tmp/caiqiyue/code_from_paper/thesis_platform/open_model/qwen_2_0_5b_instruct",
    "distilgpt2": "/root/autodl-tmp/caiqiyue/code_from_paper/thesis_platform/open_model/distilgpt2",
    "roberta_large": "/root/autodl-tmp/caiqiyue/code_from_paper/thesis_platform/open_model/roberta_large",
    "all_minilm_l6_v2": "/root/autodl-tmp/caiqiyue/code_from_paper/thesis_platform/open_model/all_minilm_l6_v2",
}

TEST_PROMPT = "Hello, how are you?"

def run_cmd(client, cmd, timeout=120):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    return out, err


def test_model(client, model_name, model_path):
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"Path: {model_path}")
    print(f"{'='*70}")

    # First check if model files exist
    out, err = run_cmd(client, f"ls -la '{model_path}'/ 2>&1 | head -20")
    print(f"Files:\n{out}")

    # Test script
    test_script = f"""
import sys
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
    import torch

    path = '{model_path}'
    print(f"Loading model from {{path}}...")

    # Auto-detect model type
    if any(x in path.lower() for x in ['llama', 'qwen', 'llama2', 'llama3']):
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=True, device_map='auto', torch_dtype=torch.float16)
    elif 'roberta' in path.lower():
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModel.from_pretrained(path, local_files_only=True, device_map='auto')
    elif 'minilm' in path.lower():
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(path)
        print(f"SUCCESS: {{type(model).__name__}} loaded!")
        print(f"Model: {{model}}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=True, device_map='auto', torch_dtype=torch.float16)

    print(f"Model type: {{type(model).__name__}}")
    print(f"Model loaded successfully!")

    # Generate a response
    if 'minilm' not in path.lower():
        inputs = tokenizer("{TEST_PROMPT}", return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Prompt: {TEST_PROMPT}")
        print(f"Response: {{response}}")
    else:
        embeddings = model.encode(["{TEST_PROMPT}"])
        print(f"Embedding shape: {{embeddings.shape}}")

    print("RESULT: OK")
except Exception as e:
    import traceback
    print(f"RESULT: FAILED")
    print(f"ERROR: {{e}}")
    traceback.print_exc()
"""

    # Write test script
    script_path = "/tmp/test_model.py"
    escaped_script = test_script.replace('"', '\\"').replace('\n', '\\n')
    run_cmd(client, f"cat > {script_path} << 'PYEOF'\n{test_script}\nPYEOF")

    print("Running test...")
    out, err = run_cmd(client, f"cd /root/autodl-tmp && /root/miniconda3/envs/pretext/bin/python {script_path}", timeout=300)
    print(f"STDOUT:\n{out}")
    if err:
        print(f"STDERR:\n{err}")

    if "RESULT: OK" in out:
        print(f"\n>>> {model_name}: OK")
        return True
    else:
        print(f"\n>>> {model_name}: FAILED")
        return False


def main():
    print(f"Connecting to {HOST}:{PORT} ...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USERNAME, password=PASSWORD, timeout=30)

    results = {}
    for model_name, model_path in MODEL_PATHS.items():
        try:
            results[model_name] = test_model(client, model_name, model_path)
        except Exception as e:
            print(f"Exception testing {model_name}: {e}")
            results[model_name] = False
        time.sleep(2)

    print("\n\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name}: {status}")

    client.close()

if __name__ == "__main__":
    main()