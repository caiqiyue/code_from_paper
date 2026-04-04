#!/usr/bin/env python3
"""Test remaining models on remote server."""

import paramiko
import time

HOST = "connect.nmb1.seetacloud.com"
PORT = 26732
USERNAME = "root"
PASSWORD = "banpFV2mtSjO"


def run_cmd(client, cmd, timeout=120):
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    return out, err


def test_all_minilm_caiyiyue(client):
    print(f"\n{'='*70}")
    print("Testing: all_minilm_l6_v2 (using caiqiyue env)")
    print(f"{'='*70}")

    model_path = "/root/autodl-tmp/caiqiyue/code_from_paper/thesis_platform/open_model/all_minilm_l6_v2"
    test_script = f"""
import sys
try:
    from sentence_transformers import SentenceTransformer
    path = '{model_path}'
    print(f"Loading SentenceTransformer from {{path}}...")
    model = SentenceTransformer(path)
    print(f"Model loaded: {{type(model).__name__}}")
    embeddings = model.encode(["Hello, how are you?"])
    print(f"Embedding shape: {{embeddings.shape}}")
    print("RESULT: OK")
except Exception as e:
    import traceback
    print(f"RESULT: FAILED")
    print(f"ERROR: {{e}}")
    traceback.print_exc()
"""
    script_path = "/tmp/test_minilm.py"
    run_cmd(client, f"cat > {script_path} << 'PYEOF'\n{test_script}\nPYEOF")
    out, err = run_cmd(client, f"cd /root/autodl-tmp && /root/miniconda3/envs/caiqiyue/bin/python {script_path}", timeout=120)
    print(f"STDOUT:\n{out}")
    if err:
        print(f"STDERR:\n{err}")
    return "RESULT: OK" in out


def test_roberta(client):
    print(f"\n{'='*70}")
    print("Testing: roberta_large (using hidden states extraction, no generate())")
    print(f"{'='*70}")

    model_path = "/root/autodl-tmp/caiqiyue/code_from_paper/thesis_platform/open_model/roberta_large"
    test_script = f"""
import sys
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
    import torch
    path = '{model_path}'
    print(f"Loading {{path}}...")

    # Use RobertaForCausalLM instead of RobertaModel
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, device_map='auto', torch_dtype=torch.float16)
    print(f"Model type: {{type(model).__name__}}")
    print(f"Model loaded successfully!")

    inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    print(f"Logits shape: {{outputs.logits.shape}}")
    print(f"Model can do causal LM: True")
    print("RESULT: OK")
except Exception as e:
    import traceback
    print(f"RESULT: FAILED")
    print(f"ERROR: {{e}}")
    traceback.print_exc()
"""
    script_path = "/tmp/test_roberta.py"
    run_cmd(client, f"cat > {script_path} << 'PYEOF'\n{test_script}\nPYEOF")
    out, err = run_cmd(client, f"cd /root/autodl-tmp && /root/miniconda3/envs/pretext/bin/python {script_path}", timeout=180)
    print(f"STDOUT:\n{out}")
    if err:
        print(f"STDERR:\n{err}")
    return "RESULT: OK" in out


def main():
    print(f"Connecting to {HOST}:{PORT} ...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USERNAME, password=PASSWORD, timeout=30)

    # Check caiqiyue env exists
    out, _ = run_cmd(client, "/root/miniconda3/bin/conda env list")
    print(f"Conda envs:\n{out}")

    results = {}

    try:
        results["all_minilm_l6_v2 (caiqiyue)"] = test_all_minilm_caiyiyue(client)
    except Exception as e:
        print(f"Exception: {e}")
        results["all_minilm_l6_v2 (caiqiyue)"] = False

    time.sleep(2)

    try:
        results["roberta_large (pretext)"] = test_roberta(client)
    except Exception as e:
        print(f"Exception: {e}")
        results["roberta_large (pretext)"] = False

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, ok in results.items():
        print(f"  {name}: {'OK' if ok else 'FAILED'}")

    client.close()

if __name__ == "__main__":
    main()