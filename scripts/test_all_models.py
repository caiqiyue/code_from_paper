#!/usr/bin/env python3
"""Test all models in both caiqiyue and pretext environments."""

import paramiko
import time

HOST = "connect.nmb1.seetacloud.com"
PORT = 26732
USERNAME = "root"
PASSWORD = "banpFV2mtSjO"

MODEL_PATHS = {
    "Meta-Llama-3-8B": "/root/autodl-tmp/caiqiyue/code_from_paper/thesis_platform/open_model/Meta-Llama-3-8B",
    "qwen_2_0_5b_instruct": "/root/autodl-tmp/caiqiyue/code_from_paper/thesis_platform/open_model/qwen_2_0_5b_instruct",
    "llama_2_7b_hf": "/root/autodl-tmp/caiqiyue/code_from_paper/thesis_platform/open_model/llama_2_7b_hf",
    "distilgpt2": "/root/autodl-tmp/caiqiyue/code_from_paper/thesis_platform/open_model/distilgpt2",
    "roberta_large": "/root/autodl-tmp/caiqiyue/code_from_paper/thesis_platform/open_model/roberta_large",
    "all_minilm_l6_v2": "/root/autodl-tmp/caiqiyue/code_from_paper/thesis_platform/open_model/all_minilm_l6_v2",
}

CAIYUE_ENV = "/root/miniconda3/envs/caiqiyue/bin/python"
PRETEXT_ENV = "/root/miniconda3/envs/pretext/bin/python"

PROMPT = "Hello, how are you?"


def make_test_script(env_name, models_to_test):
    """Generate test script for specific environment and models."""
    tests = []
    for model_name, model_path in models_to_test.items():
        escaped_path = model_path.replace('"', '\\"')
        if model_name == "all_minilm_l6_v2":
            test = f'''
print("\\n--- {model_name} ---")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("{escaped_path}")
    emb = model.encode(["{PROMPT}"])
    print(f"OK: embedding shape={{emb.shape}}")
except Exception as e:
    print(f"FAILED: {{e}}")
'''
        elif model_name == "roberta_large":
            test = f'''
print("\\n--- {model_name} ---")
try:
    from transformers import RobertaForCausalLM, AutoTokenizer
    import torch
    tok = AutoTokenizer.from_pretrained("{escaped_path}", local_files_only=True)
    model = RobertaForCausalLM.from_pretrained("{escaped_path}", local_files_only=True, device_map="auto", torch_dtype=torch.float16)
    inputs = tok("{PROMPT}", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs)
    print(f"OK: logits shape={{out.logits.shape}}")
except Exception as e:
    print(f"FAILED: {{e}}")
'''
        else:
            test = f'''
print("\\n--- {model_name} ---")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    tok = AutoTokenizer.from_pretrained("{escaped_path}", local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("{escaped_path}", local_files_only=True, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
    inputs = tok("{PROMPT}", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    response = tok.decode(out[0], skip_special_tokens=True)
    print(f"OK: {{response[:60]}}")
except Exception as e:
    print(f"FAILED: {{e}}")
'''
        tests.append(test)

    return "import sys\n" + "".join(tests)


def run_test(client, python_bin, env_name, models_to_test):
    script = make_test_script(env_name, models_to_test)
    chan = client.exec_command(f'cat > /tmp/test_models_{env_name}.py', timeout=10)
    chan[0].write(script)
    chan[0].close()
    time.sleep(1)

    print(f"\n{'='*70}")
    print(f"Testing: {env_name} environment")
    print(f"{'='*70}")

    _, stdout, stderr = client.exec_command(
        f'cd /root/autodl-tmp && {python_bin} /tmp/test_models_{env_name}.py 2>&1',
        timeout=300
    )
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    print(out)
    if err:
        # Only show error if no output
        if not out.strip():
            print('STDERR:', err[:500])
        else:
            # Show warnings
            lines = err.split('\n')
            warnings = [l for l in lines if 'Warning' in l or 'warning' in l]
            if warnings:
                print('Warnings:', warnings[:3])


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USERNAME, password=PASSWORD, timeout=30)

    print("=" * 70)
    print("Model availability test: caiqiyue + pretext environments")
    print("=" * 70)

    # caiqiyue env tests all models
    run_test(client, CAIYUE_ENV, "caiqiyue", MODEL_PATHS)

    # pretext env tests roberta_large and all_minilm_l6_v2
    pretext_models = {
        "roberta_large": MODEL_PATHS["roberta_large"],
        "all_minilm_l6_v2": MODEL_PATHS["all_minilm_l6_v2"],
    }
    run_test(client, PRETEXT_ENV, "pretext", pretext_models)

    print("\n" + "=" * 70)
    print("Test complete")
    print("=" * 70)

    client.close()


if __name__ == "__main__":
    main()