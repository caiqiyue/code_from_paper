#!/usr/bin/env python3
"""Upload fixed llama2_eval.py and run PT-2 test."""
import subprocess
import os
import tempfile

def run_ssh(cmd, timeout=30):
    """Run SSH command with password auth."""
    full_cmd = [
        'ssh',
        '-o', 'PreferredAuthentications=password',
        '-o', 'PubkeyAuthentication=no',
        '-o', 'NumberOfPasswordPrompts=1',
        '-o', 'ConnectTimeout=30',
        '-o', 'StrictHostKeyChecking=no',
        '-p', '26732',
        'root@connect.nmb1.seetacloud.com',
        cmd
    ]
    result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
    return result.stdout + result.stderr

def encode_and_send_file(local_path, remote_path):
    """Send file via SSH using base64 encoding."""
    with open(local_path, 'rb') as f:
        content = f.read()
    b64 = subprocess.run(['base64'], input=content, capture_output=True).stdout.decode()

    # Write to temp file
    tmp_path = os.path.join(tempfile.gettempdir(), 'llama2_eval_b64.txt')
    with open(tmp_path, 'w') as f:
        f.write(b64)

    # Read and send via SSH
    with open(tmp_path, 'r') as f:
        b64_content = f.read()

    # Use Python on remote to decode
    cmd = f'echo "{b64_content.strip()}" | base64 -d > {remote_path}'
    run_ssh(cmd, timeout=60)

    os.remove(tmp_path)

# Step 1: Upload the file
local_file = r'D:\学习记录\导师项目\研究\caiqiyue_file\PrE-Text\pretext_platform\evaluation\llama2_eval.py'
remote_file = '/root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/pretext_platform/evaluation/llama2_eval.py'

print("=== Upload llama2_eval.py ===")
encode_and_send_file(local_file, remote_file)

# Verify
result = run_ssh(f'wc -l {remote_file}')
print(f"Lines in uploaded file: {result.strip()}")

# Step 2: Check the device_map code is there
result = run_ssh(f'grep -n "device_map.*accelerator.device" {remote_file}')
print(f"device_map lines: {result.strip()}")

# Step 3: Modify config to set load_in_8bit=False
print("\n=== Setting load_in_8bit=False ===")
run_ssh('sed -i "s/load_in_8bit: true/load_in_8bit: false/" /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/configs/experiments/validate_jobs_tiny_thesis_platform.yaml')
result = run_ssh('grep load_in_8bit /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/configs/experiments/validate_jobs_tiny_thesis_platform.yaml')
print(f"Config now: {result.strip()}")

# Step 4: Run PT-2
print("\n=== Running PT-2 with load_in_8bit=False ===")
run_cmd = 'source /root/miniconda3/etc/profile.d/conda.sh && cd /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text && conda activate pretext && python -m pretext_platform.scripts.run_pipeline --config configs/experiments/validate_jobs_tiny_thesis_platform.yaml 2>&1'
result = run_ssh(run_cmd, timeout=600)
print(result[-5000:] if len(result) > 5000 else result)  # Last 5000 chars

# Check result
if 'metrics_summary.json' in result:
    print("\n=== PT-2 (no 8-bit): SUCCESS ===")
elif any(err in result for err in ['Error', 'Traceback', 'CUDA', 'OOM', 'OutOfMemory', 'Exception']):
    print("\n=== PT-2 (no 8-bit): FAILED ===")
else:
    print("\n=== PT-2 (no 8-bit): Check output manually ===")

# Restore config
print("\n=== Restoring config ===")
run_ssh('sed -i "s/load_in_8bit: false/load_in_8bit: true/" /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/configs/experiments/validate_jobs_tiny_thesis_platform.yaml')