#!/usr/bin/env python3
"""Run PT-2 experiment on server via SSH."""

import paramiko
import time

HOST = "connect.nmb1.seetacloud.com"
PORT = 26732
USERNAME = "root"
PASSWORD = "banpFV2mtSjO"

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USERNAME, password=PASSWORD, timeout=30)

    print("=" * 70)
    print("Running PT-2 pipeline (stage1 + bootstrap + eval_large)")
    print("=" * 70)

    cmd = (
        "cd /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text && "
        "conda activate pretext && "
        "python -m pretext_platform.scripts.run_pipeline "
        "--config configs/experiments/validate_jobs_tiny_thesis_platform.yaml 2>&1"
    )

    stdin, stdout, stderr = client.exec_command(cmd, timeout=600)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')

    print(out)
    if err:
        print('STDERR:', err[:1000])

    client.close()
    print("\nDone.")


if __name__ == "__main__":
    main()