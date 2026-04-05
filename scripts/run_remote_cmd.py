#!/usr/bin/env python3
"""Execute commands on remote server via SSH."""

import paramiko
import time

HOST = "connect.nmb1.seetacloud.com"
PORT = 26732
USERNAME = "root"
PASSWORD = "banpFV2mtSjO"

def run_cmd(client, cmd, timeout=30):
    """Run a command and return output."""
    print(f"\n{'='*60}")
    print(f"CMD: {cmd}")
    print(f"{'='*60}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    if out:
        print(out)
    if err:
        print(f"STDERR: {err}")
    return out, err

def main():
    print(f"连接 {HOST}:{PORT} ...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USERNAME, password=PASSWORD, timeout=30)

    # Step 1: 删除旧的 pretext 环境
    print("\n\n" + "="*70)
    print("  Step 1: 删除旧的 pretext 环境")
    print("="*70)

    conda_bin = "/root/miniconda3/bin/conda"
    run_cmd(client, f"{conda_bin} env remove -n pretext -y")
    run_cmd(client, "rm -rf /root/autodl-tmp/conda-envs/pretext 2>&1 || true")
    run_cmd(client, f"{conda_bin} env list")

    # Step 2: 运行安装脚本
    print("\n\n" + "="*70)
    print("  Step 2: 运行 install_pretext.sh")
    print("="*70)

    # 先确认脚本已同步
    run_cmd(client, "ls -la /root/autodl-tmp/caiqiyue/code_from_paper/install_pretext.sh")

    # 构建安装命令 - 使用 nohup 后台运行，避免 SSH 断开导致中断
    install_cmd = (
        "cd /root/autodl-tmp/caiqiyue/code_from_paper && "
        "ENV_NAME=pretext "
        "CONDA_ENV_PREFIX=/root/miniconda3/envs/pretext "
        "CONDA_BIN=/root/miniconda3/bin/conda "
        "PRETEXT_DIR=/root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text "
        "START_CLASH=true "
        "nohup bash install_pretext.sh > /root/autodl-tmp/caiqiyue/code_from_paper/install_pretext.log 2>&1 &"
    )

    print(f"\n启动安装脚本（nohup 后台运行）...")
    stdin, stdout, stderr = client.exec_command(install_cmd)
    # 等待一小段时间确保命令已提交
    time.sleep(2)

    # 检查 nohup 是否成功提交
    check_cmd = "ps aux | grep install_pretext | grep -v grep"
    stdin2, stdout2, stderr2 = client.exec_command(check_cmd)
    print(f"检查安装进程: {stdout2.read().decode()}")

    print("\n安装已在后台启动!")
    print("查看日志: tail -f /root/autodl-tmp/caiqiyue/code_from_paper/install_pretext.log")
    print("实时查看进度: watch -n 5 'tail -20 /root/autodl-tmp/caiqiyue/code_from_paper/install_pretext.log'")

    client.close()

if __name__ == "__main__":
    main()
