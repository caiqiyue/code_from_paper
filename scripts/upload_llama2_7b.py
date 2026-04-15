#!/usr/bin/env python3
"""Upload Meta-Llama-2-7b-chat-hf safetensors files to Linux server."""

import os
import paramiko

# 配置
HOST = "connect.nmb1.seetacloud.com"
PORT = 26732
USERNAME = "root"
PASSWORD = "banpFV2mtSjO"

LOCAL_FILES = [
    r"D:\学习记录\导师项目\研究\caiqiyue_file\thesis_platform\open_model\Meta-Llama-2-7b-chat-hf\model-00001-of-00002.safetensors",
    r"D:\学习记录\导师项目\研究\caiqiyue_file\thesis_platform\open_model\Meta-Llama-2-7b-chat-hf\model-00002-of-00002.safetensors",
]
REMOTE_DIR = "/root/autodl-tmp/caiqiyue/code_from_paper/thesis_platform/open_model/Meta-Llama-2-7b-chat-hf"


def main():
    # 检查本地文件
    for f in LOCAL_FILES:
        if not os.path.exists(f):
            print(f"ERROR: 本地文件不存在: {f}")
            return
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  本地: {os.path.basename(f)} ({size_mb:.1f} MB)")

    # 连接服务器
    print(f"\n连接 {HOST}:{PORT} ...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USERNAME, password=PASSWORD, timeout=30)
    sftp = client.open_sftp()

    # 确保远程目录存在
    print(f"创建远程目录: {REMOTE_DIR}")
    try:
        sftp.stat(REMOTE_DIR)
    except FileNotFoundError:
        sftp.mkdir(REMOTE_DIR)

    # 上传文件
    for local_path in LOCAL_FILES:
        filename = os.path.basename(local_path)
        remote_path = os.path.join(REMOTE_DIR, filename)
        file_size = os.path.getsize(local_path)

        print(f"\n上传 {filename} ({file_size / (1024*1024):.1f} MB) ...")

        # 使用回调显示进度
        def progress_callback(transferred, total):
            pct = transferred * 100 // total
            print(f"\r  进度: {pct}% ({transferred // (1024*1024)}/{total // (1024*1024)} MB)", end="", flush=True)

        sftp.put(local_path, remote_path, callback=progress_callback)
        print()  # 换行
        print(f"  完成: {remote_path}")

    sftp.close()
    client.close()
    print("\n所有文件上传完成!")

    # 验证
    print("\n验证远程文件...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USERNAME, password=PASSWORD, timeout=30)
    sftp = client.open_sftp()

    for local_path in LOCAL_FILES:
        filename = os.path.basename(local_path)
        remote_path = os.path.join(REMOTE_DIR, filename)
        stat = sftp.stat(remote_path)
        print(f"  {filename}: {stat.st_size / (1024*1024):.1f} MB")

    sftp.close()
    client.close()
    print("\n验证通过!")


if __name__ == "__main__":
    main()
