#!/usr/bin/env python3
"""Sync modified files to server."""

import paramiko
import os

HOST = "connect.nmb1.seetacloud.com"
PORT = 26732
USERNAME = "root"
PASSWORD = "banpFV2mtSjO"

LOCAL_FILES = [
    r"D:\学习记录\导师项目\研究\caiqiyue_file\PrE-Text\requirements.txt",
    r"D:\学习记录\导师项目\研究\caiqiyue_file\install_pretext.sh",
    r"D:\学习记录\导师项目\研究\caiqiyue_file\整体实验设计.md",
]

REMOTE_BASE = "/root/autodl-tmp/caiqiyue/code_from_paper"

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USERNAME, password=PASSWORD, timeout=30)
    sftp = client.open_sftp()

    for local_path in LOCAL_FILES:
        filename = os.path.basename(local_path)
        remote_path = f"{REMOTE_BASE}/{filename}"
        print(f"上传 {filename} -> {remote_path}")

        # 先读取本地文件内容
        with open(local_path, 'rb') as f:
            content = f.read()

        # 写入远程文件
        with sftp.file(remote_path, 'wb') as f:
            f.write(content)

        # 验证
        remote_size = sftp.stat(remote_path).st_size
        print(f"  上传成功: {remote_size} bytes")

    sftp.close()
    client.close()
    print("\n文件同步完成!")

if __name__ == "__main__":
    main()
