#!/usr/bin/env python3
"""Check install_pretext.log progress."""

import paramiko

HOST = "connect.nmb1.seetacloud.com"
PORT = 26732
USERNAME = "root"
PASSWORD = "banpFV2mtSjO"

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USERNAME, password=PASSWORD, timeout=30)
    sftp = client.open_sftp()

    log_path = "/root/autodl-tmp/caiqiyue/code_from_paper/install_pretext.log"

    # Check if log exists and get last 50 lines
    try:
        with sftp.file(log_path, 'r') as f:
            f.seek(0, 2)  # seek to end
            size = f.tell()
            if size > 10000:
                f.seek(size - 10000)
            lines = f.readlines()
            print(f"=== install_pretext.log (最新 {len(lines)} 行) ===")
            for line in lines[-50:]:
                print(line.rstrip())
    except FileNotFoundError:
        print("日志文件尚未创建，安装可能刚启动...")

    # Check if process is still running
    stdin, stdout, stderr = client.exec_command("ps aux | grep install_pretext | grep -v grep")
    output = stdout.read().decode()
    if output.strip():
        print(f"\n=== 安装进程仍在运行 ===\n{output}")
    else:
        print("\n=== 安装进程已结束 ===")

    sftp.close()
    client.close()

if __name__ == "__main__":
    main()
