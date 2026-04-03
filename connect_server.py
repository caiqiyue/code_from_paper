import paramiko
import sys

host = 'connect.nmb1.seetacloud.com'
port = 26732
username = 'root'
password = 'banpFV2mtSjO'
remote_path = '/root/autodl-tmp/caiqiyue'

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    client.connect(host, port=port, username=username, password=password, timeout=10)
    print("Connected successfully!")

    # 执行命令查看目录
    stdin, stdout, stderr = client.exec_command(f'cd {remote_path} && pwd && ls -la')
    print(stdout.read().decode())
    if stderr.read().decode():
        print("Errors:", stderr.read().decode())

    # 查看子目录
    stdin, stdout, stderr = client.exec_command(f'find {remote_path} -maxdepth 2 -type d | sort')
    print("\nDirectory structure:")
    print(stdout.read().decode())

    client.close()
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)