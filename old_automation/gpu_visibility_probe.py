from __future__ import annotations

import sys
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from old_automation.old_experiment_queue import run_remote


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visible-device", default="0")
    args = parser.parse_args()
    remote_cmd = (
        "bash -lc 'source /home/k8smaster/anaconda3/etc/profile.d/conda.sh && "
        "conda activate caiqiyue && "
        f"CUDA_VISIBLE_DEVICES={args.visible_device} python -c \"import torch; "
        "print(torch.cuda.is_available()); "
        "print(torch.cuda.device_count()); "
        "print(torch.cuda.get_device_name(0)); "
        "props = torch.cuda.get_device_properties(0); "
        "print(props.total_memory)\"'"
    )
    code, out, err, host = run_remote(remote_cmd, timeout=60)
    print(f"host={host}")
    print(f"code={code}")
    if out:
        print("stdout:")
        print(out)
    if err:
        print("stderr:")
        print(err)


if __name__ == "__main__":
    main()
