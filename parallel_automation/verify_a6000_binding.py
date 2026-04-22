from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parallel_automation.old_experiment_queue import (
    CUDA_DEVICE_ORDER,
    REMOTE_CONDA,
    VISIBLE_DEVICE_INDEX,
    run_remote,
)


EXPECTED_GPU_NAME = "NVIDIA RTX A6000"
EXPECTED_VISIBLE_DEVICE = "1"


def _fail(message: str, code: int = 1) -> None:
    print(f"ERROR: {message}")
    raise SystemExit(code)


def _check_local_binding() -> None:
    if VISIBLE_DEVICE_INDEX != EXPECTED_VISIBLE_DEVICE:
        _fail(
            f"old_experiment_queue.py has VISIBLE_DEVICE_INDEX={VISIBLE_DEVICE_INDEX!r}, "
            f"expected {EXPECTED_VISIBLE_DEVICE!r}"
        )
    print(
        f"local_ok: old_experiment_queue.py uses CUDA_VISIBLE_DEVICES={VISIBLE_DEVICE_INDEX}"
    )


def _check_remote_binding(expected_env: str) -> None:
    remote_cmd = (
        f"source {REMOTE_CONDA} && "
        f"conda activate {expected_env} && "
        f"CUDA_DEVICE_ORDER={CUDA_DEVICE_ORDER} "
        f"CUDA_VISIBLE_DEVICES={EXPECTED_VISIBLE_DEVICE} "
        "python -c \"import sys,torch; "
        "name=torch.cuda.get_device_name(0); "
        "count=torch.cuda.device_count(); "
        "print('device_count=%d' % count); "
        "print('device_name=%s' % name); "
        "print('total_memory=%s' % torch.cuda.get_device_properties(0).total_memory); "
        f"sys.exit(0 if count==1 and {EXPECTED_GPU_NAME!r} in name else 4)\""
    )
    code, out, err, host = run_remote(remote_cmd, timeout=90)
    print(f"host={host}")
    if out:
        print(out.rstrip())
    if err:
        print(err.rstrip(), file=sys.stderr)
    if code != 0:
        _fail(
            f"remote probe failed for env={expected_env!r} with exit code {code}",
            code=code,
        )
    print("remote_ok: cuda:0 resolves to NVIDIA RTX A6000")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that the old automation queue binds experiments to the A6000."
    )
    parser.add_argument(
        "--env",
        default="caiqiyue-vllm",
        help="Remote conda environment used for the probe (default: caiqiyue-vllm).",
    )
    args = parser.parse_args()

    _check_local_binding()
    _check_remote_binding(args.env)
    print("OK: queue experiments will run on the A6000 only.")


if __name__ == "__main__":
    main()
