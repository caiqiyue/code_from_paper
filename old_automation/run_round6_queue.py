#!/usr/bin/env python3
"""Sequential runner for Round 6 experiments (40 total: c01-c10 × 4 datasets) on A6000 GPU.

Direction: multi-dimensional parameter tuning for forums breakthrough.
- c01-c08: lambda_generic sweep [0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.01]
- c09: lambda_redundancy=0.20 (reduced redundancy)
- c10: lambda_redundancy=0.15 (further reduced redundancy)
All configs run on all 4 datasets (jobs, forums, microblog, congressional)."""

import argparse
import datetime
import os
import subprocess
from pathlib import Path

REPO = "/mnt/public/caiqiyue_file/code_from_paper"
PAPER_NEW_ROUND5 = Path(REPO + "/paper-new-round5")
AUTOMATION = Path(REPO + "/old_automation")
LOG_PATH = AUTOMATION / "run_round6_queue.log"

ALL_EXPERIMENTS = [
    # c01 - baseline
    ("c01", "jobs"),
    ("c01", "forums"),
    ("c01", "microblog"),
    ("c01", "congressional"),
    # c02
    ("c02", "jobs"),
    ("c02", "forums"),
    ("c02", "microblog"),
    ("c02", "congressional"),
    # c03
    ("c03", "jobs"),
    ("c03", "forums"),
    ("c03", "microblog"),
    ("c03", "congressional"),
    # c04
    ("c04", "jobs"),
    ("c04", "forums"),
    ("c04", "microblog"),
    ("c04", "congressional"),
    # c05
    ("c05", "jobs"),
    ("c05", "forums"),
    ("c05", "microblog"),
    ("c05", "congressional"),
    # c06
    ("c06", "jobs"),
    ("c06", "forums"),
    ("c06", "microblog"),
    ("c06", "congressional"),
    # c07
    ("c07", "jobs"),
    ("c07", "forums"),
    ("c07", "microblog"),
    ("c07", "congressional"),
    # c08
    ("c08", "jobs"),
    ("c08", "forums"),
    ("c08", "microblog"),
    ("c08", "congressional"),
    # c09
    ("c09", "jobs"),
    ("c09", "forums"),
    ("c09", "microblog"),
    ("c09", "congressional"),
    # c10
    ("c10", "jobs"),
    ("c10", "forums"),
    ("c10", "microblog"),
    ("c10", "congressional"),
]

ENV = {
    **os.environ,
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": "1",
    "PYTHONUNBUFFERED": "1",
    "VLLM_HOST_IP": "127.0.0.1",
    "HOST_IP": "127.0.0.1",
}

PYTHON = "/home/k8smaster/anaconda3/envs/pretext/bin/python"


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def run_experiment(cfg_id, dataset):
    exp_id = f"ns_tune6_{cfg_id}_{dataset}"
    config_rel = (
        Path("configs/experiments/single_node_tuning_round6") / f"{exp_id}.yaml"
    )
    config_abs = (PAPER_NEW_ROUND5 / config_rel).resolve()
    remote_log = AUTOMATION / f"NS-TUNE6-{cfg_id.upper()}-{dataset.upper()}.remote.log"
    log(f"Starting {exp_id}")
    log(f"Config: {config_abs}")
    env = {**ENV, "PYTHONPATH": str(PAPER_NEW_ROUND5)}
    with open(remote_log, "w") as out:
        result = subprocess.run(
            [
                PYTHON,
                "-m",
                "paper_new_selector.run_selector_single_node",
                "--config",
                str(config_abs),
            ],
            cwd=str(PAPER_NEW_ROUND5),
            env=env,
            stdout=out,
            stderr=out,
        )
    if result.returncode == 0:
        log(f"SUCCESS: {exp_id}")
        return True
    else:
        log(f"FAILED (exit {result.returncode}): {exp_id} -- see {remote_log}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments",
        default="all",
        help="Which experiments: 'all' or specific config like 'c01'",
    )
    args = parser.parse_args()

    if args.experiments == "all":
        experiments = ALL_EXPERIMENTS
    else:
        experiments = [e for e in ALL_EXPERIMENTS if e[0] == args.experiments]

    total = len(experiments)
    done = 0
    failed = 0
    log(
        f"=== Round 6 Queue Start: {total} experiments on A6000 (CUDA_VISIBLE_DEVICES=1) ==="
    )
    exp_ids = [f"ns_tune6_{c}_{d}" for c, d in experiments]
    log(f"Queue: {exp_ids}")

    for i, (cfg_id, dataset) in enumerate(experiments, 1):
        log(f"--- [{i}/{total}] ---")
        ok = run_experiment(cfg_id, dataset)
        if ok:
            done += 1
        else:
            failed += 1

    log(f"=== Round 6 Done: {done} success, {failed} failed out of {total} ===")


if __name__ == "__main__":
    main()
