#!/usr/bin/env python3
"""Sequential runner for Round 5fin experiments (12 total: f1-f3 × 4 datasets) on A6000 GPU.

Direction D1: g1 gate parameters (gate_low=0.78, gate_high=0.90) +
length-adaptive penalty alpha grid [0.05, 0.10, 0.15]."""

import argparse
import datetime
import os
import subprocess
from pathlib import Path

REPO = "/mnt/public/caiqiyue_file/code_from_paper"
PAPER_NEW_ROUND5 = Path(REPO + "/paper-new-round5")
AUTOMATION = Path(REPO + "/old_automation")
LOG_PATH = AUTOMATION / "run_round5fin_queue.log"

ALL_EXPERIMENTS = [
    ("f1", "jobs"),
    ("f1", "congressional"),
    ("f1", "forums"),
    ("f1", "microblog"),
    ("f2", "jobs"),
    ("f2", "congressional"),
    ("f2", "forums"),
    ("f2", "microblog"),
    ("f3", "jobs"),
    ("f3", "congressional"),
    ("f3", "forums"),
    ("f3", "microblog"),
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


def run_experiment(group, dataset):
    exp_id = f"ns_tune5fin_{group}_{dataset}"
    config_rel = (
        Path("configs/experiments/single_node_tuning_round5fin") / f"{exp_id}.yaml"
    )
    config_abs = (PAPER_NEW_ROUND5 / config_rel).resolve()
    remote_log = AUTOMATION / f"NS-T5FIN-{group.upper()}-{dataset.upper()}.remote.log"
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
        help="Which experiments: 'all' or specific group like 'f1'",
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
        f"=== Round 5fin Queue Start: {total} experiments on A6000 (CUDA_VISIBLE_DEVICES=1) ==="
    )
    exp_ids = [f"ns_tune5fin_{g}_{d}" for g, d in experiments]
    log(f"Queue: {exp_ids}")

    for i, (group, dataset) in enumerate(experiments, 1):
        log(f"--- [{i}/{total}] ---")
        ok = run_experiment(group, dataset)
        if ok:
            done += 1
        else:
            failed += 1

    log(f"=== Round 5fin Done: {done} success, {failed} failed out of {total} ===")


if __name__ == "__main__":
    main()
