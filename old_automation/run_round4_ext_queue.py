#!/usr/bin/env python3
"""Sequential runner for Round 4 extension experiments (12 total: g5/g6/g7 × 4 datasets) on A6000 GPU."""
import datetime
import os
import subprocess

REPO = "/mnt/public/caiqiyue_file/code_from_paper"
PAPER_NEW = REPO + "/paper-new"
AUTOMATION = REPO + "/old_automation"
LOG_PATH = AUTOMATION + "/run_round4_ext_queue.log"

EXPERIMENTS = [
    ("g5", "jobs"), ("g5", "congressional"), ("g5", "forums"), ("g5", "microblog"),
    ("g6", "jobs"), ("g6", "congressional"), ("g6", "forums"), ("g6", "microblog"),
    ("g7", "jobs"), ("g7", "congressional"), ("g7", "forums"), ("g7", "microblog"),
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
    exp_id = f"ns_tune4_{group}_{dataset}"
    config = f"configs/experiments/single_node_tuning_round4_ext/{exp_id}.yaml"
    remote_log = f"{AUTOMATION}/NS-T4-EXT-{group.upper()}-{dataset.upper()}.remote.log"
    log(f"Starting {exp_id}")
    log(f"Config: {config}")
    with open(remote_log, "w") as out:
        result = subprocess.run(
            [PYTHON, "-m", "paper_new_selector.run_selector_single_node",
             "--config", config],
            cwd=PAPER_NEW,
            env=ENV,
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
    total = len(EXPERIMENTS)
    done = 0
    failed = 0
    log(f"=== Round 4 EXT Queue Start: {total} experiments on A6000 (CUDA_VISIBLE_DEVICES=1) ===")
    exp_ids = [f"ns_tune4_{g}_{d}" for g, d in EXPERIMENTS]
    log(f"Queue: {exp_ids}")

    for i, (group, dataset) in enumerate(EXPERIMENTS, 1):
        log(f"--- [{i}/{total}] ---")
        ok = run_experiment(group, dataset)
        if ok:
            done += 1
        else:
            failed += 1

    log(f"=== Round 4 EXT Done: {done} success, {failed} failed out of {total} ===")


if __name__ == "__main__":
    main()
