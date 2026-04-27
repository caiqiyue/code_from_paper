#!/usr/bin/env python3
"""Sequential runner for Round 5 length-adaptive experiments (16 total: r1/r2/r3/r4 × 4 datasets) on A6000 GPU."""
import datetime
import os
import subprocess

REPO = "/mnt/public/caiqiyue_file/code_from_paper"
PAPER_NEW_ROUND5 = REPO + "/paper-new-round5"
AUTOMATION = REPO + "/old_automation"
LOG_PATH = AUTOMATION + "/run_round5_queue.log"

EXPERIMENTS = [
    ("r1", "jobs"), ("r1", "congressional"), ("r1", "forums"), ("r1", "microblog"),
    ("r2", "jobs"), ("r2", "congressional"), ("r2", "forums"), ("r2", "microblog"),
    ("r3", "jobs"), ("r3", "congressional"), ("r3", "forums"), ("r3", "microblog"),
    ("r4", "jobs"), ("r4", "congressional"), ("r4", "forums"), ("r4", "microblog"),
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
    exp_id = f"ns_tune5_{group}_{dataset}"
    config = f"configs/experiments/single_node_tuning_round5/{exp_id}.yaml"
    remote_log = f"{AUTOMATION}/NS-T5-{group.upper()}-{dataset.upper()}.remote.log"
    log(f"Starting {exp_id}")
    log(f"Config: {config}")
    with open(remote_log, "w") as out:
        result = subprocess.run(
            [PYTHON, "-m", "paper_new_selector.run_selector_single_node",
             "--config", config],
            cwd=PAPER_NEW_ROUND5,
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
    log(f"=== Round 5 Queue Start: {total} experiments on A6000 (CUDA_VISIBLE_DEVICES=1) ===")
    exp_ids = [f"ns_tune5_{g}_{d}" for g, d in EXPERIMENTS]
    log(f"Queue: {exp_ids}")

    for i, (group, dataset) in enumerate(EXPERIMENTS, 1):
        log(f"--- [{i}/{total}] ---")
        ok = run_experiment(group, dataset)
        if ok:
            done += 1
        else:
            failed += 1

    log(f"=== Round 5 Done: {done} success, {failed} failed out of {total} ===")


if __name__ == "__main__":
    main()
