from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "old_automation" / "old_experiment_queue_state.json"
LOG_PATH = ROOT / "old_automation" / "old_experiment_queue.log"

PLINK = ROOT / "execute" / "plink.exe"
HOST_CANDIDATES = ["58.217.205.244", "1u72c85740.zicp.fun"]
HOSTKEY = "ssh-ed25519 255 SHA256:K8SqbQTOKQ0/9mWxXIflld9+WhK3P1CHzQABxiFkfGo"
PORT = 54360
USERNAME = "k8smaster"
PASSWORD = "k8s"

REMOTE_REPO = "/mnt/public/caiqiyue_file/code_from_paper"
REMOTE_THESIS_OUTPUT = f"{REMOTE_REPO}/outputs/thesis_platform"
REMOTE_PAPER_NEW_REPO = f"{REMOTE_REPO}/paper-new"
REMOTE_PAPER_NEW_OUTPUT = f"{REMOTE_PAPER_NEW_REPO}/outputs"
REMOTE_PRETEXT_REPO = f"{REMOTE_REPO}/PrE-Text"
REMOTE_PRETEXT_OUTPUT = f"{REMOTE_PRETEXT_REPO}/outputs/pretext_platform"
REMOTE_CONDA = "/home/k8smaster/anaconda3/etc/profile.d/conda.sh"
REMOTE_AUTOMATION = f"{REMOTE_REPO}/old_automation"
CUDA_DEVICE_ORDER = "PCI_BUS_ID"
VISIBLE_DEVICE_INDEX = "1"


@dataclass(frozen=True)
class ExperimentDef:
    label: str
    actual_experiment_id: str
    config_path: str
    env_name: str
    family: str  # "paper_new" or "pretext"

    @property
    def output_root(self) -> str:
        if self.family == "paper_new":
            return REMOTE_PAPER_NEW_OUTPUT
        return REMOTE_PRETEXT_OUTPUT

    @property
    def remote_workdir(self) -> str:
        return REMOTE_PAPER_NEW_REPO if self.family == "paper_new" else REMOTE_PRETEXT_REPO

    @property
    def remote_config_file(self) -> str:
        return f"{REMOTE_REPO}/{self.config_path}"

    @property
    def remote_config_arg(self) -> str:
        if self.family == "paper_new" and self.config_path.startswith("paper-new/"):
            return self.config_path.removeprefix("paper-new/")
        if self.family == "pretext" and self.config_path.startswith("PrE-Text/"):
            return self.config_path.removeprefix("PrE-Text/")
        return self.config_path


QUEUE: list[ExperimentDef] = [
    ExperimentDef(
        label="NS-C1",
        actual_experiment_id="ns_c1_jobs_base",
        config_path="paper-new/configs/experiments/single_node_formal/ns_c1_jobs_base.yaml",
        env_name="pretext",
        family="paper_new",
    ),
    ExperimentDef(
        label="NS-C2",
        actual_experiment_id="ns_c2_congressional_base",
        config_path="paper-new/configs/experiments/single_node_formal/ns_c2_congressional_base.yaml",
        env_name="pretext",
        family="paper_new",
    ),
    ExperimentDef(
        label="NS-C3",
        actual_experiment_id="ns_c3_forums_base",
        config_path="paper-new/configs/experiments/single_node_formal/ns_c3_forums_base.yaml",
        env_name="pretext",
        family="paper_new",
    ),
    ExperimentDef(
        label="NS-C4",
        actual_experiment_id="ns_c4_microblog_base",
        config_path="paper-new/configs/experiments/single_node_formal/ns_c4_microblog_base.yaml",
        env_name="pretext",
        family="paper_new",
    ),
    ExperimentDef(
        label="SP-C1",
        actual_experiment_id="sp_c1_jobs_base",
        config_path="PrE-Text/configs/experiments/single_node_formal/sp_c1_jobs_base.yaml",
        env_name="pretext",
        family="pretext",
    ),
    ExperimentDef(
        label="SP-C2",
        actual_experiment_id="sp_c2_congressional_base",
        config_path="PrE-Text/configs/experiments/single_node_formal/sp_c2_congressional_base.yaml",
        env_name="pretext",
        family="pretext",
    ),
    ExperimentDef(
        label="SP-C3",
        actual_experiment_id="sp_c3_forums_base",
        config_path="PrE-Text/configs/experiments/single_node_formal/sp_c3_forums_base.yaml",
        env_name="pretext",
        family="pretext",
    ),
    ExperimentDef(
        label="SP-C4",
        actual_experiment_id="sp_c4_microblog_base",
        config_path="PrE-Text/configs/experiments/single_node_formal/sp_c4_microblog_base.yaml",
        env_name="pretext",
        family="pretext",
    ),
    ExperimentDef(
        label="NS-C5",
        actual_experiment_id="ns_c5_jobs_eps05",
        config_path="paper-new/configs/experiments/single_node_formal/ns_c5_jobs_eps05.yaml",
        env_name="pretext",
        family="paper_new",
    ),
    ExperimentDef(
        label="SP-C5",
        actual_experiment_id="sp_c5_jobs_eps05",
        config_path="PrE-Text/configs/experiments/single_node_formal/sp_c5_jobs_eps05.yaml",
        env_name="pretext",
        family="pretext",
    ),
    ExperimentDef(
        label="NS-C6",
        actual_experiment_id="ns_c6_jobs_eps758",
        config_path="paper-new/configs/experiments/single_node_formal/ns_c6_jobs_eps758.yaml",
        env_name="pretext",
        family="paper_new",
    ),
    ExperimentDef(
        label="SP-C6",
        actual_experiment_id="sp_c6_jobs_eps758",
        config_path="PrE-Text/configs/experiments/single_node_formal/sp_c6_jobs_eps758.yaml",
        env_name="pretext",
        family="pretext",
    ),
    ExperimentDef(
        label="NS-C7",
        actual_experiment_id="ns_c7_jobs_no_privacy",
        config_path="paper-new/configs/experiments/single_node_formal/ns_c7_jobs_no_privacy.yaml",
        env_name="pretext",
        family="paper_new",
    ),
    ExperimentDef(
        label="SP-C7",
        actual_experiment_id="sp_c7_jobs_no_privacy",
        config_path="PrE-Text/configs/experiments/single_node_formal/sp_c7_jobs_no_privacy.yaml",
        env_name="pretext",
        family="pretext",
    ),
    ExperimentDef(
        label="NS-C8",
        actual_experiment_id="ns_c8_jobs_seed123",
        config_path="paper-new/configs/experiments/single_node_formal/ns_c8_jobs_seed123.yaml",
        env_name="pretext",
        family="paper_new",
    ),
    ExperimentDef(
        label="SP-C8",
        actual_experiment_id="sp_c8_jobs_seed123",
        config_path="PrE-Text/configs/experiments/single_node_formal/sp_c8_jobs_seed123.yaml",
        env_name="pretext",
        family="pretext",
    ),
    ExperimentDef(
        label="NS-C9",
        actual_experiment_id="ns_c9_jobs_seed456",
        config_path="paper-new/configs/experiments/single_node_formal/ns_c9_jobs_seed456.yaml",
        env_name="pretext",
        family="paper_new",
    ),
    ExperimentDef(
        label="SP-C9",
        actual_experiment_id="sp_c9_jobs_seed456",
        config_path="PrE-Text/configs/experiments/single_node_formal/sp_c9_jobs_seed456.yaml",
        env_name="pretext",
        family="pretext",
    ),
]


def log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8-sig"))
    else:
        state = {"queue": [], "current_label": None}

    existing = {item["label"]: item for item in state.get("queue", [])}
    queue: list[dict[str, Any]] = []
    for exp in QUEUE:
        item = existing.get(exp.label)
        if item is None:
            item = {
                "label": exp.label,
                "actual_experiment_id": exp.actual_experiment_id,
                "config_path": exp.config_path,
                "env_name": exp.env_name,
                "family": exp.family,
                "status": "pending",
                "note": "",
                "pid": None,
                "last_checked_at": None,
            }
        item["actual_experiment_id"] = exp.actual_experiment_id
        item["config_path"] = exp.config_path
        item["env_name"] = exp.env_name
        item["family"] = exp.family
        queue.append(item)

    state["queue"] = queue
    if state.get("current_label") not in {exp.label for exp in QUEUE}:
        state["current_label"] = None
    state.setdefault("current_label", None)
    return state


def save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def run_remote(remote_cmd: str, *, timeout: int = 120) -> tuple[int, str, str, str]:
    last: tuple[int, str, str, str] | None = None
    for host in HOST_CANDIDATES:
        completed = subprocess.run(
            [
                str(PLINK),
                "-batch",
                "-hostkey",
                HOSTKEY,
                "-ssh",
                "-P",
                str(PORT),
                "-l",
                USERNAME,
                "-pw",
                PASSWORD,
                host,
                remote_cmd,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        last = (completed.returncode, completed.stdout, completed.stderr, host)
        if completed.returncode == 0:
            return last
        log(f"Connection or command failed via {host}: {completed.stderr.strip() or completed.stdout.strip()}")
    assert last is not None
    return last


def queue_item(state: dict[str, Any], label: str) -> dict[str, Any]:
    for item in state["queue"]:
        if item["label"] == label:
            return item
    raise KeyError(label)


def set_queue_item(
    state: dict[str, Any],
    label: str,
    *,
    status: str,
    note: str,
    pid: str | None = None,
) -> None:
    item = queue_item(state, label)
    item["status"] = status
    item["note"] = note
    if pid is not None:
        item["pid"] = pid
    elif status != "running":
        item["pid"] = None
    item["last_checked_at"] = datetime.now().isoformat(timespec="seconds")


def experiment_def(label: str) -> ExperimentDef:
    for exp in QUEUE:
        if exp.label == label:
            return exp
    raise KeyError(label)


def any_remote_experiment_running() -> bool:
    cmd = (
        "ps -ef | grep -E "
        "'paper_new_selector.run_selector_single_node|pretext_platform.scripts.run_pipeline' "
        "| grep -v grep || true"
    )
    code, out, err, host = run_remote(cmd, timeout=40)
    if code != 0:
        raise RuntimeError(err.strip() or out.strip() or "failed to check remote process")
    if out.strip():
        log(f"Remote experiment is already running on {host}; leaving queue unchanged.")
        log(out.strip().splitlines()[0])
        return True
    return False


def _experiment_process_patterns(exp: ExperimentDef) -> list[str]:
    if exp.family == "paper_new":
        return [f"python -m paper_new_selector.run_selector_single_node --config {exp.remote_config_arg}"]
    return [f"python -m pretext_platform.scripts.run_pipeline --config {exp.remote_config_arg}"]


def _experiment_process_snapshot(exp: ExperimentDef) -> tuple[bool, str]:
    grep_cmds = [
        f"ps -ef | grep -F {shlex.quote(pattern)} | grep -v grep || true"
        for pattern in _experiment_process_patterns(exp)
    ]
    code, out, err, _ = run_remote("\n".join(grep_cmds), timeout=40)
    if code != 0:
        raise RuntimeError(err.strip() or out.strip() or f"failed to inspect process for {exp.label}")
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    return bool(lines), (lines[0] if lines else "")


def _inspect_paper_new_experiment(exp: ExperimentDef) -> tuple[str, str]:
    cmd = f"""
cd {REMOTE_PAPER_NEW_REPO} || exit 1
/home/k8smaster/anaconda3/bin/python - <<'PY'
import json
from pathlib import Path

experiment_id = {exp.actual_experiment_id!r}
experiment_dir = Path({REMOTE_PAPER_NEW_OUTPUT!r}) / experiment_id
if not experiment_dir.exists():
    print("failed:missing experiment dir")
    raise SystemExit
summary = experiment_dir / "eval" / "downstream_eval_summary.json"
stage2 = experiment_dir / "eval" / "stage2" / "llama7b_text_syn.json"
eval_dir = experiment_dir / "eval"
if summary.exists() and stage2.exists():
    print("success:downstream summary and stage2 corpus present")
elif any(path.exists() for path in (stage2, eval_dir)):
    names = [name for name, path in (("stage2", stage2), ("eval", eval_dir)) if path.exists()]
    print(f"partial:artifacts present ({{','.join(names)}}) without final downstream summary")
else:
    print("failed:no result artifacts")
PY
"""
    code, out, err, _ = run_remote(cmd, timeout=60)
    if code != 0:
        return "failed", err.strip() or out.strip() or "inspect failed"
    result = out.strip().splitlines()[-1] if out.strip() else "failed:empty inspect output"
    status, _, note = result.partition(":")
    return status, note


def _inspect_pretext_experiment(exp: ExperimentDef) -> tuple[str, str]:
    cmd = f"""
cd {REMOTE_PRETEXT_REPO} || exit 1
/home/k8smaster/anaconda3/bin/python - <<'PY'
import json
from pathlib import Path

experiment_id = {exp.actual_experiment_id!r}
experiment_dir = Path({REMOTE_PRETEXT_OUTPUT!r}) / experiment_id
metrics = experiment_dir / "metrics_summary.json"
eval_small = experiment_dir / "eval_small_summary.json"
stage1 = experiment_dir / "stage1"
stage2 = experiment_dir / "stage2"
if not experiment_dir.exists():
    print("failed:missing experiment dir")
    raise SystemExit
if eval_small.exists() and metrics.exists():
    print("success:metrics and eval_small present")
elif eval_small.exists():
    print("success:eval_small present")
elif metrics.exists():
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    status = str(payload.get("status", "")).upper()
    if status == "FAILED":
        print(f"failed:metrics_status={{status}}")
    else:
        print("partial:metrics exists but eval_small missing")
elif any(path.exists() for path in (stage1, stage2)):
    names = [name for name, path in (("stage1", stage1), ("stage2", stage2)) if path.exists()]
    print(f"partial:artifacts present ({{','.join(names)}}) without final summaries")
else:
    print("failed:no result artifacts")
PY
"""
    code, out, err, _ = run_remote(cmd, timeout=60)
    if code != 0:
        return "failed", err.strip() or out.strip() or "inspect failed"
    result = out.strip().splitlines()[-1] if out.strip() else "failed:empty inspect output"
    status, _, note = result.partition(":")
    return status, note


def inspect_experiment(exp: ExperimentDef) -> tuple[str, str]:
    process_running, process_note = _experiment_process_snapshot(exp)
    if exp.family == "paper_new":
        artifact_status, artifact_note = _inspect_paper_new_experiment(exp)
    else:
        artifact_status, artifact_note = _inspect_pretext_experiment(exp)
    if artifact_status == "success":
        return "success", artifact_note
    if artifact_status == "running":
        return "running", artifact_note
    if process_running:
        return "running", artifact_note or process_note or "remote process still active"
    return artifact_status, artifact_note


def start_experiment(exp: ExperimentDef) -> str:
    remote_log = f"{REMOTE_AUTOMATION}/{exp.label}.remote.log"
    if exp.family == "paper_new":
        run_cmd = (
            f"source {REMOTE_CONDA} && conda activate {exp.env_name} && cd {REMOTE_PAPER_NEW_REPO} "
            f"&& export PYTHONUNBUFFERED=1 && export CUDA_DEVICE_ORDER={CUDA_DEVICE_ORDER} "
            f"&& export CUDA_VISIBLE_DEVICES={VISIBLE_DEVICE_INDEX} "
            f"&& export VLLM_HOST_IP=127.0.0.1 && export HOST_IP=127.0.0.1 "
            f"&& python -m paper_new_selector.run_selector_single_node --config {exp.remote_config_arg}"
        )
    else:
        run_cmd = (
            f"source {REMOTE_CONDA} && conda activate {exp.env_name} && cd {REMOTE_PRETEXT_REPO} "
            f"&& export PYTHONUNBUFFERED=1 && export CUDA_DEVICE_ORDER={CUDA_DEVICE_ORDER} "
            f"&& export CUDA_VISIBLE_DEVICES={VISIBLE_DEVICE_INDEX} "
            f"&& python -m pretext_platform.scripts.run_pipeline --config {exp.remote_config_arg}"
        )
    cmd = f"""
set -e
mkdir -p {REMOTE_AUTOMATION}
cd {REMOTE_REPO}
test -f {exp.remote_config_file}
nohup bash -lc '{run_cmd}' > {remote_log} 2>&1 </dev/null &
echo $!
"""
    code, out, err, host = run_remote(cmd, timeout=120)
    if code != 0:
        raise RuntimeError(err.strip() or out.strip() or f"failed to start {exp.label}")
    pid = out.strip().splitlines()[-1]
    log(f"Started {exp.label} on {host} with PID {pid}; env={exp.env_name}; log={remote_log}")
    return pid


def main() -> None:
    if not PLINK.exists():
        raise FileNotFoundError(f"plink.exe not found: {PLINK}")

    state = load_state()
    log("Queue: " + ", ".join(f"{exp.label}({exp.env_name})" for exp in QUEUE))

    current_label = state.get("current_label")
    if current_label:
        exp = experiment_def(current_label)
        status, note = inspect_experiment(exp)
        if status == "success":
            set_queue_item(state, current_label, status="success", note=note)
            state["current_label"] = None
            log(f"{current_label} completed successfully: {note}")
        elif status == "running":
            set_queue_item(state, current_label, status="running", note=note)
            log(f"{current_label} is still running: {note}")
            save_state(state)
            return
        else:
            set_queue_item(state, current_label, status="failed", note=note)
            state["current_label"] = None
            log(f"{current_label} failed or missing artifacts: {note}")
        save_state(state)

    for exp in QUEUE:
        item = queue_item(state, exp.label)
        if item["status"] != "pending":
            continue
        status, note = inspect_experiment(exp)
        if status == "success":
            set_queue_item(state, exp.label, status="success", note=f"existing result: {note}")
            log(f"{exp.label} marked success from existing remote result.")
            save_state(state)
            continue
        if status == "running":
            state["current_label"] = exp.label
            set_queue_item(state, exp.label, status="running", note=note)
            log(f"{exp.label} is already running remotely: {note}")
            save_state(state)
            return
        if status == "failed" and "missing experiment dir" not in note and "no result artifacts" not in note:
            set_queue_item(state, exp.label, status="failed", note=note)
            log(f"{exp.label} is marked failed remotely: {note}")
            save_state(state)
            continue
        if any_remote_experiment_running():
            save_state(state)
            return
        pid = start_experiment(exp)
        state["current_label"] = exp.label
        set_queue_item(state, exp.label, status="running", note="started by old server scheduler", pid=pid)
        save_state(state)
        return

    log("All old-server queued experiments are resolved.")
    save_state(state)


if __name__ == "__main__":
    main()
