from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from .thesis_bridge import load_yaml_config

REPEAT10_SEEDS = list(range(1, 11))
REPEAT10_DATASETS = ("jobs", "congressional", "forums", "microblog")
REPEAT10_SUMMARY_HEADER = [
    "experiment",
    "baseline",
    "dataset",
    "seed",
    "status",
    "source_artifact_path",
    "best_top1",
    "best_top3",
    "best_top5",
    "best_top10",
]

REPEAT10_TRANSIENT_FAILURE_MARKERS = (
    "# GPU blocks: 0",
    "No available memory for the cache blocks",
    "vllm_runtime_gpu_oom",
    "CUDA out of memory",
    "Stage 2 passed the startup memory gate but vLLM later hit CUDA out of memory.",
)
REPEAT10_DEFAULT_CUDA_VISIBLE_DEVICES = "1"
REPEAT10_CUDA_OVERRIDE_ENV = "REPEAT10_CUDA_VISIBLE_DEVICES"
REPEAT10_TARGET_GPU_NAME_TOKEN = "RTX A6000"
REPEAT10_MIN_FREE_GB_FOR_VLLM = 20.0
REPEAT10_GPU_POLL_SECONDS = 30
REPEAT10_GPU_WAIT_TIMEOUT_SECONDS = 6 * 60 * 60
REPEAT10_MAX_ATTEMPTS = 4

_BASELINE_TEMPLATE_MAP: dict[str, tuple[str, str]] = {
    "c4": ("c4", "c4"),
    "expand_only": ("eo", "expand_only"),
    "expand_private": ("ep", "expand_private"),
    "wasp": ("wasp", "wasp"),
    "dpga": ("dpga", "dpga"),
}


@dataclass(frozen=True)
class Repeat10BaselineSpec:
    baseline: str
    dataset: str
    seed: int
    experiment_id: str
    single_run_template: Path
    relative_output_root: Path
    relative_config_path: Path
    relative_source_artifact: Path | None


def resolve_repeat10_project_root(module_file: str | Path | None = None) -> Path:
    module_path = Path(module_file) if module_file is not None else Path(__file__)
    return module_path.resolve().parents[1]


def build_repeat10_run_specs() -> list[Repeat10BaselineSpec]:
    specs: list[Repeat10BaselineSpec] = []
    config_root = Path("configs/experiments/repeat10_baseline_screening")
    output_root = Path("paper-new-round19/outputs/repeat10_baseline_screening")
    template_root = Path("configs/experiments/single_run_baseline_screening")
    for seed in REPEAT10_SEEDS:
        seed_tag = f"seed{seed:02d}"
        for baseline, (prefix, output_name) in _BASELINE_TEMPLATE_MAP.items():
            for dataset in REPEAT10_DATASETS:
                experiment_id = f"{prefix}_{dataset}_repeat10_seed{seed:02d}"
                relative_source_artifact: Path | None = None
                if baseline == "wasp":
                    relative_source_artifact = (
                        Path("WASP") / "outputs" / "paper_new_screening" / "repeat10" / dataset / seed_tag / "train.jsonl"
                    )
                elif baseline == "dpga":
                    relative_source_artifact = (
                        Path("DPGA-TextSyn")
                        / "outputs"
                        / "paper_new_screening"
                        / "repeat10"
                        / dataset
                        / seed_tag
                        / "epoch_all.json"
                    )
                specs.append(
                    Repeat10BaselineSpec(
                        baseline=baseline,
                        dataset=dataset,
                        seed=seed,
                        experiment_id=experiment_id,
                        single_run_template=template_root / f"{prefix}_{dataset}_single_run.yaml",
                        relative_output_root=output_root / output_name / dataset / seed_tag,
                        relative_config_path=config_root / f"{experiment_id}.yaml",
                        relative_source_artifact=relative_source_artifact,
                    )
                )
    return specs


def write_repeat10_config(spec: Repeat10BaselineSpec, *, project_root: Path) -> Path:
    template = load_yaml_config(spec.single_run_template)
    template.setdefault("meta", {})["seed"] = spec.seed
    template["meta"]["experiment_id"] = spec.experiment_id
    template["meta"]["stage"] = "repeat10_baseline_screening"
    template.setdefault("paths", {})["output_root"] = spec.relative_output_root.as_posix()
    template.setdefault("protocol", {})["repeat_protocol"] = "round19_repeat10_baselines"
    template["protocol"]["comparison_type"] = "repeat10_four_dataset_baseline_screening"
    if spec.relative_source_artifact is not None:
        template.setdefault("external_baseline", {})["source_artifact_path"] = spec.relative_source_artifact.as_posix()
        template["external_baseline"]["summary_output_path"] = (
            spec.relative_output_root / "stage1_summary.json"
        ).as_posix()

    target_path = project_root / spec.relative_config_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        yaml.safe_dump(template, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target_path


def materialize_repeat10_configs(project_root: str | Path | None = None) -> list[Path]:
    root = Path(project_root).resolve() if project_root is not None else resolve_repeat10_project_root()
    generated: list[Path] = []
    for spec in build_repeat10_run_specs():
        generated.append(write_repeat10_config(spec, project_root=root))
    return generated


def build_repeat10_command(spec: Repeat10BaselineSpec, config_path: Path) -> list[str]:
    module = (
        "paper_new_selector.run_external_baseline_single_run"
        if spec.baseline in {"wasp", "dpga"}
        else "paper_new_selector.run_selector_single_node"
    )
    return [sys.executable, "-m", module, "--config", str(config_path)]


def classify_retryable_failure(log_text: str) -> str | None:
    if "# GPU blocks: 0" in log_text or "No available memory for the cache blocks" in log_text:
        return "retryable_vllm_cache"
    for marker in REPEAT10_TRANSIENT_FAILURE_MARKERS[2:]:
        if marker in log_text:
            return "retryable_vllm_resource"
    return None


def initialize_repeat10_summary(summary_path: Path) -> Path:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\t".join(REPEAT10_SUMMARY_HEADER) + "\n", encoding="utf-8")
    return summary_path


def append_repeat10_summary_row(summary_path: Path, spec: Repeat10BaselineSpec, status: int) -> None:
    output_dir = resolve_repeat10_runtime_output_dir(spec)
    metrics_path = output_dir / "eval" / "downstream_eval_summary.json"
    metrics: dict[str, object] = {}
    if status == 0 and metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
    row = [
        spec.experiment_id,
        spec.baseline,
        spec.dataset,
        spec.seed,
        status,
        spec.relative_source_artifact.as_posix() if spec.relative_source_artifact is not None else "NA",
        metrics.get("best_top1", "NA"),
        metrics.get("best_top3", "NA"),
        metrics.get("best_top5", "NA"),
        metrics.get("best_top10", "NA"),
    ]
    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write("\t".join(str(item) for item in row) + "\n")


def build_repeat10_child_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(env.get(REPEAT10_CUDA_OVERRIDE_ENV, REPEAT10_DEFAULT_CUDA_VISIBLE_DEVICES))
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return env


def resolve_repeat10_runtime_output_dir(spec: Repeat10BaselineSpec) -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / "repeat10_baseline_screening"
        / _BASELINE_TEMPLATE_MAP[spec.baseline][1]
        / spec.dataset
        / f"seed{spec.seed:02d}"
    )


def reset_repeat10_output_dir(output_dir: Path) -> None:
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)


def spec_requires_vllm(spec: Repeat10BaselineSpec) -> bool:
    return spec.baseline != "c4"


def parse_nvidia_smi_memory_report(report_text: str, *, target_name_token: str) -> tuple[str, float]:
    for raw_line in report_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        index, name, free_mib = parts
        if target_name_token in name:
            return index, float(free_mib) / 1024.0
    raise RuntimeError(f"Could not find GPU with token {target_name_token!r} in nvidia-smi output.")


def query_repeat10_a6000_free_gb() -> tuple[str, float]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return parse_nvidia_smi_memory_report(completed.stdout, target_name_token=REPEAT10_TARGET_GPU_NAME_TOKEN)


def wait_for_repeat10_vllm_capacity(log, *, minimum_free_gb: float = REPEAT10_MIN_FREE_GB_FOR_VLLM) -> None:
    deadline = time.time() + REPEAT10_GPU_WAIT_TIMEOUT_SECONDS
    while True:
        gpu_index, free_gb = query_repeat10_a6000_free_gb()
        if free_gb >= minimum_free_gb:
            log(
                f"GPU_READY target={REPEAT10_TARGET_GPU_NAME_TOKEN} index={gpu_index} "
                f"free_gb={free_gb:.2f} threshold={minimum_free_gb:.2f}"
            )
            return
        if time.time() >= deadline:
            raise TimeoutError(
                f"A6000 free memory stayed below {minimum_free_gb:.2f} GiB for "
                f"{REPEAT10_GPU_WAIT_TIMEOUT_SECONDS} seconds."
            )
        log(
            f"GPU_WAIT target={REPEAT10_TARGET_GPU_NAME_TOKEN} index={gpu_index} "
            f"free_gb={free_gb:.2f} threshold={minimum_free_gb:.2f}"
        )
        time.sleep(REPEAT10_GPU_POLL_SECONDS)


def resolve_repeat10_effective_status(returncode: int, output_dir: Path) -> int:
    if returncode != 0:
        return returncode
    summary_path = output_dir / "eval" / "downstream_eval_summary.json"
    if not summary_path.exists():
        return 86
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if str(payload.get("status", "")).strip().lower() != "completed":
        return 87
    return 0


def run_repeat10_batch(project_root: str | Path | None = None) -> int:
    root = Path(project_root).resolve() if project_root is not None else resolve_repeat10_project_root()
    materialize_repeat10_configs(root)

    logdir = root / "logs"
    summary_path = logdir / "repeat10_baseline_screening_summary.tsv"
    master_path = logdir / "repeat10_baseline_screening_master.log"
    child_env = build_repeat10_child_env()

    logdir.mkdir(parents=True, exist_ok=True)
    initialize_repeat10_summary(summary_path)
    master_path.write_text("", encoding="utf-8")
    had_failure = 0

    def log(message: str) -> None:
        line = f"{datetime.now().strftime('%F %T')} {message}"
        print(line, flush=True)
        with master_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    for spec in build_repeat10_run_specs():
        config_path = root / spec.relative_config_path
        log_path = logdir / f"{spec.experiment_id}.log"
        output_dir = resolve_repeat10_runtime_output_dir(spec)
        status = 1
        log(f"START {spec.experiment_id} dataset={spec.dataset} seed={spec.seed} cfg={config_path}")

        for attempt in range(1, REPEAT10_MAX_ATTEMPTS + 1):
            if spec_requires_vllm(spec):
                wait_for_repeat10_vllm_capacity(log)
            reset_repeat10_output_dir(output_dir)
            mode = "w" if attempt == 1 else "a"
            with log_path.open(mode, encoding="utf-8") as handle:
                if attempt > 1:
                    handle.write(f"\n===== retry attempt {attempt} =====\n")
                completed = subprocess.run(
                    build_repeat10_command(spec, config_path),
                    cwd=root,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                    env=child_env,
                )
            status = resolve_repeat10_effective_status(completed.returncode, output_dir)
            if status == 0:
                break
            failure_class = classify_retryable_failure(log_path.read_text(encoding="utf-8"))
            if failure_class not in {"retryable_vllm_cache", "retryable_vllm_resource"} or attempt == REPEAT10_MAX_ATTEMPTS:
                break
            log(
                f"RETRY {spec.experiment_id} attempt={attempt} class={failure_class} "
                f"status={status}"
            )
            time.sleep(5)

        append_repeat10_summary_row(summary_path, spec, status)
        had_failure = had_failure or int(status != 0)
        log(f"END {spec.experiment_id} dataset={spec.dataset} seed={spec.seed} status={status}")
        time.sleep(2)

    return had_failure


if __name__ == "__main__":
    raise SystemExit(run_repeat10_batch())
