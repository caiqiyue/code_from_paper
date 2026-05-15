from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


COLLECTION_MANIFEST = Path(
    "configs/experiments/single_node_tuning_round19/round23_collection_repeat40/round19_round23_collection_repeat40_manifest.tsv"
)
COLLECTION_SUMMARY_HEADER = [
    "experiment_id",
    "dataset_name",
    "meta_seed",
    "budget_k",
    "attempts",
    "status",
    "returncode",
    "error_class",
    "best_top1",
    "best_top3",
    "best_top5",
    "best_top10",
]
COLLECTION_SUMMARY_NAME = "round19_round23_collection_repeat40_summary.tsv"
COLLECTION_MASTER_NAME = "round19_round23_collection_repeat40_master.log"
COLLECTION_DEFAULT_CUDA_VISIBLE_DEVICES = "1"
COLLECTION_CUDA_OVERRIDE_ENV = "ROUND19_R23_COLLECTION_CUDA_VISIBLE_DEVICES"
COLLECTION_TARGET_GPU_NAME_TOKEN = "RTX A6000"
COLLECTION_MIN_FREE_GB_FOR_VLLM = 2.0
COLLECTION_GPU_POLL_SECONDS = 30
COLLECTION_GPU_WAIT_TIMEOUT_SECONDS = 6 * 60 * 60
COLLECTION_MAX_ATTEMPTS = 3
COLLECTION_RETRY_SLEEP_SECONDS = 5
COLLECTION_EXPERIMENT_SLEEP_SECONDS = 2
COLLECTION_TRANSIENT_FAILURE_MARKERS = (
    "# GPU blocks: 0",
    "No available memory for the cache blocks",
    "vllm_runtime_gpu_oom",
    "CUDA out of memory",
    "Stage 2 passed the startup memory gate but vLLM later hit CUDA out of memory.",
)


@dataclass(frozen=True)
class Round19Round23CollectionSpec:
    experiment_id: str
    dataset_name: str
    meta_seed: int
    budget_k: int
    config_path: Path
    output_root: Path
    group_name: str


def resolve_round19_round23_collection_project_root(module_file: str | Path | None = None) -> Path:
    module_path = Path(module_file) if module_file is not None else Path(__file__)
    return module_path.resolve().parents[1]


def build_round19_round23_collection_specs(
    project_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> list[Round19Round23CollectionSpec]:
    root = Path(project_root).resolve() if project_root is not None else resolve_round19_round23_collection_project_root()
    manifest = root / COLLECTION_MANIFEST if manifest_path is None else Path(manifest_path).resolve()
    specs: list[Round19Round23CollectionSpec] = []
    with manifest.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            config_rel = Path(row["config_path"])
            specs.append(
                Round19Round23CollectionSpec(
                    experiment_id=row["experiment_id"],
                    dataset_name=row["dataset_name"],
                    meta_seed=int(row["meta_seed"]),
                    budget_k=int(row["budget_k"]),
                    config_path=(root / config_rel).resolve(),
                    output_root=normalize_collection_output_root(root, row["output_root"]),
                    group_name=row.get("group_name", "round19_round23_collection_repeat40"),
                )
            )
    return specs


def normalize_collection_output_root(project_root: Path, configured_output_root: str | Path) -> Path:
    path = Path(configured_output_root)
    if path.is_absolute():
        return path.resolve()
    parts = path.parts
    if parts and parts[0] == project_root.name:
        path = Path(*parts[1:])
    return (project_root / path).resolve()


def initialize_round19_round23_collection_summary(summary_path: str | Path) -> Path:
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\t".join(COLLECTION_SUMMARY_HEADER) + "\n", encoding="utf-8")
    return summary_path


def build_round19_round23_collection_child_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(
        env.get(COLLECTION_CUDA_OVERRIDE_ENV, env.get("CUDA_VISIBLE_DEVICES", COLLECTION_DEFAULT_CUDA_VISIBLE_DEVICES))
    )
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return env


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


def query_collection_a6000_free_gb() -> tuple[str, float]:
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
    return parse_nvidia_smi_memory_report(completed.stdout, target_name_token=COLLECTION_TARGET_GPU_NAME_TOKEN)


def wait_for_collection_vllm_capacity(
    log,
    *,
    minimum_free_gb: float = COLLECTION_MIN_FREE_GB_FOR_VLLM,
) -> None:
    deadline = time.time() + COLLECTION_GPU_WAIT_TIMEOUT_SECONDS
    while True:
        gpu_index, free_gb = query_collection_a6000_free_gb()
        if free_gb >= minimum_free_gb:
            log(
                f"GPU_READY target={COLLECTION_TARGET_GPU_NAME_TOKEN} index={gpu_index} "
                f"free_gb={free_gb:.2f} threshold={minimum_free_gb:.2f}"
            )
            return
        if time.time() >= deadline:
            raise TimeoutError(
                f"A6000 free memory stayed below {minimum_free_gb:.2f} GiB for "
                f"{COLLECTION_GPU_WAIT_TIMEOUT_SECONDS} seconds."
            )
        log(
            f"GPU_WAIT target={COLLECTION_TARGET_GPU_NAME_TOKEN} index={gpu_index} "
            f"free_gb={free_gb:.2f} threshold={minimum_free_gb:.2f}"
        )
        time.sleep(COLLECTION_GPU_POLL_SECONDS)


def classify_collection_failure(log_text: str) -> str:
    if "# GPU blocks: 0" in log_text or "No available memory for the cache blocks" in log_text:
        return "retryable_vllm_cache"
    for marker in COLLECTION_TRANSIENT_FAILURE_MARKERS[2:]:
        if marker in log_text:
            return "retryable_vllm_resource"
    return "runtime_failure"


def reset_collection_output_dir(output_dir: Path) -> None:
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_collection_output_dir(output_dir: Path) -> tuple[int, str, dict[str, object]]:
    try:
        _load_json(output_dir / "collection" / "context_summary.json")
        final_result = _load_json(output_dir / "collection" / "final_result_summary.json")
        eval_summary = _load_json(output_dir / "eval" / "downstream_eval_summary.json")
        budget_table = output_dir / "collection" / "budget_table.jsonl"
        if not budget_table.exists():
            return 92, "missing_budget_table", {}
        lines = [line for line in budget_table.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            return 93, "empty_budget_table", {}
        metrics = eval_summary.get("metrics") or {}
        for key in ("best_top1", "best_top3", "best_top5", "best_top10"):
            if final_result.get(key) is None:
                return 94, "missing_final_metrics", {}
            if final_result.get(key) != metrics.get(key):
                return 95, "mismatched_eval_metrics", {}
        return 0, "completed", metrics
    except FileNotFoundError:
        return 91, "missing_required_artifact", {}
    except json.JSONDecodeError:
        return 96, "invalid_json_artifact", {}


def append_round19_round23_collection_summary_row(
    summary_path: str | Path,
    spec: Round19Round23CollectionSpec,
    *,
    attempts: int,
    status: str,
    returncode: int,
    error_class: str,
    metrics: dict[str, object] | None = None,
) -> None:
    metrics = metrics or {}
    row = [
        spec.experiment_id,
        spec.dataset_name,
        spec.meta_seed,
        spec.budget_k,
        attempts,
        status,
        returncode,
        error_class,
        metrics.get("best_top1", "NA"),
        metrics.get("best_top3", "NA"),
        metrics.get("best_top5", "NA"),
        metrics.get("best_top10", "NA"),
    ]
    with Path(summary_path).open("a", encoding="utf-8") as handle:
        handle.write("\t".join(str(item) for item in row) + "\n")


def run_round19_round23_collection_batch(project_root: str | Path | None = None) -> int:
    root = Path(project_root).resolve() if project_root is not None else resolve_round19_round23_collection_project_root()
    logdir = root / "logs"
    summary_path = logdir / COLLECTION_SUMMARY_NAME
    master_path = logdir / COLLECTION_MASTER_NAME
    child_env = build_round19_round23_collection_child_env()

    logdir.mkdir(parents=True, exist_ok=True)
    initialize_round19_round23_collection_summary(summary_path)
    master_path.write_text("", encoding="utf-8")
    had_failure = 0

    def log(message: str) -> None:
        line = f"{datetime.now().strftime('%F %T')} {message}"
        print(line, flush=True)
        with master_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    for spec in build_round19_round23_collection_specs(root):
        log_path = logdir / f"{spec.experiment_id}.log"
        log(f"START {spec.experiment_id} dataset={spec.dataset_name} seed={spec.meta_seed} k={spec.budget_k} cfg={spec.config_path}")

        final_status = "failed"
        final_returncode = 1
        final_error_class = "runtime_failure"
        final_metrics: dict[str, object] = {}
        attempts = 0

        for attempt in range(1, COLLECTION_MAX_ATTEMPTS + 1):
            attempts = attempt
            wait_for_collection_vllm_capacity(log)
            reset_collection_output_dir(spec.output_root)
            mode = "w" if attempt == 1 else "a"
            with log_path.open(mode, encoding="utf-8") as handle:
                if attempt > 1:
                    handle.write(f"\n===== retry attempt {attempt} =====\n")
                completed = subprocess.run(
                    [sys.executable, "-m", "paper_new_selector.run_selector_single_node", "--config", str(spec.config_path)],
                    cwd=root,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                    env=child_env,
                )
            final_returncode = int(completed.returncode)
            validation_code, validation_status, validation_metrics = validate_collection_output_dir(spec.output_root)
            if completed.returncode == 0 and validation_code == 0:
                final_status = "completed"
                final_error_class = validation_status
                final_metrics = validation_metrics
                break

            log_text = log_path.read_text(encoding="utf-8")
            failure_class = classify_collection_failure(log_text)
            if completed.returncode == 0 and validation_code != 0:
                failure_class = validation_status
                final_returncode = validation_code
            final_error_class = failure_class
            log(
                f"RETRY {spec.experiment_id} attempt={attempt} "
                f"returncode={final_returncode} class={failure_class}"
            )
            if attempt < COLLECTION_MAX_ATTEMPTS:
                time.sleep(COLLECTION_RETRY_SLEEP_SECONDS)

        append_round19_round23_collection_summary_row(
            summary_path,
            spec,
            attempts=attempts,
            status=final_status,
            returncode=final_returncode,
            error_class=final_error_class,
            metrics=final_metrics,
        )
        had_failure = had_failure or int(final_status != "completed")
        log(
            f"END {spec.experiment_id} dataset={spec.dataset_name} seed={spec.meta_seed} "
            f"status={final_status} returncode={final_returncode} attempts={attempts}"
        )
        time.sleep(COLLECTION_EXPERIMENT_SLEEP_SECONDS)

    return had_failure


if __name__ == "__main__":
    raise SystemExit(run_round19_round23_collection_batch())
