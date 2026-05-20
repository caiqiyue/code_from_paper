from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


E1_SEEN_DATASETS = ("jobs", "congressional", "forums", "microblog")
E1_SMOKE_SEEDS = [42]
E1_PILOT_SEEDS = [42, 123, 456]
E1_REPEAT10_SEEDS = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 100, 200]
E1_REPEAT15_SEEDS = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 100, 200, 300, 400, 500, 600, 700]
E1_REPEAT30_SEEDS = [
    42, 123, 456, 789, 1024, 2048, 4096, 8192, 100, 200,
    300, 400, 500, 600, 700, 800, 900, 1111, 1212, 1313,
    1414, 1515, 1616, 1717, 1818, 1919, 2020, 2222, 2468, 3141,
]
E1_MODE_SEEDS = {
    "thesis_main_seen_smoke": E1_SMOKE_SEEDS,
    "thesis_main_seen_pilot": E1_PILOT_SEEDS,
    "thesis_main_seen_repeat10": E1_REPEAT10_SEEDS,
    "thesis_main_seen_repeat15": E1_REPEAT15_SEEDS,
    "thesis_main_seen_repeat30": E1_REPEAT30_SEEDS,
}
E1_MODE_CONFIG_DIRS = {
    "thesis_main_seen_smoke": "thesis_e1_main_seen_smoke",
    "thesis_main_seen_pilot": "thesis_e1_main_seen_pilot",
    "thesis_main_seen_repeat10": "thesis_e1_main_seen_repeat10",
    "thesis_main_seen_repeat15": "thesis_e1_main_seen_repeat15",
    "thesis_main_seen_repeat30": "thesis_e1_main_seen_repeat30",
}
E1_DEFAULT_TARGET_GPU_NAME_TOKEN = "RTX A6000"
E1_DEFAULT_MIN_FREE_GB_FOR_VLLM = 24.0
E1_DEFAULT_GPU_WAIT_POLL_SECONDS = 30
E1_DEFAULT_GPU_WAIT_TIMEOUT_SECONDS = 6 * 60 * 60
E1_RETRYABLE_RESOURCE_FAILURE_MARKERS = (
    "No available memory for the cache blocks",
    "vllm_runtime_gpu_oom",
    "CUDA out of memory",
)
E1_METHODS: dict[str, dict[str, str | None]] = {
    "pretext": {
        "display_name": "PrE-Text",
        "kind": "selector_single_node",
        "template_key": None,
        "candidate_template_key": "expand_private",
        "implementation_key": "expand_private",
        "mapping_status": "needs_verification",
    },
    "round19": {
        "display_name": "round19",
        "kind": "selector_single_node",
        "config_family": "single_node_tuning_round19",
        "config_source": "generated_from_round19_base",
        "implementation_key": "round19",
        "mapping_status": "canonical_round19",
    },
    "wasp": {
        "display_name": "WASP",
        "kind": "external_baseline",
        "template_key": "wasp",
        "implementation_key": "wasp",
        "mapping_status": "external_adapter",
    },
    "dpga": {
        "display_name": "DPGA-TextSyn",
        "kind": "external_baseline",
        "template_key": "dpga",
        "implementation_key": "dpga",
        "mapping_status": "external_adapter",
    },
}
E1_METHOD_ORDER = ("pretext", "round19", "wasp", "dpga")
E1_SUMMARY_FIELDS = [
    "experiment_id",
    "method",
    "method_display_name",
    "dataset",
    "seed",
    "status",
    "config_path",
    "output_root",
    "source_artifact_path",
    "implementation_key",
    "pretext_template_key",
    "mapping_status",
    "best_top1",
    "best_top3",
    "best_top5",
    "best_top10",
    "error_excerpt",
]


@dataclass(frozen=True)
class E1BaselineSpec:
    method: str
    method_display_name: str
    dataset: str
    seed: int
    experiment_id: str
    kind: str
    relative_config_path: Path
    relative_output_root: Path
    source_artifact_path: Path | None
    implementation_key: str
    pretext_template_key: str
    mapping_status: str


def resolve_project_root(module_file: str | Path | None = None) -> Path:
    module_path = Path(module_file) if module_file is not None else Path(__file__)
    return module_path.resolve().parents[1]


def _mode_config_dir(mode: str) -> str:
    if mode not in E1_MODE_CONFIG_DIRS:
        raise ValueError(f"Unsupported E1 mode: {mode}")
    return E1_MODE_CONFIG_DIRS[mode]


def _mode_seeds(mode: str) -> list[int]:
    if mode not in E1_MODE_SEEDS:
        raise ValueError(f"Unsupported E1 mode: {mode}")
    return list(E1_MODE_SEEDS[mode])


def _external_artifact(method: str, dataset: str, seed: int) -> Path | None:
    if method == "wasp":
        return Path("WASP") / "outputs" / "paper_new_screening" / "thesis_e1_main" / dataset / f"seed{seed}" / "train.jsonl"
    if method == "dpga":
        return (
            Path("DPGA-TextSyn")
            / "outputs"
            / "paper_new_screening"
            / "thesis_e1_main"
            / dataset
            / f"seed{seed}"
            / "epoch_all.json"
        )
    return None


def resolve_project_output_path(project_root: Path, configured_output_root: Path) -> Path:
    output_root = Path(configured_output_root)
    if output_root.is_absolute():
        return output_root
    parts = output_root.parts
    if parts and parts[0] in {project_root.name, "paper-new-round19"}:
        output_root = Path(*parts[1:])
    return project_root / output_root


def build_e1_run_specs(mode: str) -> list[E1BaselineSpec]:
    specs: list[E1BaselineSpec] = []
    config_dir = Path("configs") / "experiments" / _mode_config_dir(mode)
    for method in E1_METHOD_ORDER:
        method_cfg = E1_METHODS[method]
        for dataset in E1_SEEN_DATASETS:
            for seed in _mode_seeds(mode):
                experiment_id = f"{method}_{dataset}_seed{seed}"
                output_root = (
                    Path("paper-new-round19")
                    / "outputs"
                    / _mode_config_dir(mode)
                    / method
                    / dataset
                    / f"seed{seed}"
                )
                specs.append(
                    E1BaselineSpec(
                        method=method,
                        method_display_name=str(method_cfg["display_name"]),
                        dataset=dataset,
                        seed=seed,
                        experiment_id=experiment_id,
                        kind=str(method_cfg["kind"]),
                        relative_config_path=config_dir / f"{experiment_id}.yaml",
                        relative_output_root=output_root,
                        source_artifact_path=_external_artifact(method, dataset, seed),
                        implementation_key=str(method_cfg.get("implementation_key") or ""),
                        pretext_template_key=(
                            str(method_cfg.get("candidate_template_key") or "")
                            if method == "pretext"
                            else ""
                        ),
                        mapping_status=str(method_cfg.get("mapping_status") or ""),
                    )
                )
    return specs


def _build_config(spec: E1BaselineSpec, mode: str) -> dict[str, Any]:
    if spec.method == "round19":
        inherits = [
            "../single_node_tuning_round19/_base_selector_tuning_round19.yaml",
            f"../single_node_tuning_round19/_data_{spec.dataset}.yaml",
        ]
    else:
        template_key = spec.implementation_key
        prefix = {
            "expand_private": "ep",
            "wasp": "wasp",
            "dpga": "dpga",
        }[template_key]
        inherits = [f"../single_run_baseline_screening/{prefix}_{spec.dataset}_single_run.yaml"]

    cfg: dict[str, Any] = {
        "inherits": inherits,
        "meta": {
            "experiment_id": spec.experiment_id,
            "seed": spec.seed,
            "dataset_name": spec.dataset,
            "stage": mode,
            "method": spec.method,
            "method_display_name": spec.method_display_name,
            "implementation_key": spec.implementation_key,
            "pretext_template_key": spec.pretext_template_key,
            "mapping_status": spec.mapping_status,
        },
        "paths": {
            "output_root": spec.relative_output_root.as_posix(),
        },
        "protocol": {
            "experiment_family": "thesis_e1_main_seen",
            "mode": mode,
        },
    }
    if spec.source_artifact_path is not None:
        cfg["external_baseline"] = {
            "source_artifact_path": spec.source_artifact_path.as_posix(),
            "source_dataset_name": spec.dataset,
            "summary_output_path": (spec.relative_output_root / "stage1_summary.json").as_posix(),
        }
    return cfg


def write_e1_config(spec: E1BaselineSpec, *, project_root: Path, mode: str) -> Path:
    target = project_root / spec.relative_config_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(_build_config(spec, mode), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target


def _write_manifest(project_root: Path, mode: str, specs: list[E1BaselineSpec]) -> Path:
    manifest = project_root / "configs" / "experiments" / _mode_config_dir(mode) / f"{_mode_config_dir(mode)}_manifest.tsv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment_id",
        "method",
        "method_display_name",
        "dataset",
        "seed",
        "config_path",
        "output_root",
        "source_artifact_path",
        "implementation_key",
        "pretext_template_key",
        "mapping_status",
    ]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for spec in specs:
            writer.writerow(
                {
                    "experiment_id": spec.experiment_id,
                    "method": spec.method,
                    "method_display_name": spec.method_display_name,
                    "dataset": spec.dataset,
                    "seed": spec.seed,
                    "config_path": spec.relative_config_path.as_posix(),
                    "output_root": spec.relative_output_root.as_posix(),
                    "source_artifact_path": spec.source_artifact_path.as_posix() if spec.source_artifact_path else "",
                    "implementation_key": spec.implementation_key,
                    "pretext_template_key": spec.pretext_template_key,
                    "mapping_status": spec.mapping_status,
                }
            )
    return manifest


def parse_nvidia_smi_memory_report(report_text: str, *, target_name_token: str) -> tuple[str, float]:
    for raw_line in report_text.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 3:
            continue
        index, name, free_mib = parts
        if target_name_token in name:
            return index, float(free_mib) / 1024.0
    raise RuntimeError(f"Could not find GPU with token {target_name_token!r} in nvidia-smi output.")


def query_target_gpu_free_gb(*, target_name_token: str) -> tuple[str, float]:
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
    return parse_nvidia_smi_memory_report(completed.stdout, target_name_token=target_name_token)


def wait_for_vllm_capacity(
    log_path: Path,
    *,
    target_name_token: str,
    minimum_free_gb: float,
    poll_seconds: float,
    timeout_seconds: float,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout_seconds
    while True:
        gpu_index, free_gb = query_target_gpu_free_gb(target_name_token=target_name_token)
        with log_path.open("a", encoding="utf-8") as handle:
            if free_gb >= minimum_free_gb:
                handle.write(
                    f"{datetime.now().isoformat()} GPU_READY target={target_name_token} "
                    f"index={gpu_index} free_gb={free_gb:.2f} threshold={minimum_free_gb:.2f}\n"
                )
                return
            handle.write(
                f"{datetime.now().isoformat()} GPU_WAIT target={target_name_token} "
                f"index={gpu_index} free_gb={free_gb:.2f} threshold={minimum_free_gb:.2f}\n"
            )
        if time.time() >= deadline:
            raise TimeoutError(
                f"A6000 free memory stayed below {minimum_free_gb:.2f} GiB for {timeout_seconds:.0f} seconds."
            )
        time.sleep(poll_seconds)


def is_retryable_resource_failure(text: str) -> bool:
    return any(marker in text for marker in E1_RETRYABLE_RESOURCE_FAILURE_MARKERS)


def materialize_e1_configs(project_root: str | Path | None = None, *, mode: str = "thesis_main_seen_pilot") -> list[Path]:
    root = Path(project_root).resolve() if project_root is not None else resolve_project_root()
    specs = build_e1_run_specs(mode)
    generated = [write_e1_config(spec, project_root=root, mode=mode) for spec in specs]
    _write_manifest(root, mode, specs)
    return generated


def load_manifest(project_root: Path, mode: str) -> list[E1BaselineSpec]:
    manifest = project_root / "configs" / "experiments" / _mode_config_dir(mode) / f"{_mode_config_dir(mode)}_manifest.tsv"
    if not manifest.exists():
        materialize_e1_configs(project_root, mode=mode)
    rows: list[E1BaselineSpec] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows.append(
                E1BaselineSpec(
                    method=str(row["method"]),
                    method_display_name=str(row["method_display_name"]),
                    dataset=str(row["dataset"]),
                    seed=int(row["seed"]),
                    experiment_id=str(row["experiment_id"]),
                    kind=str(E1_METHODS[str(row["method"])]["kind"]),
                    relative_config_path=Path(str(row["config_path"])),
                    relative_output_root=Path(str(row["output_root"])),
                    source_artifact_path=Path(str(row["source_artifact_path"])) if row.get("source_artifact_path") else None,
                    implementation_key=str(row.get("implementation_key", "")),
                    pretext_template_key=str(row.get("pretext_template_key", "")),
                    mapping_status=str(row.get("mapping_status", "")),
                )
            )
    return rows


def build_e1_command(spec: E1BaselineSpec, config_path: Path) -> list[str]:
    module = (
        "paper_new_selector.run_external_baseline_single_run"
        if spec.kind == "external_baseline"
        else "paper_new_selector.run_selector_single_node"
    )
    return [sys.executable, "-m", module, "--config", str(config_path)]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_eval_metric(eval_summary: dict[str, Any], metric: str) -> Any:
    if metric in eval_summary:
        return eval_summary.get(metric)
    metrics = eval_summary.get("metrics")
    if isinstance(metrics, dict):
        return metrics.get(metric, "")
    return ""


def build_summary_row(
    spec: E1BaselineSpec,
    *,
    project_root: Path,
    status: str,
    error_excerpt: str,
) -> dict[str, Any]:
    resolved_output_root = resolve_project_output_path(project_root, spec.relative_output_root)
    eval_summary = _read_json(resolved_output_root / "eval" / "downstream_eval_summary.json")
    return {
        "experiment_id": spec.experiment_id,
        "method": spec.method,
        "method_display_name": spec.method_display_name,
        "dataset": spec.dataset,
        "seed": spec.seed,
        "status": status,
        "config_path": spec.relative_config_path.as_posix(),
        "output_root": spec.relative_output_root.as_posix(),
        "source_artifact_path": spec.source_artifact_path.as_posix() if spec.source_artifact_path else "",
        "implementation_key": spec.implementation_key,
        "pretext_template_key": spec.pretext_template_key,
        "mapping_status": spec.mapping_status,
        "best_top1": _extract_eval_metric(eval_summary, "best_top1"),
        "best_top3": _extract_eval_metric(eval_summary, "best_top3"),
        "best_top5": _extract_eval_metric(eval_summary, "best_top5"),
        "best_top10": _extract_eval_metric(eval_summary, "best_top10"),
        "error_excerpt": error_excerpt,
    }


def _append_tsv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=E1_SUMMARY_FIELDS, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def completed_experiment_ids(summary_tsv: Path) -> set[str]:
    if not summary_tsv.exists():
        return set()
    with summary_tsv.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            str(row["experiment_id"])
            for row in csv.DictReader(handle, delimiter="\t")
            if str(row.get("status", "")).lower() == "success"
        }


def run_e1_batch(
    project_root: str | Path | None = None,
    *,
    mode: str = "thesis_main_seen_pilot",
    dry_run: bool = True,
    limit: int = 0,
    max_attempts: int = 2,
    retry_delay_seconds: float = 10.0,
    target_gpu_name_token: str = E1_DEFAULT_TARGET_GPU_NAME_TOKEN,
    min_free_gb_for_vllm: float = E1_DEFAULT_MIN_FREE_GB_FOR_VLLM,
    gpu_wait_poll_seconds: float = E1_DEFAULT_GPU_WAIT_POLL_SECONDS,
    gpu_wait_timeout_seconds: float = E1_DEFAULT_GPU_WAIT_TIMEOUT_SECONDS,
    reset_summary: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve() if project_root is not None else resolve_project_root()
    materialize_e1_configs(root, mode=mode)
    specs = load_manifest(root, mode)
    summary_tsv = root / "logs" / f"{_mode_config_dir(mode)}_summary.tsv"
    master_log = root / "logs" / f"{_mode_config_dir(mode)}_master.log"
    if reset_summary and not dry_run:
        summary_tsv.unlink(missing_ok=True)
        master_log.unlink(missing_ok=True)
    done = completed_experiment_ids(summary_tsv)
    specs = [spec for spec in specs if spec.experiment_id not in done]
    if limit > 0:
        specs = specs[:limit]
    if dry_run:
        return {
            "mode": mode,
            "pending_count": len(specs),
            "manifest": str((root / "configs" / "experiments" / _mode_config_dir(mode) / f"{_mode_config_dir(mode)}_manifest.tsv").resolve()),
            "summary_tsv": str(summary_tsv.resolve()),
            "first_experiments": [spec.experiment_id for spec in specs[:10]],
        }

    failures = 0
    for spec in specs:
        config_path = root / spec.relative_config_path
        success = False
        last_error = ""
        for attempt in range(1, max_attempts + 1):
            with master_log.open("a", encoding="utf-8") as handle:
                handle.write(f"{datetime.now().isoformat()} START {spec.experiment_id} attempt={attempt}\n")
            wait_for_vllm_capacity(
                master_log,
                target_name_token=target_gpu_name_token,
                minimum_free_gb=min_free_gb_for_vllm,
                poll_seconds=gpu_wait_poll_seconds,
                timeout_seconds=gpu_wait_timeout_seconds,
            )
            child_env = dict(os.environ)
            child_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            child_env["CUDA_VISIBLE_DEVICES"] = child_env.get("CUDA_VISIBLE_DEVICES", "1")
            completed = subprocess.run(
                build_e1_command(spec, config_path),
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
                env=child_env,
            )
            last_error = (completed.stderr or completed.stdout)
            status = "success" if completed.returncode == 0 else "failed"
            _append_tsv(
                summary_tsv,
                build_summary_row(
                    spec,
                    project_root=root,
                    status=status,
                    error_excerpt=last_error[:400].replace("\n", "\\n"),
                ),
            )
            with master_log.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{datetime.now().isoformat()} END {spec.experiment_id} "
                    f"status={status} returncode={completed.returncode}\n"
                )
            if completed.returncode == 0:
                success = True
                break
            if attempt >= max_attempts or not is_retryable_resource_failure(last_error):
                break
            if retry_delay_seconds > 0:
                time.sleep(retry_delay_seconds)
        failures += int(not success)
    return {"mode": mode, "status": "failed" if failures else "success", "failures": failures}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or run thesis E1 baseline configs")
    parser.add_argument("--mode", choices=sorted(E1_MODE_SEEDS.keys()), default="thesis_main_seen_pilot")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true", help="Actually run subprocesses; default is dry-run.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--retry-delay-seconds", type=float, default=10.0)
    parser.add_argument("--target-gpu-name-token", default=E1_DEFAULT_TARGET_GPU_NAME_TOKEN)
    parser.add_argument("--min-free-gb-for-vllm", type=float, default=E1_DEFAULT_MIN_FREE_GB_FOR_VLLM)
    parser.add_argument("--gpu-wait-poll-seconds", type=float, default=E1_DEFAULT_GPU_WAIT_POLL_SECONDS)
    parser.add_argument("--gpu-wait-timeout-seconds", type=float, default=E1_DEFAULT_GPU_WAIT_TIMEOUT_SECONDS)
    parser.add_argument("--reset-summary", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_e1_batch(
        mode=args.mode,
        dry_run=not args.execute,
        limit=args.limit,
        max_attempts=args.max_attempts,
        retry_delay_seconds=args.retry_delay_seconds,
        target_gpu_name_token=args.target_gpu_name_token,
        min_free_gb_for_vllm=args.min_free_gb_for_vllm,
        gpu_wait_poll_seconds=args.gpu_wait_poll_seconds,
        gpu_wait_timeout_seconds=args.gpu_wait_timeout_seconds,
        reset_summary=args.reset_summary,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status", "success") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
