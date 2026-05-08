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

ROUND20_REPEAT5_SUMMARY_HEADER = [
    "experiment",
    "base_experiment",
    "dataset",
    "base_seed",
    "repeat_index",
    "status",
    "mode",
    "regime",
    "configured_seed_top_k",
    "resolved_seed_top_k",
    "selection_stage",
    "arbitration_triggered",
    "arbitration_winner_policy",
    "best_top1",
    "best_top3",
    "best_top5",
    "best_top10",
]

ROUND20_REPEAT5_BASE_CONFIGS: list[Path] = sorted(
    list(Path("configs/experiments/single_node_tuning_round20/router_scans").glob("*.yaml"))
    + list(Path("configs/experiments/single_node_tuning_round20/uncertain_repeats").glob("*.yaml"))
)


@dataclass(frozen=True)
class Round20Repeat5RunSpec:
    base_experiment_id: str
    dataset: str
    base_seed: int
    repeat_index: int
    experiment_id: str
    base_config: Path
    relative_output_root: Path


def resolve_round20_repeat5_project_root(module_file: str | Path | None = None) -> Path:
    module_path = Path(module_file) if module_file is not None else Path(__file__)
    return module_path.resolve().parents[1]


def _infer_dataset_from_experiment_id(experiment_id: str) -> str:
    if "_jobs_" in experiment_id or experiment_id.startswith("r20_jobs"):
        return "jobs"
    if "_microblog_" in experiment_id or experiment_id.startswith("r20_microblog"):
        return "microblog"
    raise ValueError(f"Unable to infer dataset from experiment id: {experiment_id}")


def build_round20_repeat5_run_specs(repeat_count: int = 5) -> list[Round20Repeat5RunSpec]:
    specs: list[Round20Repeat5RunSpec] = []
    for base_config in ROUND20_REPEAT5_BASE_CONFIGS:
        config = load_yaml_config(base_config)
        base_experiment_id = str(config["meta"]["experiment_id"])
        dataset = _infer_dataset_from_experiment_id(base_experiment_id)
        base_seed = int(config["meta"]["seed"])
        for repeat_index in range(1, repeat_count + 1):
            experiment_id = f"{base_experiment_id}_repeat{repeat_index:02d}"
            specs.append(
                Round20Repeat5RunSpec(
                    base_experiment_id=base_experiment_id,
                    dataset=dataset,
                    base_seed=base_seed,
                    repeat_index=repeat_index,
                    experiment_id=experiment_id,
                    base_config=base_config,
                    relative_output_root=Path("paper-new-round20/outputs/r20_repeat5") / experiment_id,
                )
            )
    return specs


def write_round20_repeat5_config(spec: Round20Repeat5RunSpec, target_path: str | Path) -> Path:
    config = load_yaml_config(spec.base_config)
    config.setdefault("meta", {})["experiment_id"] = spec.experiment_id
    config.setdefault("paths", {})["output_root"] = spec.relative_output_root.as_posix()

    resolved_target = Path(target_path)
    resolved_target.parent.mkdir(parents=True, exist_ok=True)
    resolved_target.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return resolved_target


def resolve_round20_repeat5_runtime_output_dir(
    project_root: str | Path,
    spec: Round20Repeat5RunSpec,
) -> Path:
    root = Path(project_root)
    return root / "outputs" / "r20_repeat5" / spec.experiment_id


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def initialize_round20_repeat5_summary(summary_path: str | Path) -> Path:
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        "\t".join(ROUND20_REPEAT5_SUMMARY_HEADER) + "\n",
        encoding="utf-8",
    )
    return summary_path


def append_round20_repeat5_summary_row(
    summary_path: str | Path,
    spec: Round20Repeat5RunSpec,
    status: int,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    calibration = _read_json(output_dir / "stage1_budget_calibration.json")
    evaluation = _read_json(output_dir / "eval" / "downstream_eval_summary.json")
    metrics = evaluation.get("metrics") or {}
    row = [
        spec.experiment_id,
        spec.base_experiment_id,
        spec.dataset,
        spec.base_seed,
        spec.repeat_index,
        status,
        calibration.get("mode", "NA"),
        calibration.get("regime", "NA"),
        calibration.get("configured_seed_top_k", "NA"),
        calibration.get("resolved_seed_top_k", "NA"),
        calibration.get("selection_stage", "NA"),
        calibration.get("arbitration_triggered", "NA"),
        calibration.get("arbitration_winner_policy", "NA"),
        metrics.get("best_top1", "NA"),
        metrics.get("best_top3", "NA"),
        metrics.get("best_top5", "NA"),
        metrics.get("best_top10", "NA"),
    ]
    with Path(summary_path).open("a", encoding="utf-8") as handle:
        handle.write("\t".join(str(item) for item in row) + "\n")


def append_round20_repeat5_error_row(
    summary_path: str | Path,
    spec: Round20Repeat5RunSpec,
    status: int,
) -> None:
    row = [
        spec.experiment_id,
        spec.base_experiment_id,
        spec.dataset,
        spec.base_seed,
        spec.repeat_index,
        status,
    ] + ["ERROR"] * (len(ROUND20_REPEAT5_SUMMARY_HEADER) - 6)
    with Path(summary_path).open("a", encoding="utf-8") as handle:
        handle.write("\t".join(str(item) for item in row) + "\n")


def build_round20_repeat5_child_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)
    if env.get("CUDA_VISIBLE_DEVICES") and not env.get("CUDA_DEVICE_ORDER"):
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return env


def run_round20_repeat5_batch(project_root: str | Path | None = None, repeat_count: int = 5) -> int:
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else resolve_round20_repeat5_project_root()
    )
    logdir = root / "logs"
    tmpdir = root / "tmp_round20_repeat5"
    output_root = root / "outputs" / "r20_repeat5"
    summary_path = logdir / "round20_repeat5_summary.tsv"
    master_path = logdir / "round20_repeat5_master.log"

    logdir.mkdir(parents=True, exist_ok=True)
    tmpdir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    initialize_round20_repeat5_summary(summary_path)
    master_path.write_text("", encoding="utf-8")
    child_env = build_round20_repeat5_child_env()
    had_failure = 0

    def log(message: str) -> None:
        line = f"{datetime.now().strftime('%F %T')} {message}"
        print(line, flush=True)
        with master_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    for spec in build_round20_repeat5_run_specs(repeat_count=repeat_count):
        config_path = tmpdir / f"{spec.experiment_id}.yaml"
        output_dir = resolve_round20_repeat5_runtime_output_dir(root, spec)
        log_path = logdir / f"{spec.experiment_id}.log"
        write_round20_repeat5_config(spec, config_path)
        shutil.rmtree(output_dir, ignore_errors=True)
        log(
            f"START {spec.experiment_id} base={spec.base_experiment_id} "
            f"dataset={spec.dataset} repeat={spec.repeat_index} cfg={config_path}"
        )

        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [sys.executable, "-m", "paper_new_selector.run_selector_single_node", "--config", str(config_path)],
                cwd=root,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                env=child_env,
            )

        if completed.returncode == 0:
            append_round20_repeat5_summary_row(summary_path, spec, 0, output_dir)
        else:
            had_failure = 1
            append_round20_repeat5_error_row(summary_path, spec, completed.returncode)

        log(
            f"END {spec.experiment_id} base={spec.base_experiment_id} "
            f"dataset={spec.dataset} repeat={spec.repeat_index} status={completed.returncode}"
        )
        time.sleep(2)

    return had_failure


if __name__ == "__main__":
    raise SystemExit(run_round20_repeat5_batch())
