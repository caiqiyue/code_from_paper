from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from .thesis_bridge import load_yaml_config

REPEAT15_SUMMARY_HEADER = [
    "experiment",
    "dataset",
    "seed",
    "status",
    "mode",
    "regime",
    "configured_seed_top_k",
    "resolved_seed_top_k",
    "runner_up_seed_top_k",
    "selection_stage",
    "fallback_used",
    "policy_fallback_used",
    "feasible_budgets",
    "shape_score",
    "best_top1",
    "best_top3",
    "best_top5",
    "best_top10",
]

FULL_RUN_DATASETS: list[tuple[str, Path]] = [
    ("jobs", Path("configs/experiments/single_node_tuning_round19/full_run/r19_full_jobs.yaml")),
    (
        "congressional",
        Path("configs/experiments/single_node_tuning_round19/full_run/r19_full_congressional.yaml"),
    ),
    ("forums", Path("configs/experiments/single_node_tuning_round19/full_run/r19_full_forums.yaml")),
    (
        "microblog",
        Path("configs/experiments/single_node_tuning_round19/full_run/r19_full_microblog.yaml"),
    ),
]


@dataclass(frozen=True)
class Repeat15RunSpec:
    dataset: str
    seed: int
    experiment_id: str
    base_config: Path
    relative_output_root: Path


def resolve_repeat15_project_root(module_file: str | Path | None = None) -> Path:
    module_path = Path(module_file) if module_file is not None else Path(__file__)
    return module_path.resolve().parents[1]


def build_repeat15_run_specs() -> list[Repeat15RunSpec]:
    specs: list[Repeat15RunSpec] = []
    for seed in range(1, 16):
        seed_padded = f"{seed:02d}"
        round_tag = f"round{seed_padded}"
        for dataset, base_config in FULL_RUN_DATASETS:
            experiment_id = f"r19_repeat15_{round_tag}_{dataset}_seed{seed_padded}"
            specs.append(
                Repeat15RunSpec(
                    dataset=dataset,
                    seed=seed,
                    experiment_id=experiment_id,
                    base_config=base_config,
                    relative_output_root=Path("paper-new-round19/outputs/repeat15_rounds") / experiment_id,
                )
            )
    return specs


def write_repeat15_config(spec: Repeat15RunSpec, target_path: str | Path) -> Path:
    config = load_yaml_config(spec.base_config)
    config.setdefault("meta", {})["seed"] = spec.seed
    config["meta"]["experiment_id"] = spec.experiment_id
    config.setdefault("paths", {})["output_root"] = spec.relative_output_root.as_posix()

    resolved_target = Path(target_path)
    resolved_target.parent.mkdir(parents=True, exist_ok=True)
    resolved_target.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return resolved_target


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _stringify(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _extract_feasible_budgets(calibration: dict) -> object:
    if bool(calibration.get("policy_fallback_used", False)) and calibration.get("policy_feasible_budgets") is not None:
        return calibration.get("policy_feasible_budgets")
    if calibration.get("feasible_budgets") is not None:
        return calibration.get("feasible_budgets")
    coverage_constraint = calibration.get("coverage_constraint") or {}
    if coverage_constraint.get("feasible_budgets") is not None:
        return coverage_constraint.get("feasible_budgets")
    if calibration.get("policy_feasible_budgets") is not None:
        return calibration.get("policy_feasible_budgets")
    return None


def append_repeat15_summary_row(
    summary_path: str | Path,
    experiment: str,
    dataset: str,
    seed: int,
    status: int,
    output_dir: str | Path,
) -> None:
    summary_path = Path(summary_path)
    output_dir = Path(output_dir)
    calibration = _read_json(output_dir / "stage1_budget_calibration.json")
    evaluation = _read_json(output_dir / "eval" / "downstream_eval_summary.json")
    metrics = evaluation.get("metrics") or {}

    row = [
        experiment,
        dataset,
        seed,
        status,
        calibration.get("mode", "NA"),
        calibration.get("regime", "NA"),
        calibration.get("configured_seed_top_k", "NA"),
        calibration.get("resolved_seed_top_k", "NA"),
        calibration.get("runner_up_seed_top_k", "NA"),
        calibration.get("selection_stage", "NA"),
        calibration.get("fallback_used", "NA"),
        calibration.get("policy_fallback_used", "NA"),
        _stringify(_extract_feasible_budgets(calibration)),
        calibration.get("shape_score", "NA"),
        metrics.get("best_top1", "NA"),
        metrics.get("best_top3", "NA"),
        metrics.get("best_top5", "NA"),
        metrics.get("best_top10", "NA"),
    ]

    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write("\t".join(str(item) for item in row) + "\n")


def append_repeat15_error_row(
    summary_path: str | Path,
    experiment: str,
    dataset: str,
    seed: int,
    status: int,
) -> None:
    row = [experiment, dataset, seed, status] + ["ERROR"] * (len(REPEAT15_SUMMARY_HEADER) - 4)
    with Path(summary_path).open("a", encoding="utf-8") as handle:
        handle.write("\t".join(str(item) for item in row) + "\n")


def initialize_repeat15_summary(summary_path: str | Path) -> Path:
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\t".join(REPEAT15_SUMMARY_HEADER) + "\n", encoding="utf-8")
    return summary_path


def run_repeat15_batch(project_root: str | Path | None = None) -> int:
    root = Path(project_root).resolve() if project_root is not None else resolve_repeat15_project_root()
    logdir = root / "logs"
    tmpdir = root / "tmp_round19_repeat15"
    output_root = root / "outputs" / "repeat15_rounds"
    summary_path = logdir / "round19_full_repeat15_summary.tsv"
    master_path = logdir / "round19_full_repeat15_master.log"

    logdir.mkdir(parents=True, exist_ok=True)
    tmpdir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    initialize_repeat15_summary(summary_path)
    master_path.write_text("", encoding="utf-8")

    had_failure = 0

    def log(message: str) -> None:
        line = f"{datetime.now().strftime('%F %T')} {message}"
        print(line, flush=True)
        with master_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    for spec in build_repeat15_run_specs():
        config_path = tmpdir / f"{spec.experiment_id}.yaml"
        output_dir = root / spec.relative_output_root
        log_path = logdir / f"{spec.experiment_id}.log"

        write_repeat15_config(spec, config_path)
        shutil.rmtree(output_dir, ignore_errors=True)
        log(f"START {spec.experiment_id} dataset={spec.dataset} seed={spec.seed} cfg={config_path}")

        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [sys.executable, "-m", "paper_new_selector.run_selector_single_node", "--config", str(config_path)],
                cwd=root,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )

        if completed.returncode == 0:
            append_repeat15_summary_row(summary_path, spec.experiment_id, spec.dataset, spec.seed, 0, output_dir)
        else:
            had_failure = 1
            append_repeat15_error_row(summary_path, spec.experiment_id, spec.dataset, spec.seed, completed.returncode)

        log(f"END {spec.experiment_id} dataset={spec.dataset} seed={spec.seed} status={completed.returncode}")
        time.sleep(2)

    return had_failure


if __name__ == "__main__":
    raise SystemExit(run_repeat15_batch())
