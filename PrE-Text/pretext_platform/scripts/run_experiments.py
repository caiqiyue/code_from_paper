"""Batch experiment runner for the executable PrE-Text configs.

This runner focuses on the unique runnable PrE-Text experiment configs instead
of the conceptual paper groups that may map to the same config file.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pretext_platform.core.config import load_experiment_config
from pretext_platform.core.preflight import format_preflight_report, run_preflight


PRETEXT_EXPERIMENTS: dict[str, dict[str, Any]] = {
    "jobs_real_eps05": {
        "config": "configs/experiments/jobs_real_eps05.yaml",
        "description": "PrE-Text Jobs ε=0.5",
        "suite": "jobs",
        "tags": ["GP1"],
    },
    "jobs_real_eps129": {
        "config": "configs/experiments/jobs_real_eps129.yaml",
        "description": "PrE-Text Jobs ε=1.29",
        "suite": "jobs",
        "tags": ["GC2", "GP2"],
    },
    "jobs_real_eps758": {
        "config": "configs/experiments/jobs_real_eps758.yaml",
        "description": "PrE-Text Jobs ε=7.58",
        "suite": "jobs",
        "tags": ["GP3"],
    },
    "jobs_real_no_privacy": {
        "config": "configs/experiments/jobs_real_no_privacy.yaml",
        "description": "PrE-Text Jobs without privacy",
        "suite": "jobs",
        "tags": ["GP4"],
    },
    "forums_real_eps05": {
        "config": "configs/experiments/forums_real_eps05.yaml",
        "description": "PrE-Text Forums ε=0.5",
        "suite": "forums",
    },
    "forums_real_eps129": {
        "config": "configs/experiments/forums_real_eps129.yaml",
        "description": "PrE-Text Forums ε=1.29",
        "suite": "forums",
        "tags": ["GC8"],
    },
    "forums_real_eps758": {
        "config": "configs/experiments/forums_real_eps758.yaml",
        "description": "PrE-Text Forums ε=7.58",
        "suite": "forums",
    },
    "forums_real_no_privacy": {
        "config": "configs/experiments/forums_real_no_privacy.yaml",
        "description": "PrE-Text Forums without privacy",
        "suite": "forums",
    },
    "microblog_real_eps05": {
        "config": "configs/experiments/microblog_real_eps05.yaml",
        "description": "PrE-Text Microblog ε=0.5",
        "suite": "microblog",
    },
    "microblog_real_eps129": {
        "config": "configs/experiments/microblog_real_eps129.yaml",
        "description": "PrE-Text Microblog ε=1.29",
        "suite": "microblog",
        "tags": ["GC9"],
    },
    "microblog_real_eps758": {
        "config": "configs/experiments/microblog_real_eps758.yaml",
        "description": "PrE-Text Microblog ε=7.58",
        "suite": "microblog",
    },
    "microblog_real_no_privacy": {
        "config": "configs/experiments/microblog_real_no_privacy.yaml",
        "description": "PrE-Text Microblog without privacy",
        "suite": "microblog",
    },
    "congressional_real_eps05": {
        "config": "configs/experiments/congressional_real_eps05.yaml",
        "description": "PrE-Text Congressional ε=0.5",
        "suite": "congressional",
        "tags": ["GC4"],
    },
    "congressional_real_eps129": {
        "config": "configs/experiments/congressional_real_eps129.yaml",
        "description": "PrE-Text Congressional ε=1.29",
        "suite": "congressional",
    },
    "congressional_real_eps758": {
        "config": "configs/experiments/congressional_real_eps758.yaml",
        "description": "PrE-Text Congressional ε=7.58",
        "suite": "congressional",
    },
    "congressional_real_no_privacy": {
        "config": "configs/experiments/congressional_real_no_privacy.yaml",
        "description": "PrE-Text Congressional without privacy",
        "suite": "congressional",
    },
}

EXPERIMENT_ALIASES = {
    "GC2_jobs_real_eps129": "jobs_real_eps129",
    "GC4_congressional_real_eps05": "congressional_real_eps05",
    "GC8_forums_real_eps129": "forums_real_eps129",
    "GC9_microblog_real_eps129": "microblog_real_eps129",
    "GP1_jobs_real_eps05": "jobs_real_eps05",
    "GP2_jobs_real_eps129": "jobs_real_eps129",
    "GP3_jobs_real_eps758": "jobs_real_eps758",
    "GP4_jobs_real_no_privacy": "jobs_real_no_privacy",
}

EXPERIMENT_SUITES = {
    "jobs": [
        "jobs_real_eps05",
        "jobs_real_eps129",
        "jobs_real_eps758",
        "jobs_real_no_privacy",
    ],
    "forums": [
        "forums_real_eps05",
        "forums_real_eps129",
        "forums_real_eps758",
        "forums_real_no_privacy",
    ],
    "microblog": [
        "microblog_real_eps05",
        "microblog_real_eps129",
        "microblog_real_eps758",
        "microblog_real_no_privacy",
    ],
    "congressional": [
        "congressional_real_eps05",
        "congressional_real_eps129",
        "congressional_real_eps758",
        "congressional_real_no_privacy",
    ],
}

DEFAULT_GLUE_TASKS = ["sst2", "qqp", "qnli", "imdb", "rotten_tomatoes"]
DEFAULT_OUTPUT_BASE = Path("./outputs/pretext_platform")
DEFAULT_LOG_DIR = Path("./logs")
DEFAULT_RESULT_DIR = Path("./results")


class ExperimentLogger:
    """File-backed logger for a single experiment run."""

    def __init__(self, experiment_id: str, log_dir: Path):
        self.experiment_id = experiment_id
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.main_log = self.log_dir / "main.log"
        self.error_log = self.log_dir / "error.log"
        self.stage_log = self.log_dir / "stages.log"
        self.start_time = time.time()
        self._main_handle = self.main_log.open("w", encoding="utf-8")
        self._error_handle = self.error_log.open("w", encoding="utf-8")
        self._stage_handle = self.stage_log.open("w", encoding="utf-8")
        self.log("=" * 70)
        self.log(f"EXPERIMENT: {experiment_id}")
        self.log(f"START TIME: {datetime.now().isoformat()}")
        self.log(f"LOG DIR: {self.log_dir}")
        self.log("=" * 70)

    def log(self, message: str, *, stage: str | None = None) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        print(formatted)
        self._main_handle.write(formatted + "\n")
        self._main_handle.flush()
        if stage is not None:
            self._stage_handle.write(f"[{timestamp}] [{stage}] {message}\n")
            self._stage_handle.flush()

    def log_error(self, message: str, exception: Exception | None = None) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] ERROR: {message}"
        if exception is not None:
            formatted += f"\n  Exception: {type(exception).__name__}: {exception}"
        print(formatted, file=sys.stderr)
        self._error_handle.write(formatted + "\n")
        self._error_handle.flush()

    def close(self, *, status: str) -> None:
        elapsed = time.time() - self.start_time
        self.log("=" * 70)
        self.log(f"END TIME: {datetime.now().isoformat()}")
        self.log(f"STATUS: {status}")
        self.log(f"ELAPSED: {elapsed / 3600:.2f} hours ({elapsed:.1f} seconds)")
        self.log("=" * 70)
        self._main_handle.close()
        self._error_handle.close()
        self._stage_handle.close()


class PreflightFailed(RuntimeError):
    """Raised when a config cannot run in the current environment."""


def _canonical_experiment_id(experiment_id: str) -> str:
    return EXPERIMENT_ALIASES.get(experiment_id, experiment_id)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _write_runtime_config(
    config_path: str,
    *,
    output_root: Path,
    override: dict[str, Any] | None = None,
) -> tuple[Path, str]:
    config = load_experiment_config(config_path)
    payload = copy.deepcopy(config.raw)
    paths_payload = payload.setdefault("paths", {})
    paths_payload["repo_root"] = str(config.repo_root())
    paths_payload["dataset_root"] = str(config.dataset_root())
    paths_payload["model_root"] = str(config.model_root())
    paths_payload["output_root"] = str(output_root)
    if override is not None:
        payload = _deep_merge(payload, override)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        runtime_path = Path(handle.name)
    return runtime_path, str(payload.get("meta", {}).get("experiment_id", config.experiment_id()))


def _expected_stage_names(config_path: str, *, with_glue: bool) -> list[str]:
    config = load_experiment_config(config_path)
    expected: list[str] = []
    if bool(config.stage1.get("enabled", True)):
        expected.append("stage1")
    if bool(config.bootstrap.get("enabled", True)):
        expected.append("stage2")
    if bool(config.eval_small.get("enabled", False)):
        expected.append("eval_small")
    if bool(config.eval_large.get("enabled", False)):
        expected.append("eval_large")
    if with_glue:
        expected.append("eval_glue")
    return expected


def _stage_summary_path(experiment_dir: Path, stage_name: str) -> Path:
    if stage_name == "eval_glue":
        return experiment_dir / "eval_glue" / "glue_summary.json"
    return experiment_dir / f"{stage_name}_summary.json"


def _all_expected_artifacts_exist(experiment_dir: Path, expected_stages: list[str]) -> bool:
    return all(_stage_summary_path(experiment_dir, stage_name).exists() for stage_name in expected_stages)


def check_experiment_completed(
    experiment_id: str,
    *,
    result_dir: Path,
    experiment_dir: Path,
    expected_stages: list[str],
) -> bool:
    result_file = result_dir / f"{experiment_id}.json"
    if result_file.exists():
        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
            if payload.get("status") == "SUCCESS" and _all_expected_artifacts_exist(experiment_dir, expected_stages):
                return True
        except (OSError, json.JSONDecodeError):
            pass

    metrics_file = experiment_dir / "metrics_summary.json"
    if metrics_file.exists():
        try:
            payload = json.loads(metrics_file.read_text(encoding="utf-8"))
            if payload.get("status") == "SUCCESS" and _all_expected_artifacts_exist(experiment_dir, expected_stages):
                return True
        except (OSError, json.JSONDecodeError):
            pass

    return False


def get_experiment_stage_status(experiment_dir: Path) -> dict[str, str]:
    stages = {}
    for stage_name in ["stage1", "stage2", "eval_small", "eval_large", "eval_glue"]:
        stages[stage_name] = "COMPLETED" if _stage_summary_path(experiment_dir, stage_name).exists() else "PENDING"
    return stages


def _load_stage_result(experiment_dir: Path, stage_name: str) -> dict[str, Any] | None:
    summary_path = _stage_summary_path(experiment_dir, stage_name)
    if not summary_path.exists():
        return None
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if stage_name == "eval_glue":
        return {"status": "COMPLETED", "summary": payload}
    return {
        "status": "COMPLETED",
        "artifacts": payload.get("artifacts", {}),
        "metrics": payload.get("metrics", {}),
        "message": payload.get("message", ""),
    }


def _collect_stage_results(experiment_dir: Path, expected_stages: list[str]) -> dict[str, Any]:
    return {
        stage_name: stage_result
        for stage_name in expected_stages
        if (stage_result := _load_stage_result(experiment_dir, stage_name)) is not None
    }


def _run_subprocess(cmd: list[str], logger: ExperimentLogger, *, stage: str) -> None:
    logger.log(f"Command: {' '.join(cmd)}", stage=stage)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(Path(__file__).parent.parent.parent),
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        logger.log(line.rstrip(), stage=stage)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{stage} failed with return code {proc.returncode}")


def run_single_experiment(
    experiment_id: str,
    experiment_config: dict[str, Any],
    *,
    with_glue: bool = False,
    output_base: Path | None = None,
    log_dir: Path | None = None,
    result_dir: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    output_base = Path(output_base or DEFAULT_OUTPUT_BASE)
    log_dir = Path(log_dir or DEFAULT_LOG_DIR)
    result_dir = Path(result_dir or DEFAULT_RESULT_DIR)

    canonical_id = _canonical_experiment_id(experiment_id)
    config_path = str(experiment_config["config"])
    expected_stages = _expected_stage_names(config_path, with_glue=with_glue)
    exp_output_dir = output_base / canonical_id
    exp_log_dir = log_dir / canonical_id
    result_file = result_dir / f"{canonical_id}.json"

    result: dict[str, Any] = {
        "experiment_id": canonical_id,
        "requested_id": experiment_id,
        "description": experiment_config["description"],
        "config": config_path,
        "status": "RUNNING",
        "start_time": datetime.now().isoformat(),
        "output_dir": str(exp_output_dir),
        "log_dir": str(exp_log_dir),
        "stages": {},
        "glue_results": None,
        "preflight": None,
    }

    logger = ExperimentLogger(canonical_id, exp_log_dir)
    logger.log(f"Requested experiment id: {experiment_id}")
    logger.log(f"Description: {experiment_config['description']}")
    logger.log(f"Config: {config_path}")
    logger.log(f"Resume mode: {resume}")
    logger.log(f"Expected stages: {expected_stages}")

    if resume and check_experiment_completed(
        canonical_id,
        result_dir=result_dir,
        experiment_dir=exp_output_dir,
        expected_stages=expected_stages,
    ):
        result["status"] = "SKIPPED"
        result["end_time"] = datetime.now().isoformat()
        result["message"] = "Already completed, skipped due to --resume"
        result["stages"] = _collect_stage_results(exp_output_dir, expected_stages)
        logger.log(f"Stage status: {get_experiment_stage_status(exp_output_dir)}")
        logger.close(status="SKIPPED")
        result_dir.mkdir(parents=True, exist_ok=True)
        result_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result

    output_base.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    runtime_config_path: Path | None = None
    glue_config_path: Path | None = None

    try:
        runtime_config_path, runtime_id = _write_runtime_config(config_path, output_root=output_base)
        if runtime_id != canonical_id:
            raise ValueError(
                f"Configured experiment_id '{runtime_id}' does not match requested id '{canonical_id}'."
            )

        preflight_report = run_preflight(runtime_config_path, with_glue=with_glue)
        result["preflight"] = preflight_report.to_dict()
        logger.log(format_preflight_report(preflight_report))
        if not preflight_report.ready:
            raise PreflightFailed("Preflight validation failed")

        pipeline_cmd = [
            sys.executable,
            "-m",
            "pretext_platform.scripts.run_pipeline",
            "--config",
            str(runtime_config_path),
        ]
        _run_subprocess(pipeline_cmd, logger, stage="PIPELINE")

        result["stages"] = _collect_stage_results(exp_output_dir, [s for s in expected_stages if s != "eval_glue"])

        if with_glue:
            glue_override = {
                "eval_glue": {
                    "enabled": True,
                    "tasks": DEFAULT_GLUE_TASKS,
                }
            }
            glue_config_path, _ = _write_runtime_config(
                config_path,
                output_root=output_base,
                override=glue_override,
            )
            glue_cmd = [
                sys.executable,
                "-m",
                "pretext_platform.scripts.run_glue",
                "--config",
                str(glue_config_path),
                "--tasks",
                "all",
            ]
            _run_subprocess(glue_cmd, logger, stage="GLUE")
            result["glue_results"] = _load_stage_result(exp_output_dir, "eval_glue")
            if result["glue_results"] is not None:
                result["stages"]["eval_glue"] = result["glue_results"]

        result["status"] = "SUCCESS"
        result["end_time"] = datetime.now().isoformat()
        result["total_elapsed_seconds"] = time.time() - logger.start_time
        logger.close(status="SUCCESS")
    except Exception as exc:
        result["status"] = "FAILED_PRECHECK" if isinstance(exc, PreflightFailed) else "FAILED"
        result["end_time"] = datetime.now().isoformat()
        result["total_elapsed_seconds"] = time.time() - logger.start_time
        result["stages"] = _collect_stage_results(exp_output_dir, expected_stages)
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        logger.log_error(f"Experiment failed: {exc}", exc)
        logger.close(status=result["status"])
    finally:
        for path in (runtime_config_path, glue_config_path):
            if path is not None and path.exists():
                path.unlink(missing_ok=True)

    result_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _save_progress(
    result_dir: Path,
    *,
    total: int,
    succeeded: list[str],
    failed: list[str],
    skipped: list[str],
    results: dict[str, Any],
) -> None:
    payload = {
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "in_progress": [],
        "results": results,
    }
    (result_dir / "progress.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_experiment_worker(args: tuple[Any, ...]) -> dict[str, Any]:
    return run_single_experiment(*args[:2], with_glue=args[2], output_base=args[3], log_dir=args[4], result_dir=args[5], resume=args[6])


def run_all_experiments(
    experiments: dict[str, dict[str, Any]],
    *,
    with_glue: bool = False,
    parallel: bool = False,
    max_parallel: int = 2,
    output_base: Path | None = None,
    log_dir: Path | None = None,
    result_dir: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    output_base = Path(output_base or DEFAULT_OUTPUT_BASE)
    log_dir = Path(log_dir or DEFAULT_LOG_DIR)
    result_dir = Path(result_dir or DEFAULT_RESULT_DIR)
    output_base.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    succeeded: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []

    experiment_list = {
        exp_id: {
            "description": config["description"],
            "config": config["config"],
            "suite": config["suite"],
            "tags": config.get("tags", []),
        }
        for exp_id, config in experiments.items()
    }
    (result_dir / "experiment_list.json").write_text(
        json.dumps(experiment_list, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 70)
    print(f"RUNNING {len(experiments)} PrE-Text EXPERIMENT CONFIGS")
    print(f"OUTPUT BASE: {output_base}")
    print(f"LOG DIR: {log_dir}")
    print(f"RESULT DIR: {result_dir}")
    print(f"PARALLEL: {parallel} (max {max_parallel})")
    print(f"RESUME: {resume}")
    print(f"GLUE EVAL: {with_glue}")
    print("=" * 70)

    exp_items = list(experiments.items())
    if parallel and max_parallel > 1:
        with ProcessPoolExecutor(max_workers=max_parallel) as executor:
            future_to_exp = {}
            for exp_id, exp_config in exp_items:
                args = (exp_id, exp_config, with_glue, output_base, log_dir, result_dir, resume)
                future = executor.submit(_run_experiment_worker, args)
                future_to_exp[future] = exp_id

            for future in as_completed(future_to_exp):
                exp_id = future_to_exp[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "experiment_id": exp_id,
                        "status": "FAILED",
                        "error": {"type": "ExecutorException", "message": str(exc)},
                    }
                results[result["experiment_id"]] = result
                if result["status"] == "SUCCESS":
                    succeeded.append(result["experiment_id"])
                elif result["status"] == "SKIPPED":
                    skipped.append(result["experiment_id"])
                else:
                    failed.append(result["experiment_id"])
                _save_progress(
                    result_dir,
                    total=len(experiments),
                    succeeded=succeeded,
                    failed=failed,
                    skipped=skipped,
                    results=results,
                )
    else:
        for index, (exp_id, exp_config) in enumerate(exp_items, start=1):
            print(f"\n{'=' * 70}")
            print(f"EXPERIMENT {index}/{len(exp_items)}: {exp_id}")
            print(f"{'=' * 70}")
            result = run_single_experiment(
                exp_id,
                exp_config,
                with_glue=with_glue,
                output_base=output_base,
                log_dir=log_dir,
                result_dir=result_dir,
                resume=resume,
            )
            results[result["experiment_id"]] = result
            if result["status"] == "SUCCESS":
                succeeded.append(result["experiment_id"])
            elif result["status"] == "SKIPPED":
                skipped.append(result["experiment_id"])
            else:
                failed.append(result["experiment_id"])
            _save_progress(
                result_dir,
                total=len(experiments),
                succeeded=succeeded,
                failed=failed,
                skipped=skipped,
                results=results,
            )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total": len(experiments),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "skipped": len(skipped),
        "succeeded_experiments": succeeded,
        "failed_experiments": failed,
        "skipped_experiments": skipped,
        "parallel": parallel,
        "max_parallel": max_parallel,
        "with_glue": with_glue,
        "resume": resume,
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def list_experiments() -> None:
    print("\nAvailable runnable PrE-Text experiment configs:")
    print("=" * 70)
    for suite_name, experiment_ids in EXPERIMENT_SUITES.items():
        print(f"\n[{suite_name}]")
        for exp_id in experiment_ids:
            config = PRETEXT_EXPERIMENTS[exp_id]
            tags = f" ({', '.join(config.get('tags', []))})" if config.get("tags") else ""
            print(f"  {exp_id}{tags}")
            print(f"    {config['description']}")
            print(f"    {config['config']}")


def validate_experiments(
    experiments: dict[str, dict[str, Any]],
    *,
    with_glue: bool,
    output_base: Path,
    result_dir: Path,
) -> dict[str, Any]:
    output_base.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    reports: dict[str, Any] = {}
    ready: list[str] = []
    blocked: list[str] = []

    print("=" * 70)
    print(f"VALIDATING {len(experiments)} PrE-Text EXPERIMENT CONFIGS")
    print(f"OUTPUT BASE: {output_base}")
    print(f"RESULT DIR: {result_dir}")
    print(f"GLUE EVAL: {with_glue}")
    print("=" * 70)

    for exp_id, exp_config in experiments.items():
        canonical_id = _canonical_experiment_id(exp_id)
        runtime_config_path: Path | None = None
        try:
            runtime_config_path, runtime_id = _write_runtime_config(exp_config["config"], output_root=output_base)
            if runtime_id != canonical_id:
                raise ValueError(
                    f"Configured experiment_id '{runtime_id}' does not match requested id '{canonical_id}'."
                )
            report = run_preflight(runtime_config_path, with_glue=with_glue)
            payload = report.to_dict()
            payload["description"] = exp_config["description"]
            payload["config"] = exp_config["config"]
            reports[canonical_id] = payload
            if report.ready:
                ready.append(canonical_id)
                print(f"[READY]   {canonical_id}")
            else:
                blocked.append(canonical_id)
                print(f"[BLOCKED] {canonical_id}")
                print(format_preflight_report(report))
        except Exception as exc:
            blocked.append(canonical_id)
            reports[canonical_id] = {
                "experiment_id": canonical_id,
                "config": exp_config["config"],
                "description": exp_config["description"],
                "ready": False,
                "enabled_stages": [],
                "errors": [{"severity": "error", "category": "config", "message": str(exc)}],
                "warnings": [],
            }
            print(f"[BLOCKED] {canonical_id}")
            print(f"  config error: {exc}")
        finally:
            if runtime_config_path is not None and runtime_config_path.exists():
                runtime_config_path.unlink(missing_ok=True)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total": len(experiments),
        "ready": len(ready),
        "blocked": len(blocked),
        "ready_experiments": ready,
        "blocked_experiments": blocked,
        "reports": reports,
    }
    (result_dir / "preflight_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def resolve_requested_experiments(
    *,
    run_all: bool,
    requested_experiments: list[str] | None,
    requested_suites: list[str] | None,
) -> dict[str, dict[str, Any]]:
    if run_all:
        return dict(PRETEXT_EXPERIMENTS)

    ordered: dict[str, dict[str, Any]] = {}
    if requested_suites:
        for suite_name in requested_suites:
            for exp_id in EXPERIMENT_SUITES[suite_name]:
                ordered[exp_id] = PRETEXT_EXPERIMENTS[exp_id]

    if requested_experiments:
        for raw_exp_id in requested_experiments:
            exp_id = _canonical_experiment_id(raw_exp_id)
            if exp_id not in PRETEXT_EXPERIMENTS:
                raise ValueError(f"Unknown experiment '{raw_exp_id}'. Use --list to inspect valid ids.")
            ordered[exp_id] = PRETEXT_EXPERIMENTS[exp_id]

    if not ordered:
        raise ValueError("Must specify --all, --suite, or --experiment.")

    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the executable PrE-Text experiment configs with logging and result tracking."
    )
    parser.add_argument("--list", action="store_true", help="List available experiment ids and exit.")
    parser.add_argument("--all", action="store_true", help="Run all executable PrE-Text configs.")
    parser.add_argument(
        "--suite",
        nargs="+",
        choices=sorted(EXPERIMENT_SUITES),
        help="Run one or more experiment suites (jobs/forums/microblog/congressional).",
    )
    parser.add_argument(
        "--experiment",
        nargs="+",
        help="Run one or more specific experiment ids. Old GC*/GP* aliases are also accepted.",
    )
    parser.add_argument("--with-glue", action="store_true", help="Run GLUE evaluation after the main pipeline.")
    parser.add_argument(
        "--parallel",
        nargs="?",
        const=2,
        default=1,
        type=int,
        help="Run experiments in parallel. Use '--parallel' for 2 workers or '--parallel N'.",
    )
    parser.add_argument("--output-base", type=str, default=None, help="Base output directory.")
    parser.add_argument("--log-dir", type=str, default=None, help="Directory for log files.")
    parser.add_argument("--result-dir", type=str, default=None, help="Directory for result JSON files.")
    parser.add_argument("--resume", action="store_true", help="Skip experiments that already completed successfully.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run preflight validation for the selected configs and exit without executing them.",
    )

    args = parser.parse_args()

    if args.list:
        list_experiments()
        return

    try:
        experiments = resolve_requested_experiments(
            run_all=args.all,
            requested_experiments=args.experiment,
            requested_suites=args.suite,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    max_parallel = max(1, int(args.parallel))
    output_base = Path(args.output_base) if args.output_base else DEFAULT_OUTPUT_BASE
    log_dir = Path(args.log_dir) if args.log_dir else DEFAULT_LOG_DIR
    result_dir = Path(args.result_dir) if args.result_dir else DEFAULT_RESULT_DIR

    if args.validate_only:
        summary = validate_experiments(
            experiments,
            with_glue=args.with_glue,
            output_base=output_base,
            result_dir=result_dir,
        )
        sys.exit(0 if summary["blocked"] == 0 else 1)

    run_all_experiments(
        experiments,
        with_glue=args.with_glue,
        parallel=max_parallel > 1,
        max_parallel=max_parallel,
        output_base=output_base,
        log_dir=log_dir,
        result_dir=result_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
