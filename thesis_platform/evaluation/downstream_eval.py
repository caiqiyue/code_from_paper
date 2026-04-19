from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Any

from thesis_platform.core.artifact_manifest import ARTIFACT_SCHEMA_VERSION
from thesis_platform.core.io_utils import ensure_dir, read_json, to_jsonable, write_json

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _clear_gpu_memory() -> None:
    """Explicitly clear GPU memory between stages to avoid OOM."""
    if not _TORCH_AVAILABLE:
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()


def resolve_large_eval_mode(downstream_cfg: dict[str, Any], *, platform_name: str | None = None) -> str:
    """Resolve the large-eval backend mode for the active platform."""

    platform_name = (platform_name or sys.platform).lower()
    requested = str(downstream_cfg.get("large_eval_mode", "auto")).strip().lower()
    if requested and requested != "auto":
        return requested
    if platform_name.startswith("win"):
        return str(downstream_cfg.get("windows_large_eval_mode", "peft_lora")).strip().lower()
    return str(downstream_cfg.get("linux_large_eval_mode", "peft_lora")).strip().lower()


def resolve_small_eval_mode(downstream_cfg: dict[str, Any], *, platform_name: str | None = None) -> str:
    """Resolve the small-eval backend mode for the active platform."""

    platform_name = (platform_name or sys.platform).lower()
    requested = str(downstream_cfg.get("small_eval_mode", "auto")).strip().lower()
    if requested and requested != "auto":
        return requested
    checkpoint = downstream_cfg.get("c4_checkpoint_path")
    if platform_name.startswith("win"):
        return str(downstream_cfg.get("windows_small_eval_mode", "gpt2")).strip().lower()
    if checkpoint not in (None, ""):
        return str(downstream_cfg.get("linux_small_eval_mode", "distilgpt2")).strip().lower()
    return "gpt2"


def export_synthetic_corpus(
    synthetic_texts: list[str],
    *,
    output_dir: Path,
    filename: str = "llama7b_text_syn.json",
) -> Path:
    """Write the final synthetic corpus in the format expected by downstream evaluation."""

    output_dir = ensure_dir(output_dir)
    corpus_path = output_dir / filename
    deduped: list[str] = []
    seen: set[str] = set()
    for text in synthetic_texts:
        cleaned = str(text).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    corpus_path.write_text(json.dumps(deduped, ensure_ascii=False, indent=2), encoding="utf-8")
    return corpus_path


def _ensure_pretext_import(repo_root: Path) -> None:
    # Strategy: Use repo_root to reliably locate Pre-Text sibling.
    # repo_root points to caiqiyue_file (project root containing both thesis_platform and Pre-Text).
    # thesis_platform is a subdir of repo_root, so Pre-Text is at repo_root / "PrE-Text".
    import os
    repo_root_resolved = repo_root.resolve()

    # Primary approach: Pre-Text is a sibling of thesis_platform at repo_root level
    # repo_root is like: /path/to/caiqiyue_file/thesis_platform
    # So Pre-Text is at: /path/to/caiqiyue_file/PrE-Text
    candidate_paths = [
        repo_root_resolved / "PrE-Text",           # repo_root/PrE-Text
        repo_root_resolved / "Pre-Text",            # repo_root/Pre-Text (alternative naming)
    ]
    # Also try going up from repo_root (in case repo_root includes thesis_platform subdir)
    for _ in range(5):
        for cp in candidate_paths:
            if cp.is_dir():
                pretext_root = str(cp.resolve())
                if pretext_root not in sys.path:
                    sys.path.insert(0, pretext_root)
                return
        candidate_parent = repo_root_resolved.parent
        if candidate_parent == repo_root_resolved:
            break
        repo_root_resolved = candidate_parent
        candidate_paths = [
            repo_root_resolved / "PrE-Text",
            repo_root_resolved / "Pre-Text",
        ]

    # Last resort: walk up from cwd
    cwd = os.getcwd()
    candidate = Path(cwd)
    for _ in range(10):
        for cp in [candidate / "PrE-Text", candidate / "Pre-Text"]:
            if cp.is_dir():
                pretext_root = str(cp.resolve())
                if pretext_root not in sys.path:
                    sys.path.insert(0, pretext_root)
                return
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent


def _build_pretext_raw(
    thesis_config,
    *,
    output_dir: Path,
    enable_large_eval: bool,
    enable_small_eval: bool,
) -> dict[str, Any]:
    downstream_cfg = thesis_config.downstream_eval
    repo_root = thesis_config.repo_root()
    large_eval_mode = resolve_large_eval_mode(downstream_cfg)
    small_eval_mode = resolve_small_eval_mode(downstream_cfg)
    return {
        "meta": {
            "experiment_id": f"{thesis_config.meta.get('experiment_id', 'experiment')}_pretext_eval",
            "seed": int(thesis_config.meta.get("seed", 42)),
        },
        "paths": {
            "repo_root": str(repo_root),
            "output_root": str(output_dir),
            "dataset_root": str(
                thesis_config.resolve_path(downstream_cfg.get("dataset_root", "thesis_platform/datasets"))
            ),
            "model_root": str(
                thesis_config.resolve_path(downstream_cfg.get("model_root", "thesis_platform/open_model"))
            ),
        },
        "data": {
            "dataset_name": str(thesis_config.data.get("dataset_name", "jobs")),
            "train_path": str(
                thesis_config.resolve_path(
                    downstream_cfg.get("train_path", thesis_config.data.get("train_path", ""))
                )
            ),
            "eval_path": str(
                thesis_config.resolve_path(
                    downstream_cfg.get("eval_path", thesis_config.data.get("eval_path", ""))
                )
            ),
            "initialization_path": str(
                thesis_config.resolve_path(
                    downstream_cfg.get(
                        "initialization_path",
                        thesis_config.data.get(
                            "initialization_path",
                            "thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json",
                        ),
                    )
                )
            ),
            "max_samples_per_client": int(thesis_config.data.get("max_samples_per_client", 8)),
            "initialization_min_words": int(thesis_config.data.get("initialization_min_words", 20)),
        },
        "models": {
            "minilm_path": str(
                thesis_config.resolve_path(downstream_cfg.get("minilm_path", "thesis_platform/open_model/all_minilm_l6_v2"))
            ),
            "roberta_large_path": str(
                thesis_config.resolve_path(
                    downstream_cfg.get("roberta_large_path", "thesis_platform/open_model/roberta_large")
                )
            ),
            "llama2_7b_path": str(
                thesis_config.resolve_path(downstream_cfg.get("llama2_7b_path", "thesis_platform/open_model/llama_2_7b_hf"))
            ),
            "distilgpt2_path": str(
                thesis_config.resolve_path(downstream_cfg.get("distilgpt2_path", "thesis_platform/open_model/distilgpt2"))
            ),
            "c4_checkpoint_path": str(
                thesis_config.resolve_path(downstream_cfg["c4_checkpoint_path"])
                if downstream_cfg.get("c4_checkpoint_path")
                else ""
            ),
        },
        "stage1": {"enabled": False, "rounds": 1},
        "bootstrap": {"enabled": False},
        "eval_small": {
            "enabled": enable_small_eval,
            "eval_mode": small_eval_mode,
            "cutoff_len": int(downstream_cfg.get("small_cutoff_len", downstream_cfg.get("cutoff_len", 64))),
            "grad_accum_steps": int(downstream_cfg.get("small_grad_accum_steps", 64)),
            "epochs": int(downstream_cfg.get("small_epochs", 20)),
            "batch_size": int(downstream_cfg.get("small_batch_size", 256)),
            "eval_batch_size": int(downstream_cfg.get("small_eval_batch_size", 8)),
            "learning_rate": float(downstream_cfg.get("small_learning_rate", 0.0002)),
            "num_proc": int(downstream_cfg.get("small_num_proc", 1)),
        },
        "eval_large": {
            "enabled": enable_large_eval,
            "eval_mode": large_eval_mode,
            "cutoff_len": int(downstream_cfg.get("cutoff_len", 64)),
            "grad_accum_steps": int(downstream_cfg.get("grad_accum_steps", 16)),
            "epochs": int(downstream_cfg.get("epochs", 1)),
            "batch_size": int(downstream_cfg.get("batch_size", 8)),
            "eval_batch_size": int(downstream_cfg.get("eval_batch_size", 2)),
            "learning_rate": float(downstream_cfg.get("learning_rate", 0.0002)),
            "num_proc": int(downstream_cfg.get("num_proc", 1)),
            "lora_rank": int(downstream_cfg.get("lora_rank", 4)),
            "lora_alpha": int(downstream_cfg.get("lora_alpha", 8)),
            "lora_dropout": float(downstream_cfg.get("lora_dropout", 0.0)),
        },
        "runtime": {
            "device": str(thesis_config.runtime.get("device", "cuda")),
        },
    }


def _build_pretext_config(thesis_config, *, output_dir: Path, enable_large_eval: bool, enable_small_eval: bool):
    repo_root = thesis_config.repo_root()
    _ensure_pretext_import(repo_root)

    from pretext_platform.core.config import ExperimentConfig as PretextExperimentConfig

    return PretextExperimentConfig.from_mapping(
        _build_pretext_raw(
            thesis_config,
            output_dir=output_dir,
            enable_large_eval=enable_large_eval,
            enable_small_eval=enable_small_eval,
        ),
        base_dir=repo_root,
        name="thesis_v3_pretext_eval.yaml",
    )


def run_pretext_large_eval(thesis_config, *, stage2_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Run the PrE-Text large-model downstream evaluation in-process."""

    pretext_config = _build_pretext_config(
        thesis_config,
        output_dir=output_dir,
        enable_large_eval=True,
        enable_small_eval=False,
    )

    from pretext_platform.core.models import resolve_model_paths
    from pretext_platform.data.loaders import load_dataset_bundle
    from pretext_platform.evaluation.llama2_eval import run_llama2_eval

    dataset_bundle = load_dataset_bundle(pretext_config)
    model_paths = resolve_model_paths(pretext_config)
    eval_mode = str(pretext_config.eval_large.get("eval_mode", "peft_lora")).strip().lower()
    if eval_mode != "peft_lora":
        raise ValueError("downstream large eval only supports peft_lora on the fixed Linux server.")
    summary = run_llama2_eval(pretext_config, dataset_bundle, model_paths, stage2_dir, output_dir)
    return to_jsonable(summary)


def run_pretext_small_eval(thesis_config, *, stage2_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Run the PrE-Text small-model downstream evaluation in-process."""

    pretext_config = _build_pretext_config(
        thesis_config,
        output_dir=output_dir,
        enable_large_eval=False,
        enable_small_eval=True,
    )

    from pretext_platform.core.models import resolve_model_paths
    from pretext_platform.data.loaders import load_dataset_bundle
    from pretext_platform.evaluation.distilgpt2_eval import run_distilgpt2_eval
    from pretext_platform.evaluation.gpt2_eval import run_gpt2_eval

    dataset_bundle = load_dataset_bundle(pretext_config)
    model_paths = resolve_model_paths(pretext_config)
    eval_mode = str(pretext_config.eval_small.get("eval_mode", "distilgpt2")).strip().lower()
    if eval_mode == "gpt2":
        summary = run_gpt2_eval(pretext_config, dataset_bundle, model_paths, stage2_dir, output_dir)
    else:
        summary = run_distilgpt2_eval(pretext_config, dataset_bundle, model_paths, stage2_dir, output_dir)
    return to_jsonable(summary)


def run_pretext_glue_eval(
    thesis_config,
    *,
    stage2_dir: Path,
    output_dir: Path,
    tasks: list[str],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run GLUE-style classification evaluation on one exported synthetic corpus."""

    pretext_config = _build_pretext_config(
        thesis_config,
        output_dir=output_dir,
        enable_large_eval=False,
        enable_small_eval=False,
    )
    eval_glue_cfg = {
        "enabled": True,
        "tasks": list(tasks),
    }
    if overrides:
        eval_glue_cfg.update(dict(overrides))
    pretext_config.raw["eval_glue"] = eval_glue_cfg

    from pretext_platform.core.models import resolve_model_paths
    from pretext_platform.evaluation.glue_classification_eval import run_glue_classification_eval

    model_paths = resolve_model_paths(pretext_config)
    task_summaries: dict[str, Any] = {}
    for task in tasks:
        summary = run_glue_classification_eval(
            config=pretext_config,
            model_paths=model_paths,
            stage2_dir=stage2_dir,
            output_dir=output_dir,
            task_name=task,
        )
        stage_summary = to_jsonable(summary)
        task_summaries[task] = {
            "stage_name": stage_summary.get("stage_name", f"glue_eval_{task}"),
            "output_dir": stage_summary.get("output_dir", str(output_dir)),
            "artifacts": stage_summary.get("artifacts", {}),
            "metrics": stage_summary.get("metrics", {}),
            "message": stage_summary.get("message", ""),
            "skipped": bool(stage_summary.get("skipped", False)),
        }
        write_json(output_dir / f"glue_{task}_summary.json", task_summaries[task])

    summary_payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "glue_eval_summary",
        "experiment_id": str(thesis_config.meta.get("experiment_id", "experiment")),
        "stage2_dir": str(stage2_dir),
        "output_dir": str(output_dir),
        "tasks": task_summaries,
    }
    write_json(ensure_dir(output_dir) / "glue_summary.json", summary_payload)
    return summary_payload


def collect_baseline_summaries(repo_root: Path, summary_paths: list[str], *, output_dir: Path) -> dict[str, Any]:
    """Collect existing baseline summary files into one normalized payload."""

    resolved: dict[str, Any] = {}
    for raw_path in summary_paths:
        path = (repo_root / raw_path).resolve()
        if not path.exists():
            resolved[raw_path] = {"missing": True}
            continue
        with path.open("r", encoding="utf-8") as handle:
            resolved[raw_path] = json.load(handle)
    write_json(ensure_dir(output_dir) / "baseline_summaries.json", resolved)
    return resolved


class DownstreamEvalManager:
    """Own the whole v3 downstream-eval lifecycle and stable reporting contract."""

    def __init__(self, thesis_config, *, experiment_id: str, output_dir: Path):
        self.thesis_config = thesis_config
        self.experiment_id = experiment_id
        self.output_dir = ensure_dir(output_dir)
        self.repo_root = thesis_config.repo_root()
        self.downstream_cfg = thesis_config.downstream_eval

    def _stage_summary_path(self, stage_key: str) -> Path:
        return self.output_dir / f"pretext_{stage_key}_summary.json"

    def _stage_disabled_payload(self, *, stage_key: str, stage_name: str, message: str) -> dict[str, Any]:
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": f"downstream_eval_{stage_key}_summary",
            "experiment_id": self.experiment_id,
            "stage_key": stage_key,
            "stage_name": stage_name,
            "enabled": False,
            "status": "disabled",
            "message": message,
            "summary_path": str(self._stage_summary_path(stage_key)),
            "metrics": {},
            "artifacts": {},
            "missing_assets": [],
        }

    def _missing_assets(self, required_paths: dict[str, Path | None]) -> list[dict[str, str]]:
        missing: list[dict[str, str]] = []
        for label, path in required_paths.items():
            if path is None or not path.exists():
                missing.append({"label": label, "path": str(path or Path("<unset>"))})
        return missing

    def _write_stage_payload(self, stage_key: str, payload: dict[str, Any]) -> dict[str, Any]:
        path = self._stage_summary_path(stage_key)
        payload["summary_path"] = str(path)
        write_json(path, payload)
        return payload

    def _reuse_existing_stage_payload(self, stage_key: str) -> dict[str, Any] | None:
        path = self._stage_summary_path(stage_key)
        if not path.exists():
            return None
        try:
            payload = dict(read_json(path))
        except Exception:
            return None
        status = str(payload.get("status", "")).strip().lower()
        reusable_statuses = {"completed", "disabled", "blocked", "completed_with_blocked_stages"}
        if status in reusable_statuses or status.startswith("blocked_"):
            return payload
        return None

    def _stage_blocked_payload(
        self,
        *,
        stage_key: str,
        stage_name: str,
        status: str,
        message: str,
        output_dir: Path,
        missing_assets: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": f"downstream_eval_{stage_key}_summary",
            "experiment_id": self.experiment_id,
            "stage_key": stage_key,
            "stage_name": stage_name,
            "enabled": True,
            "status": status,
            "message": message,
            "output_dir": str(output_dir),
            "metrics": {},
            "artifacts": {},
            "missing_assets": missing_assets or [],
        }

    def _run_large_stage(self, *, stage2_dir: Path) -> dict[str, Any]:
        stage_key = "large_eval"
        stage_name = "eval_large"
        reusable = self._reuse_existing_stage_payload(stage_key)
        if reusable is not None:
            return reusable
        if not bool(self.downstream_cfg.get("run_large_eval", False)):
            return self._write_stage_payload(
                stage_key,
                self._stage_disabled_payload(
                    stage_key=stage_key,
                    stage_name=stage_name,
                    message="large_eval is disabled by downstream_eval.run_large_eval.",
                ),
            )

        output_dir = ensure_dir(self.output_dir / "pretext_large_eval")
        eval_mode = resolve_large_eval_mode(self.downstream_cfg)
        if (
            bool(self.downstream_cfg.get("guard_windows_llama2_large_eval", True))
            and sys.platform.startswith("win")
            and eval_mode == "peft_lora"
        ):
            return self._write_stage_payload(
                stage_key,
                self._stage_blocked_payload(
                    stage_key=stage_key,
                    stage_name=stage_name,
                    status="blocked_unsupported_platform",
                    message=(
                        "large_eval is blocked on Windows because loading the local LLaMA2-7B checkpoint "
                        "is known to terminate the process with an access violation in this environment. "
                        "Run downstream large eval on Linux/WSL or disable downstream_eval.run_large_eval."
                    ),
                    output_dir=output_dir,
                ),
            )

        missing_assets = self._missing_assets(
            self._required_large_eval_assets(eval_mode)
        )
        if missing_assets:
            return self._write_stage_payload(
                stage_key,
                self._stage_blocked_payload(
                    stage_key=stage_key,
                    stage_name=stage_name,
                    status="blocked_missing_asset",
                    message="large_eval cannot run because required assets are missing.",
                    output_dir=output_dir,
                    missing_assets=missing_assets,
                ),
            )

        try:
            raw_summary = run_pretext_large_eval(self.thesis_config, stage2_dir=stage2_dir, output_dir=output_dir)
            return self._write_stage_payload(
                stage_key,
                {
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "artifact_type": f"downstream_eval_{stage_key}_summary",
                    "experiment_id": self.experiment_id,
                    "stage_key": stage_key,
                    "stage_name": raw_summary.get("stage_name", stage_name),
                    "enabled": True,
                    "status": "completed",
                    "message": raw_summary.get("message", ""),
                    "output_dir": str(output_dir),
                    "metrics": raw_summary.get("metrics", {}),
                    "artifacts": raw_summary.get("artifacts", {}),
                    "result": raw_summary,
                    "missing_assets": [],
                },
            )
        except Exception as exc:  # pragma: no cover - exercised through integration runs
            return self._write_stage_payload(
                stage_key,
                {
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "artifact_type": f"downstream_eval_{stage_key}_summary",
                    "experiment_id": self.experiment_id,
                    "stage_key": stage_key,
                    "stage_name": stage_name,
                    "enabled": True,
                    "status": "failed",
                    "message": str(exc),
                    "error_type": type(exc).__name__,
                    "output_dir": str(output_dir),
                    "metrics": {},
                    "artifacts": {},
                    "missing_assets": [],
                },
            )

    def _run_small_stage(self, *, stage2_dir: Path) -> dict[str, Any]:
        stage_key = "small_eval"
        stage_name = "eval_small"
        reusable = self._reuse_existing_stage_payload(stage_key)
        if reusable is not None:
            return reusable
        if not bool(self.downstream_cfg.get("run_small_eval", False)):
            return self._write_stage_payload(
                stage_key,
                self._stage_disabled_payload(
                    stage_key=stage_key,
                    stage_name=stage_name,
                    message="small_eval is disabled by downstream_eval.run_small_eval.",
                ),
            )

        missing_assets = self._missing_assets(
            self._required_small_eval_assets(resolve_small_eval_mode(self.downstream_cfg))
        )
        output_dir = ensure_dir(self.output_dir / "pretext_small_eval")
        if missing_assets:
            return self._write_stage_payload(
                stage_key,
                self._stage_blocked_payload(
                    stage_key=stage_key,
                    stage_name=stage_name,
                    status="blocked_missing_asset",
                    message="small_eval cannot run because required assets are missing.",
                    output_dir=output_dir,
                    missing_assets=missing_assets,
                ),
            )

        try:
            raw_summary = run_pretext_small_eval(self.thesis_config, stage2_dir=stage2_dir, output_dir=output_dir)
            return self._write_stage_payload(
                stage_key,
                {
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "artifact_type": f"downstream_eval_{stage_key}_summary",
                    "experiment_id": self.experiment_id,
                    "stage_key": stage_key,
                    "stage_name": raw_summary.get("stage_name", stage_name),
                    "enabled": True,
                    "status": "completed",
                    "message": raw_summary.get("message", ""),
                    "output_dir": str(output_dir),
                    "metrics": raw_summary.get("metrics", {}),
                    "artifacts": raw_summary.get("artifacts", {}),
                    "result": raw_summary,
                    "missing_assets": [],
                },
            )
        except Exception as exc:  # pragma: no cover - exercised through integration runs
            return self._write_stage_payload(
                stage_key,
                {
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "artifact_type": f"downstream_eval_{stage_key}_summary",
                    "experiment_id": self.experiment_id,
                    "stage_key": stage_key,
                    "stage_name": stage_name,
                    "enabled": True,
                    "status": "failed",
                    "message": str(exc),
                    "error_type": type(exc).__name__,
                    "output_dir": str(output_dir),
                    "metrics": {},
                    "artifacts": {},
                    "missing_assets": [],
                },
            )

    def _write_pretext_alias(self, corpus_path: Path) -> Path:
        alias_path = corpus_path.parent / "llama7b_text_syn.json"
        if alias_path != corpus_path:
            alias_path.write_text(corpus_path.read_text(encoding="utf-8"), encoding="utf-8")
        return alias_path

    def run(self, synthetic_texts: list[str]) -> dict[str, Any]:
        """Run downstream export, optional pretext eval stages, and baseline collection."""

        stage2_dir = ensure_dir(self.output_dir / "stage2")
        corpus_path = export_synthetic_corpus(
            synthetic_texts,
            output_dir=stage2_dir,
            filename=str(self.downstream_cfg.get("export_filename", "llama7b_text_syn.json")),
        )
        canonical_corpus_path = self._write_pretext_alias(corpus_path)
        large_stage = self._run_large_stage(stage2_dir=stage2_dir)
        # Clear GPU memory between stages to avoid OOM
        _clear_gpu_memory()
        small_stage = self._run_small_stage(stage2_dir=stage2_dir)
        baseline_paths = list(self.downstream_cfg.get("baseline_summary_paths", []))
        baseline_summaries = collect_baseline_summaries(self.repo_root, baseline_paths, output_dir=self.output_dir)
        baseline_summaries_path = self.output_dir / "baseline_summaries.json"

        stage_statuses = [large_stage.get("status"), small_stage.get("status")]
        active_stage_statuses = [status for status in stage_statuses if status != "disabled"]
        if not active_stage_statuses:
            overall_status = "disabled"
        elif "failed" in stage_statuses:
            overall_status = "failed"
        elif any(str(status).startswith("blocked_") for status in stage_statuses):
            if "completed" in stage_statuses:
                overall_status = "completed_with_blocked_stages"
            else:
                overall_status = "blocked"
        elif "blocked_missing_asset" in stage_statuses:
            overall_status = "completed_with_blocked_stages"
        else:
            overall_status = "completed"

        primary_stage = next(
            (
                stage
                for stage in (large_stage, small_stage)
                if stage.get("status") not in {"disabled", None}
            ),
            small_stage,
        )
        summary = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "downstream_eval_summary",
            "experiment_id": self.experiment_id,
            "enabled": bool(active_stage_statuses),
            "kind": self.downstream_cfg.get("kind", "none"),
            "status": overall_status,
            "stage_name": primary_stage.get("stage_name", ""),
            "metrics": primary_stage.get("metrics", {}),
            "synthetic_corpus_path": str(corpus_path),
            "canonical_synthetic_corpus_path": str(canonical_corpus_path),
            "stage2_dir": str(stage2_dir),
            "stages": {
                "large_eval": large_stage,
                "small_eval": small_stage,
            },
            "resolved_modes": {
                "large_eval_mode": resolve_large_eval_mode(self.downstream_cfg),
                "small_eval_mode": resolve_small_eval_mode(self.downstream_cfg),
            },
            "baseline_summaries": baseline_summaries,
            "baseline_summaries_path": str(baseline_summaries_path),
        }
        summary_path = self.output_dir / "downstream_eval_summary.json"
        summary["summary_path"] = str(summary_path)
        write_json(summary_path, summary)
        return summary

    def _required_large_eval_assets(self, eval_mode: str) -> dict[str, Path | None]:
        """Resolve the asset set required by the chosen large-eval mode."""

        assets: dict[str, Path | None] = {
            "model_root": self.thesis_config.resolve_path(
                self.downstream_cfg.get("model_root", "thesis_platform/open_model")
            ),
        }
        if eval_mode == "peft_lora":
            assets["llama2_7b_path"] = self.thesis_config.resolve_path(
                self.downstream_cfg.get("llama2_7b_path", "thesis_platform/open_model/llama_2_7b_hf")
            )
        else:
            raise ValueError(f"Unsupported large eval mode: {eval_mode}")
        return assets

    def _required_small_eval_assets(self, eval_mode: str) -> dict[str, Path | None]:
        """Resolve the asset set required by the chosen small-eval mode."""

        assets: dict[str, Path | None] = {
            "distilgpt2_path": self.thesis_config.resolve_path(
                self.downstream_cfg.get("distilgpt2_path", "thesis_platform/open_model/distilgpt2")
            ),
        }
        # c4_checkpoint_path is only needed for distilgpt2 mode, not for gpt2 mode
        if eval_mode != "gpt2":
            c4_path = self.downstream_cfg.get("c4_checkpoint_path")
            if c4_path:
                assets["c4_checkpoint_path"] = self.thesis_config.resolve_path(c4_path)
        return assets






