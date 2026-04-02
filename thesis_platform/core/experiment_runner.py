from __future__ import annotations

import copy
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import signal
import socket
import sys
import traceback
from typing import Any

import json

import thesis_platform.adapters  # Populate the adapter registry via import side effects.

from thesis_platform.core.artifact_manifest import (
    ARTIFACT_SCHEMA_VERSION,
    build_experiment_manifest,
    build_round_manifest,
)
from thesis_platform.core.checkpoint import CheckpointManager
from thesis_platform.core.config import ExperimentConfig
from thesis_platform.core.context import ClientContext, ServerContext
from thesis_platform.core.io_utils import ensure_dir, read_json, read_jsonl, write_json, write_text
from thesis_platform.core.logging_utils import (
    close_experiment_file_logger,
    get_logger,
    setup_experiment_file_logger,
)
from thesis_platform.core.preflight import validate_preflight
from thesis_platform.core.privacy import PrivacyLedger, PrivacyPolicy
from thesis_platform.core.registry import create
from thesis_platform.core.round_runner import RoundRunner
from thesis_platform.data.loaders import load_samples
from thesis_platform.data.partition import partition_samples
from thesis_platform.evaluation.downstream_eval import DownstreamEvalManager
from thesis_platform.models.backends import build_text_backend
from thesis_platform.models.embedding import build_embedder

try:
    from tqdm import tqdm
except (
    ModuleNotFoundError
):  # pragma: no cover - tqdm is declared in requirements but keep a fallback
    tqdm = None


class ExperimentRunner:
    """Top-level experiment orchestrator for one resolved experiment config."""

    def __init__(self, config: ExperimentConfig):
        """Prepare stable paths, logging, and output directories for one run."""

        self.config = config
        self.logger = get_logger()
        self.repo_root = config.repo_root()
        self.output_root = ensure_dir(config.output_root())
        self.experiment_id = str(config.meta.get("experiment_id", config.path.stem))
        # Append timestamp to experiment directory name for uniqueness
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._started_at = datetime.now(timezone.utc)
        self._stop_requested = False
        self._received_signal_name: str | None = None
        self._original_signal_handlers: dict[int, Any] = {}
        self.latest_run_path = self.output_root / "latest_run.json"
        self.latest_experiment_pointer_path = (
            self.output_root / f"{self.experiment_id}_latest.json"
        )
        self._set_experiment_dir(
            ensure_dir(self.output_root / f"{self.experiment_id}_{timestamp}")
        )
        self.logger.info("Experiment output directory: %s", self.experiment_dir)

    def _set_experiment_dir(self, experiment_dir: Path) -> None:
        """Bind all per-run artifact paths to one concrete experiment directory."""

        self.experiment_dir = ensure_dir(Path(experiment_dir))
        self.run_instance_id = self.experiment_dir.name
        self.run_registry_dir = ensure_dir(self.output_root / "run_registry" / self.experiment_id)
        self.run_registry_path = self.run_registry_dir / f"{self.run_instance_id}.json"
        self.resolved_config_path = self.experiment_dir / "resolved_config.json"
        self.metrics_summary_path = self.experiment_dir / "metrics_summary.json"
        self.privacy_ledger_path = self.experiment_dir / "privacy_ledger.json"
        self.artifact_manifest_path = self.experiment_dir / "artifact_manifest.json"
        self.failure_summary_path = self.experiment_dir / "failure_summary.json"
        self.run_state_path = self.experiment_dir / "run_state.json"
        self.config_snapshot_path = self.experiment_dir / "config.yaml"
        setup_experiment_file_logger(self.experiment_dir, name="thesis_platform")

    @staticmethod
    def _is_pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _resolve_resume_directory(self, resume_dir: str | Path | None = None) -> Path | None:
        """Resolve one safe resume directory for this experiment instance."""

        if resume_dir not in (None, ""):
            candidate = Path(str(resume_dir)).expanduser()
            if not candidate.exists():
                raise ValueError(f"resume_dir does not exist: {candidate}")
            return candidate

        candidates: list[dict[str, Any]] = []
        if self.run_registry_dir.exists():
            for pointer_path in sorted(self.run_registry_dir.glob('*.json')):
                try:
                    pointer = dict(read_json(pointer_path))
                except Exception:
                    continue
                experiment_dir_raw = str(pointer.get('experiment_dir', '')).strip()
                if not experiment_dir_raw:
                    continue
                experiment_dir = Path(experiment_dir_raw)
                if not experiment_dir.exists():
                    continue
                run_state_raw = str(pointer.get('run_state_path', '')).strip()
                run_state_path = Path(run_state_raw) if run_state_raw else experiment_dir / 'run_state.json'
                run_state = {}
                if run_state_path.exists():
                    try:
                        run_state = dict(read_json(run_state_path))
                    except Exception:
                        run_state = {}
                status = str(run_state.get('status', pointer.get('status', ''))).strip().lower()
                pid = int(run_state.get('pid') or pointer.get('pid') or 0)
                hostname = str(run_state.get('hostname') or pointer.get('hostname') or '')
                if status == 'completed':
                    continue
                if status == 'running':
                    if hostname and hostname != socket.gethostname():
                        continue
                    if self._is_pid_alive(pid):
                        continue
                candidates.append(
                    {
                        'experiment_dir': experiment_dir,
                        'updated_at': str(run_state.get('updated_at') or pointer.get('updated_at') or ''),
                    }
                )

        if not candidates and self.latest_experiment_pointer_path.exists():
            try:
                pointer = dict(read_json(self.latest_experiment_pointer_path))
                experiment_dir_raw = str(pointer.get('experiment_dir', '')).strip()
                if experiment_dir_raw:
                    experiment_dir = Path(experiment_dir_raw)
                    if experiment_dir.exists():
                        candidates.append(
                            {
                                'experiment_dir': experiment_dir,
                                'updated_at': str(pointer.get('updated_at', '')),
                            }
                        )
            except Exception:
                pass

        if not candidates:
            return None

        candidates.sort(key=lambda item: (item['updated_at'], str(item['experiment_dir'])))
        unique_dirs: list[Path] = []
        for item in candidates:
            experiment_dir = Path(item['experiment_dir'])
            if experiment_dir not in unique_dirs:
                unique_dirs.append(experiment_dir)
        if len(unique_dirs) > 1:
            joined = ', '.join(str(path) for path in unique_dirs)
            raise ValueError(
                'Multiple resumable runs found for the same experiment_id. '
                f'Pass --resume_dir explicitly. Candidates: {joined}'
            )
        return unique_dirs[0]

    def _activate_resume_directory_if_available(self, resume_dir: str | Path | None = None) -> None:
        """Reuse one safe run directory when resume=True."""

        resolved = self._resolve_resume_directory(resume_dir)
        if resolved is None:
            return
        if resolved.resolve() == self.experiment_dir.resolve():
            return
        self._set_experiment_dir(resolved)
        self.logger.info(
            "Experiment %s | resuming in existing output directory %s",
            self.experiment_id,
            self.experiment_dir,
        )

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _build_run_pointer_payload(self, *, status: str) -> dict[str, Any]:
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "latest_run_pointer",
            "experiment_id": self.experiment_id,
            "experiment_dir": str(self.experiment_dir),
            "status": status,
            "run_state_path": str(self.run_state_path),
            "metrics_summary_path": str(self.metrics_summary_path),
            "updated_at": self._now_iso(),
        }

    def _write_run_pointers(self, *, status: str) -> None:
        payload = self._build_run_pointer_payload(status=status)
        write_json(self.latest_run_path, payload)
        write_json(self.latest_experiment_pointer_path, payload)

    def _write_resolved_config_artifacts(self) -> None:
        write_json(
            self.resolved_config_path,
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "artifact_type": "resolved_config",
                "experiment_id": self.experiment_id,
                "config_path": str(self.config.path),
                "config": self.config.raw,
            },
        )
        write_text(
            self.config_snapshot_path,
            json.dumps(self.config.raw, ensure_ascii=False, indent=2),
        )

    def _write_privacy_ledger_snapshot(self, privacy_ledger: PrivacyLedger | None) -> None:
        if privacy_ledger is None:
            return
        write_json(
            self.privacy_ledger_path,
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "artifact_type": "privacy_ledger",
                "experiment_id": self.experiment_id,
                **privacy_ledger.report(),
            },
        )

    @staticmethod
    def _render_sample_payload(payload: dict[str, Any]) -> str:
        text = str(payload.get("text") or "").strip()
        if text:
            return text
        instruction = payload.get("instruction")
        response = payload.get("response")
        if instruction is not None and response is not None:
            return f"Instruction: {instruction}\nResponse: {response}".strip()
        if instruction is not None:
            return str(instruction)
        if response is not None:
            return str(response)
        return ""

    def _restore_synthetic_texts_for_downstream(self, *, last_completed_round: int | None) -> list[str]:
        export_filename = str(
            self.config.downstream_eval.get("export_filename", "llama7b_text_syn.json")
        )
        stage2_dir = self.experiment_dir / "downstream_eval" / "stage2"
        for candidate in (stage2_dir / export_filename, stage2_dir / "llama7b_text_syn.json"):
            if not candidate.exists():
                continue
            try:
                payload = read_json(candidate)
            except Exception:
                continue
            if isinstance(payload, list):
                texts = [str(item).strip() for item in payload if str(item).strip()]
                if texts:
                    return texts
        if last_completed_round is None:
            return []
        client_assigned_path = self.experiment_dir / f"round_{last_completed_round:03d}" / "client_assigned_samples.jsonl"
        if not client_assigned_path.exists():
            return []
        return [
            rendered
            for rendered in (
                self._render_sample_payload(dict(row)) for row in read_jsonl(client_assigned_path)
            )
            if rendered
        ]

    def _build_downstream_stub(self, *, enabled: bool, status: str, kind: str, message: str = "") -> dict[str, Any]:
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "downstream_eval_summary",
            "experiment_id": self.experiment_id,
            "enabled": enabled,
            "status": status,
            "kind": kind,
            "message": message,
        }

    def _build_metrics_summary(
        self,
        *,
        rounds_total: int,
        server_ctx: ServerContext | None,
        all_round_metrics: list[dict[str, Any]],
        privacy_ledger: PrivacyLedger | None,
        status: str,
        downstream_summary: dict[str, Any] | None,
        last_completed_round: int | None,
        resume_requested: bool,
    ) -> dict[str, Any]:
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "metrics_summary",
            "experiment_id": self.experiment_id,
            "status": status,
            "resume_requested": resume_requested,
            "experiment_dir": str(self.experiment_dir),
            "round_count": rounds_total,
            "completed_rounds": len(all_round_metrics),
            "last_completed_round": last_completed_round,
            "final_prompt": server_ctx.prompt_text if server_ctx is not None else "",
            "round_metrics": all_round_metrics,
            "privacy": privacy_ledger.summary() if privacy_ledger is not None else {},
            "artifacts": {
                "resolved_config_path": str(self.resolved_config_path),
                "privacy_ledger_path": str(self.privacy_ledger_path),
                "artifact_manifest_path": str(self.artifact_manifest_path),
                "run_state_path": str(self.run_state_path),
                "failure_summary_path": str(self.failure_summary_path),
            },
            "downstream_eval": downstream_summary
            or self._build_downstream_stub(
                enabled=False,
                status="disabled",
                kind=self.config.downstream_eval.get("kind", "none"),
            ),
            "updated_at": self._now_iso(),
        }

    def _write_artifact_manifest(
        self,
        *,
        round_manifests: list[dict[str, Any]],
        downstream_summary: dict[str, Any] | None,
    ) -> None:
        write_json(
            self.artifact_manifest_path,
            build_experiment_manifest(
                experiment_id=self.experiment_id,
                experiment_dir=self.experiment_dir,
                resolved_config_path=self.resolved_config_path,
                metrics_summary_path=self.metrics_summary_path,
                privacy_ledger_path=self.privacy_ledger_path,
                run_state_path=self.run_state_path,
                failure_summary_path=(
                    self.failure_summary_path if self.failure_summary_path.exists() else None
                ),
                round_manifests=round_manifests,
                downstream_summary=downstream_summary,
            ),
        )

    def _write_run_state(
        self,
        *,
        status: str,
        phase: str,
        rounds_total: int,
        completed_rounds: int,
        current_round: int | None,
        checkpoint_path: str | None = None,
        downstream_status: str | None = None,
        last_error: dict[str, Any] | None = None,
        resume_requested: bool = False,
    ) -> None:
        payload = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "run_state",
            "experiment_id": self.experiment_id,
            "experiment_dir": str(self.experiment_dir),
            "config_path": str(self.config.path),
            "status": status,
            "phase": phase,
            "resume_requested": resume_requested,
            "started_at": self._started_at.isoformat(),
            "updated_at": self._now_iso(),
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "current_round": current_round,
            "completed_rounds": completed_rounds,
            "rounds_total": rounds_total,
            "checkpoint_path": checkpoint_path,
            "downstream_status": downstream_status,
            "signal": self._received_signal_name,
            "stop_requested": self._stop_requested,
            "last_error": last_error,
            "artifacts": {
                "resolved_config_path": str(self.resolved_config_path),
                "metrics_summary_path": str(self.metrics_summary_path),
                "privacy_ledger_path": str(self.privacy_ledger_path),
                "artifact_manifest_path": str(self.artifact_manifest_path),
                "failure_summary_path": str(self.failure_summary_path),
            },
        }
        write_json(self.run_state_path, payload)
        self._write_run_pointers(status=status)

    def _write_failure_summary(
        self,
        *,
        status: str,
        phase: str,
        exc: BaseException,
        rounds_total: int,
        completed_rounds: int,
        current_round: int | None,
        checkpoint_path: str | None,
        resume_requested: bool,
    ) -> None:
        write_json(
            self.failure_summary_path,
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "artifact_type": "failure_summary",
                "experiment_id": self.experiment_id,
                "experiment_dir": str(self.experiment_dir),
                "status": status,
                "phase": phase,
                "resume_requested": resume_requested,
                "rounds_total": rounds_total,
                "completed_rounds": completed_rounds,
                "current_round": current_round,
                "checkpoint_path": checkpoint_path,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "signal": self._received_signal_name,
                "updated_at": self._now_iso(),
            },
        )

    def _handle_signal(self, signum, _frame) -> None:
        self._stop_requested = True
        signal_name = signal.Signals(signum).name
        self._received_signal_name = signal_name
        self.logger.warning(
            "Experiment %s | received %s | will stop after the current safe point",
            self.experiment_id,
            signal_name,
        )

    def _install_signal_handlers(self) -> None:
        for signum in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
            if signum is None:
                continue
            self._original_signal_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle_signal)

    def _restore_signal_handlers(self) -> None:
        for signum, handler in self._original_signal_handlers.items():
            signal.signal(signum, handler)
        self._original_signal_handlers.clear()

    def _build_text_backends(self) -> tuple[Any, Any]:
        """Build shared client/server text backends when configured."""

        llm_cfg = self.config.llm
        client_cfg = dict(llm_cfg.get("client", {}))
        server_cfg = dict(llm_cfg.get("server", {}))

        client_backend = (
            build_text_backend(
                {**client_cfg, "role": "client"}, repo_root=self.repo_root
            )
            if client_cfg
            else None
        )
        server_backend = (
            build_text_backend(
                {**server_cfg, "role": "server"}, repo_root=self.repo_root
            )
            if server_cfg
            else None
        )
        return client_backend, server_backend

    def _load_public_seed_samples(self) -> list[Any]:
        """Load the public seed pool used by the server-side generator."""

        public_seed_path = self.config.resolve_path(
            self.config.data.get("public_seed_path")
        )
        if public_seed_path is None:
            self.logger.info("No public seed dataset configured.")
            return []
        public_seed_sample_format = str(
            self.config.data.get(
                "public_seed_sample_format",
                self.config.data.get("sample_format", "raw_text"),
            )
        )
        samples = load_samples(
            public_seed_path,
            dataset_name=str(self.config.data.get("dataset_name", "dataset")),
            source="public_seed",
            task_type=str(self.config.data.get("task_type", "instruction_tuning")),
            round_id=0,
            client_id="server",
            prefix="seed",
            sample_format=public_seed_sample_format,
            limit=(
                int(self.config.data.get("max_public_seed_samples"))
                if self.config.data.get("max_public_seed_samples") not in (None, "")
                else None
            ),
        )
        self.logger.info(
            "Loaded %d public seed samples from %s", len(samples), public_seed_path
        )
        return samples

    def _load_client_contexts(self, *, client_backend: Any) -> list[ClientContext]:
        """Load the dataset and partition it into per-client runtime contexts."""

        train_path = self.config.resolve_path(self.config.data.get("train_path"))
        if train_path is None:
            raise ValueError("data.train_path must be configured.")
        sample_format = str(self.config.data.get("sample_format", "raw_text"))
        all_samples = load_samples(
            train_path,
            dataset_name=str(self.config.data.get("dataset_name", "dataset")),
            source="real",
            task_type=str(self.config.data.get("task_type", "instruction_tuning")),
            round_id=0,
            client_id="raw",
            prefix="real",
            sample_format=sample_format,
            limit=(
                int(self.config.data.get("train_limit"))
                if self.config.data.get("train_limit") not in (None, "")
                else None
            ),
        )
        self.logger.info(
            "Loaded %d private training samples from %s", len(all_samples), train_path
        )
        partitions = partition_samples(
            all_samples,
            num_clients=int(self.config.data.get("num_clients", 3)),
            max_samples_per_client=int(
                self.config.data.get("max_samples_per_client", 16)
            ),
            validation_ratio=float(self.config.data.get("validation_ratio", 0.1)),
            seed=int(self.config.meta.get("seed", 42)),
            strategy=str(
                self.config.data.get("partition_strategy", "shuffle_round_robin")
            ),
        )
        retriever_cfg = self.config.retriever
        embedder = build_embedder(
            retriever_cfg.get("embedding_model"),
            self.repo_root,
            allow_fallback=bool(retriever_cfg.get("allow_hashing_fallback", True)),
        )
        self.logger.info(
            "Partitioned dataset into %d clients with max %d samples/client and validation_ratio=%.2f",
            len(partitions),
            int(self.config.data.get("max_samples_per_client", 16)),
            float(self.config.data.get("validation_ratio", 0.1)),
        )
        self.logger.info(
            "Embedder backend: %s",
            getattr(embedder, "backend_name", type(embedder).__name__),
        )

        all_partition_samples = [
            sample for partition in partitions for sample in partition["all"]
        ]
        contexts: list[ClientContext] = []
        objective_type = str(
            self.config.scorer.get(
                "objective",
                "pair_alignment"
                if str(self.config.scorer.get("name", "")).lower() == "ira"
                else "domain_probe",
            )
        )
        for idx, bucket in enumerate(partitions):
            client_id = f"client_{idx}"
            bucket_sample_ids = {sample.sample_id for sample in bucket["all"]}
            negative_samples = [
                sample
                for sample in all_partition_samples
                if sample.sample_id not in bucket_sample_ids
            ]
            contexts.append(
                ClientContext(
                    client_id=client_id,
                    train_samples=bucket["train"],
                    validation_samples=bucket["validation"],
                    all_samples=bucket["all"],
                    embedder=embedder,
                    config=self.config.raw,
                    negative_samples=negative_samples,
                    text_backend=client_backend,
                    objective_type=objective_type,
                )
            )
        return contexts

    def _run_cross_domain_eval(
        self,
        *,
        synthetic_texts: list[str],
        cross_domain_cfg: dict[str, Any],
        current_round: int | None,
        rounds_total: int,
        checkpoint_path: str | None,
    ) -> dict[str, Any]:
        """Run cross-domain downstream evaluation for transfer learning experiments.

        This evaluates the synthetic corpus generated from the source domain
        on a target domain's downstream task to measure transfer learning capability.
        """
        from thesis_platform.evaluation.downstream_eval import export_synthetic_corpus

        self._write_run_state(
            status="running",
            phase="cross_domain_eval",
            rounds_total=rounds_total,
            completed_rounds=len(self._all_round_metrics) if hasattr(self, "_all_round_metrics") else 0,
            current_round=current_round,
            checkpoint_path=checkpoint_path,
            downstream_status="cross_domain_running",
            resume_requested=False,
        )

        target_dataset = str(cross_domain_cfg.get("target_dataset", "unknown"))
        target_train_path = self.config.resolve_path(cross_domain_cfg.get("target_train_path"))
        target_eval_path = self.config.resolve_path(cross_domain_cfg.get("target_eval_path"))

        if not target_train_path or not target_train_path.exists():
            self.logger.warning(
                "Cross-domain eval skipped: target_train_path not found at %s",
                target_train_path,
            )
            return {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "artifact_type": "cross_domain_eval_summary",
                "experiment_id": self.experiment_id,
                "enabled": True,
                "status": "skipped",
                "message": f"Target dataset '{target_dataset}' train path not found",
                "target_dataset": target_dataset,
            }

        cross_domain_root = ensure_dir(self.experiment_dir / "cross_domain_eval" / target_dataset)
        corpus_path = export_synthetic_corpus(
            synthetic_texts,
            output_dir=cross_domain_root,
            filename="synthetic_corpus.json",
        )

        self.logger.info(
            "Cross-domain eval: source=%s -> target=%s | corpus=%s",
            self.config.data.get("dataset_name"),
            target_dataset,
            corpus_path,
        )

        try:
            target_raw = copy.deepcopy(self.config.raw)
            target_raw.setdefault("meta", {})["experiment_id"] = f"{self.experiment_id}_xfer_{target_dataset}"
            target_raw.setdefault("data", {})["dataset_name"] = target_dataset
            target_raw["data"]["train_path"] = str(cross_domain_cfg.get("target_train_path", ""))
            if cross_domain_cfg.get("target_eval_path") not in (None, ""):
                target_raw["data"]["eval_path"] = str(cross_domain_cfg.get("target_eval_path"))

            target_downstream_cfg = dict(target_raw.get("downstream_eval", {}))
            target_downstream_cfg["enabled"] = True
            target_downstream_cfg["kind"] = str(cross_domain_cfg.get("kind", "pretext_large_eval"))
            target_downstream_cfg["run_large_eval"] = bool(cross_domain_cfg.get("run_large_eval", True))
            target_downstream_cfg["run_small_eval"] = bool(cross_domain_cfg.get("run_small_eval", False))
            for key in (
                "large_eval_mode",
                "windows_large_eval_mode",
                "linux_large_eval_mode",
                "small_eval_mode",
                "windows_small_eval_mode",
                "linux_small_eval_mode",
                "model_root",
                "distilgpt2_path",
                "gpt2_xl_path",
                "llama2_7b_path",
                "llama_3_2_3b_instruct_path",
                "c4_checkpoint_path",
                "batch_size",
                "eval_batch_size",
                "grad_accum_steps",
                "epochs",
                "learning_rate",
                "num_proc",
                "lora_rank",
                "lora_alpha",
                "lora_dropout",
                "small_batch_size",
                "small_eval_batch_size",
                "small_grad_accum_steps",
                "small_epochs",
                "small_learning_rate",
                "small_num_proc",
            ):
                if key in cross_domain_cfg:
                    target_downstream_cfg[key] = cross_domain_cfg[key]
            target_raw["downstream_eval"] = target_downstream_cfg

            target_config = ExperimentConfig(path=self.config.path, raw=target_raw)
            transfer_output_dir = ensure_dir(cross_domain_root / "downstream_eval")
            eval_results = DownstreamEvalManager(
                target_config,
                experiment_id=f"{self.experiment_id}_xfer_{target_dataset}",
                output_dir=transfer_output_dir,
            ).run(synthetic_texts)
            write_json(cross_domain_root / "cross_domain_results.json", eval_results)
            status = str(eval_results.get("status", "completed"))
            message = str(eval_results.get("message", ""))
            metrics = dict(eval_results.get("metrics", {}))
            stage2_dir = str(eval_results.get("stage2_dir", transfer_output_dir / "stage2"))
            eval_corpus_path = str(eval_results.get("synthetic_corpus_path", corpus_path))
        except Exception as exc:
            self.logger.warning("Cross-domain eval failed: %s", exc)
            eval_results = {}
            status = "failed"
            message = str(exc)
            metrics = {}
            stage2_dir = str(cross_domain_root / "downstream_eval" / "stage2")
            eval_corpus_path = str(corpus_path)

        summary = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "cross_domain_eval_summary",
            "experiment_id": self.experiment_id,
            "enabled": True,
            "status": status,
            "message": message,
            "source_dataset": self.config.data.get("dataset_name"),
            "target_dataset": target_dataset,
            "corpus_path": eval_corpus_path,
            "source_corpus_alias_path": str(corpus_path),
            "stage2_dir": stage2_dir,
            "target_train_path": str(target_train_path),
            "target_eval_path": str(target_eval_path) if target_eval_path else None,
            "metrics": metrics,
            "downstream_eval": eval_results,
            "updated_at": self._now_iso(),
        }

        write_json(cross_domain_root / "cross_domain_summary.json", summary)
        return summary

    def run(self, resume: bool = False, resume_dir: str | Path | None = None) -> dict[str, Any]:
        """Run the configured experiment end to end and return the summary payload.

        Args:
            resume: If True, attempt to resume from the latest checkpoint.`r`n            resume_dir: Explicit experiment output directory to resume when multiple candidates exist.

        Returns:
            Summary dictionary with experiment results.
        """
        rounds = int(self.config.federation.get("rounds", 1))
        downstream_cfg = self.config.downstream_eval
        checkpoint_mgr = CheckpointManager(
            output_dir=self.experiment_dir,
            max_checkpoints=3,
            save_artifacts=True,
        )
        all_round_metrics: list[dict[str, Any]] = []
        round_manifests: list[dict[str, Any]] = []
        current_round: int | None = None
        last_checkpoint_path: str | None = None
        downstream_summary: dict[str, Any] | None = None
        server_ctx: ServerContext | None = None
        privacy_ledger: PrivacyLedger | None = None
        progress = None

        if resume:
            self._activate_resume_directory_if_available(resume_dir)
        if self.failure_summary_path.exists():
            self.failure_summary_path.unlink()
        self._write_resolved_config_artifacts()
        self._write_artifact_manifest(
            round_manifests=round_manifests,
            downstream_summary=None,
        )
        write_json(
            self.metrics_summary_path,
            self._build_metrics_summary(
                rounds_total=rounds,
                server_ctx=None,
                all_round_metrics=all_round_metrics,
                privacy_ledger=None,
                status="initializing",
                downstream_summary=self._build_downstream_stub(
                    enabled=bool(downstream_cfg.get("enabled")),
                    status="pending" if bool(downstream_cfg.get("enabled")) else "disabled",
                    kind=downstream_cfg.get("kind", "none"),
                    message="Experiment has started and is preparing runtime dependencies.",
                ),
                last_completed_round=None,
                resume_requested=resume,
            ),
        )
        self._write_run_state(
            status="running",
            phase="initializing",
            rounds_total=rounds,
            completed_rounds=0,
            current_round=None,
            resume_requested=resume,
        )
        self._install_signal_handlers()

        try:
            from thesis_platform import adapters  # noqa: F401

            self.logger.info(
                "Experiment %s | starting with config %s",
                self.experiment_id,
                self.config.path,
            )
            validate_preflight(self.config)
            privacy_policy = PrivacyPolicy.from_config(self.config.privacy)
            privacy_ledger = PrivacyLedger(policy=privacy_policy)
            client_backend, server_backend = self._build_text_backends()
            public_seed_samples = self._load_public_seed_samples()
            client_contexts = self._load_client_contexts(client_backend=client_backend)

            start_round = 0
            checkpoint_data = None
            restored_experiment_state = None
            privacy_ledger_data = None
            if resume:
                checkpoint = checkpoint_mgr.load_checkpoint()
                if checkpoint:
                    start_round = checkpoint.get("round_id", -1) + 1
                    checkpoint_data = checkpoint.get("server_ctx_data")
                    restored_experiment_state = checkpoint.get("experiment_state", {})
                    privacy_ledger_data = checkpoint.get("privacy_ledger_data")
                    last_checkpoint_path = checkpoint.get("checkpoint_path")
                    self.logger.info(
                        "Resuming experiment %s from round %d",
                        self.experiment_id,
                        start_round,
                    )
                else:
                    self.logger.info("No checkpoint found, starting from round 0")

            if privacy_ledger_data is not None:
                privacy_ledger = PrivacyLedger.restore_from_report(privacy_ledger_data)
                self.logger.info("PrivacyLedger restored from checkpoint")
            else:
                privacy_ledger = PrivacyLedger(policy=privacy_policy)

            generator = create(
                "generator",
                str(self.config.generator.get("name")),
                self.config.generator,
                self.repo_root,
            )
            scorer = create(
                "scorer",
                str(self.config.scorer.get("name")),
                self.config.scorer,
                self.repo_root,
            )
            retriever = create(
                "retriever",
                str(self.config.retriever.get("name", "knn")),
                self.config.retriever,
                self.repo_root,
            )
            critic = create(
                "critic",
                str(self.config.critic.get("name", "none")),
                self.config.critic,
                self.repo_root,
            )
            aggregator = create(
                "aggregator",
                str(self.config.aggregator.get("name", "none")),
                self.config.aggregator,
                self.repo_root,
            )
            self.logger.info(
                "Experiment %s | adapters generator=%s scorer=%s retriever=%s critic=%s aggregator=%s",
                self.experiment_id,
                type(generator).__name__,
                type(scorer).__name__,
                type(retriever).__name__,
                type(critic).__name__,
                type(aggregator).__name__,
            )
            round_runner = RoundRunner(
                generator=generator,
                scorer=scorer,
                retriever=retriever,
                critic=critic,
                aggregator=aggregator,
            )

            initial_prompt = str(
                self.config.generator.get(
                    "initial_prompt",
                    f"Generate text consistent with dataset {self.config.data.get('dataset_name', 'dataset')}.",
                )
            )
            if checkpoint_data is not None:
                server_ctx = ServerContext.restore_from_checkpoint(
                    checkpoint_data=checkpoint_data,
                    config=self.config.raw,
                    output_dir=self.experiment_dir,
                    text_backend=server_backend,
                )
                self.logger.info("ServerContext restored from checkpoint")
            else:
                server_ctx = ServerContext(
                    experiment_id=self.experiment_id,
                    prompt_text=initial_prompt,
                    prompt_history=[initial_prompt],
                    config=self.config.raw,
                    output_dir=self.experiment_dir,
                    text_backend=server_backend,
                    base_prompt=initial_prompt,
                )

            if restored_experiment_state is not None:
                all_round_metrics = list(restored_experiment_state.get("round_metrics", []))
                round_manifests = list(restored_experiment_state.get("round_manifests", []))
                self.logger.info(
                    "Restored %d completed rounds from checkpoint",
                    len(all_round_metrics),
                )
            self._write_privacy_ledger_snapshot(privacy_ledger)
            running_downstream_summary = self._build_downstream_stub(
                enabled=bool(downstream_cfg.get("enabled")),
                status="pending" if bool(downstream_cfg.get("enabled")) else "disabled",
                kind=downstream_cfg.get("kind", "none"),
                message="Rounds are in progress; downstream evaluation has not started yet.",
            )
            write_json(
                self.metrics_summary_path,
                self._build_metrics_summary(
                    rounds_total=rounds,
                    server_ctx=server_ctx,
                    all_round_metrics=all_round_metrics,
                    privacy_ledger=privacy_ledger,
                    status="running",
                    downstream_summary=running_downstream_summary,
                    last_completed_round=all_round_metrics[-1]["round_id"] if all_round_metrics else None,
                    resume_requested=resume,
                ),
            )
            self._write_artifact_manifest(
                round_manifests=round_manifests,
                downstream_summary=running_downstream_summary,
            )
            self._write_run_state(
                status="running",
                phase="round_loop",
                rounds_total=rounds,
                completed_rounds=len(all_round_metrics),
                current_round=None,
                checkpoint_path=last_checkpoint_path,
                downstream_status=running_downstream_summary["status"],
                resume_requested=resume,
            )

            last_artifacts = None
            self.logger.info(
                "Experiment %s | executing rounds %d to %d (of %d total)",
                self.experiment_id,
                start_round,
                rounds - 1,
                rounds,
            )
            remaining_rounds = rounds - start_round
            progress = (
                tqdm(
                    range(remaining_rounds),
                    total=remaining_rounds,
                    desc=(
                        f"{self.experiment_id} (resumed from r{start_round})"
                        if start_round > 0
                        else f"{self.experiment_id}"
                    ),
                    unit="round",
                )
                if tqdm is not None
                else None
            )
            for round_id in range(start_round, rounds):
                if self._stop_requested:
                    break
                current_round = round_id
                self._write_run_state(
                    status="running",
                    phase="round_running",
                    rounds_total=rounds,
                    completed_rounds=len(all_round_metrics),
                    current_round=current_round,
                    checkpoint_path=last_checkpoint_path,
                    downstream_status=running_downstream_summary["status"],
                    resume_requested=resume,
                )
                self.logger.info("Round %d/%d | start", round_id + 1, rounds)
                round_dir = ensure_dir(self.experiment_dir / f"round_{round_id:03d}")
                artifacts = round_runner.run_round(
                    round_id=round_id,
                    server_ctx=server_ctx,
                    client_contexts=client_contexts,
                    public_seed_samples=public_seed_samples,
                    federation_cfg=self.config.federation,
                    output_dir=round_dir,
                    prototype_cfg=self.config.prototype,
                    routing_cfg=self.config.routing,
                    privacy_ledger=privacy_ledger,
                )
                last_artifacts = artifacts
                server_ctx.generated_history.append(
                    [sample.rendered_text() for sample in artifacts.generated_samples]
                )
                server_ctx.prompt_text = artifacts.updated_prompt
                server_ctx.prompt_history.append(server_ctx.prompt_text)
                all_round_metrics.append({"round_id": round_id, **artifacts.round_metrics})
                round_manifests.append(
                    build_round_manifest(
                        experiment_id=self.experiment_id,
                        round_id=round_id,
                        round_dir=round_dir,
                    )
                )
                if progress is not None:
                    progress.set_postfix(
                        generated=artifacts.round_metrics.get("generated_count", 0),
                        critiques=artifacts.round_metrics.get("critique_count", 0),
                    )
                    progress.update(1)
                self.logger.info(
                    "Round %d/%d | complete | generated=%s selected_bad=%d critiques=%d output=%s",
                    round_id + 1,
                    rounds,
                    artifacts.round_metrics.get("generated_count", 0),
                    len(artifacts.selected_bad_samples),
                    len(artifacts.critiques),
                    round_dir,
                )

                try:
                    last_checkpoint_path = str(
                        checkpoint_mgr.save_checkpoint(
                            round_id=round_id,
                            experiment_state={
                                "round_metrics": all_round_metrics,
                                "round_manifests": round_manifests,
                                "prompt_history": server_ctx.prompt_history,
                            },
                            server_ctx=server_ctx,
                            privacy_ledger=privacy_ledger,
                            config=self.config.raw,
                        )
                    )
                    self.logger.debug("Checkpoint saved for round %d", round_id)
                except Exception as exc:
                    self.logger.warning(
                        "Failed to save checkpoint for round %d: %s", round_id, exc
                    )

                self._write_privacy_ledger_snapshot(privacy_ledger)
                write_json(
                    self.metrics_summary_path,
                    self._build_metrics_summary(
                        rounds_total=rounds,
                        server_ctx=server_ctx,
                        all_round_metrics=all_round_metrics,
                        privacy_ledger=privacy_ledger,
                        status="running",
                        downstream_summary=running_downstream_summary,
                        last_completed_round=round_id,
                        resume_requested=resume,
                    ),
                )
                self._write_artifact_manifest(
                    round_manifests=round_manifests,
                    downstream_summary=running_downstream_summary,
                )
                self._write_run_state(
                    status="running",
                    phase="round_complete",
                    rounds_total=rounds,
                    completed_rounds=len(all_round_metrics),
                    current_round=round_id,
                    checkpoint_path=last_checkpoint_path,
                    downstream_status=running_downstream_summary["status"],
                    resume_requested=resume,
                )

            if progress is not None:
                progress.close()
                progress = None

            if self._stop_requested:
                interrupted_downstream = self._build_downstream_stub(
                    enabled=bool(downstream_cfg.get("enabled")),
                    status="skipped_due_to_interrupt" if bool(downstream_cfg.get("enabled")) else "disabled",
                    kind=downstream_cfg.get("kind", "none"),
                    message="Experiment stopped before downstream evaluation completed.",
                )
                self._write_privacy_ledger_snapshot(privacy_ledger)
                interrupted_summary = self._build_metrics_summary(
                    rounds_total=rounds,
                    server_ctx=server_ctx,
                    all_round_metrics=all_round_metrics,
                    privacy_ledger=privacy_ledger,
                    status="interrupted",
                    downstream_summary=interrupted_downstream,
                    last_completed_round=all_round_metrics[-1]["round_id"] if all_round_metrics else None,
                    resume_requested=resume,
                )
                write_json(self.metrics_summary_path, interrupted_summary)
                self._write_artifact_manifest(
                    round_manifests=round_manifests,
                    downstream_summary=interrupted_downstream,
                )
                self._write_run_state(
                    status="interrupted",
                    phase="interrupted",
                    rounds_total=rounds,
                    completed_rounds=len(all_round_metrics),
                    current_round=current_round,
                    checkpoint_path=last_checkpoint_path,
                    downstream_status=interrupted_downstream["status"],
                    resume_requested=resume,
                )
                self.logger.warning(
                    "Experiment %s | interrupted | completed_rounds=%d",
                    self.experiment_id,
                    len(all_round_metrics),
                )
                return interrupted_summary

            synthetic_texts: list[str] = []
            if bool(downstream_cfg.get("enabled")):
                if last_artifacts is not None:
                    synthetic_texts = [
                        sample.rendered_text()
                        for sample in last_artifacts.client_assigned_samples
                    ]
                else:
                    synthetic_texts = self._restore_synthetic_texts_for_downstream(
                        last_completed_round=all_round_metrics[-1]["round_id"] if all_round_metrics else None,
                    )
                if not synthetic_texts:
                    raise ValueError(
                        "Cannot resume downstream evaluation because no synthetic corpus could be restored from stage2 or the last completed round artifacts."
                    )
                self._write_run_state(
                    status="running",
                    phase="downstream_eval",
                    rounds_total=rounds,
                    completed_rounds=len(all_round_metrics),
                    current_round=current_round,
                    checkpoint_path=last_checkpoint_path,
                    downstream_status="running",
                    resume_requested=resume,
                )
                downstream_root = ensure_dir(self.experiment_dir / "downstream_eval")
                downstream_summary = DownstreamEvalManager(
                    self.config,
                    experiment_id=self.experiment_id,
                    output_dir=downstream_root,
                ).run(synthetic_texts)
            else:
                # Restore synthetic texts from last completed round for cross-domain eval
                synthetic_texts = self._restore_synthetic_texts_for_downstream(
                    last_completed_round=all_round_metrics[-1]["round_id"] if all_round_metrics else None,
                )
                downstream_summary = self._build_downstream_stub(
                    enabled=False,
                    status="disabled",
                    kind=downstream_cfg.get("kind", "none"),
                )

            # Run cross-domain evaluation if enabled (for transfer learning experiments)
            cross_domain_summary = None
            cross_domain_cfg = self.config.cross_domain_eval
            if bool(cross_domain_cfg.get("enabled", False)) and synthetic_texts:
                cross_domain_summary = self._run_cross_domain_eval(
                    synthetic_texts=synthetic_texts,
                    cross_domain_cfg=cross_domain_cfg,
                    current_round=current_round,
                    rounds_total=rounds,
                    checkpoint_path=last_checkpoint_path,
                )

            self._write_privacy_ledger_snapshot(privacy_ledger)
            summary = self._build_metrics_summary(
                rounds_total=rounds,
                server_ctx=server_ctx,
                all_round_metrics=all_round_metrics,
                privacy_ledger=privacy_ledger,
                status="completed",
                downstream_summary=downstream_summary,
                last_completed_round=all_round_metrics[-1]["round_id"] if all_round_metrics else None,
                resume_requested=resume,
            )
            if downstream_summary is not None:
                summary["synthetic_corpus_path"] = downstream_summary.get(
                    "synthetic_corpus_path"
                )
                summary["baseline_summaries"] = downstream_summary.get(
                    "baseline_summaries", {}
                )
            write_json(self.metrics_summary_path, summary)
            self._write_artifact_manifest(
                round_manifests=round_manifests,
                downstream_summary=downstream_summary,
            )
            self._write_run_state(
                status="completed",
                phase="finished",
                rounds_total=rounds,
                completed_rounds=len(all_round_metrics),
                current_round=current_round,
                checkpoint_path=last_checkpoint_path,
                downstream_status=downstream_summary.get("status") if downstream_summary is not None else None,
                resume_requested=resume,
            )
            self.logger.info(
                "Experiment %s | finished | summary=%s",
                self.experiment_id,
                self.metrics_summary_path,
            )
            return summary
        except KeyboardInterrupt as exc:
            self._write_failure_summary(
                status="interrupted",
                phase="interrupted",
                exc=exc,
                rounds_total=rounds,
                completed_rounds=len(all_round_metrics),
                current_round=current_round,
                checkpoint_path=last_checkpoint_path,
                resume_requested=resume,
            )
            self._write_privacy_ledger_snapshot(privacy_ledger)
            interrupted_summary = self._build_metrics_summary(
                rounds_total=rounds,
                server_ctx=server_ctx,
                all_round_metrics=all_round_metrics,
                privacy_ledger=privacy_ledger,
                status="interrupted",
                downstream_summary=self._build_downstream_stub(
                    enabled=bool(downstream_cfg.get("enabled")),
                    status="skipped_due_to_interrupt" if bool(downstream_cfg.get("enabled")) else "disabled",
                    kind=downstream_cfg.get("kind", "none"),
                    message="Experiment was interrupted by signal or keyboard input.",
                ),
                last_completed_round=all_round_metrics[-1]["round_id"] if all_round_metrics else None,
                resume_requested=resume,
            )
            write_json(self.metrics_summary_path, interrupted_summary)
            self._write_artifact_manifest(
                round_manifests=round_manifests,
                downstream_summary=interrupted_summary["downstream_eval"],
            )
            self._write_run_state(
                status="interrupted",
                phase="interrupted",
                rounds_total=rounds,
                completed_rounds=len(all_round_metrics),
                current_round=current_round,
                checkpoint_path=last_checkpoint_path,
                downstream_status=interrupted_summary["downstream_eval"]["status"],
                last_error={
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                resume_requested=resume,
            )
            raise
        except Exception as exc:
            self._write_failure_summary(
                status="failed",
                phase="failed",
                exc=exc,
                rounds_total=rounds,
                completed_rounds=len(all_round_metrics),
                current_round=current_round,
                checkpoint_path=last_checkpoint_path,
                resume_requested=resume,
            )
            self._write_privacy_ledger_snapshot(privacy_ledger)
            failed_summary = self._build_metrics_summary(
                rounds_total=rounds,
                server_ctx=server_ctx,
                all_round_metrics=all_round_metrics,
                privacy_ledger=privacy_ledger,
                status="failed",
                downstream_summary=self._build_downstream_stub(
                    enabled=bool(downstream_cfg.get("enabled")),
                    status="failed_before_eval" if bool(downstream_cfg.get("enabled")) else "disabled",
                    kind=downstream_cfg.get("kind", "none"),
                    message="Experiment failed before completing downstream evaluation.",
                ),
                last_completed_round=all_round_metrics[-1]["round_id"] if all_round_metrics else None,
                resume_requested=resume,
            )
            failed_summary["error"] = {
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            write_json(self.metrics_summary_path, failed_summary)
            self._write_artifact_manifest(
                round_manifests=round_manifests,
                downstream_summary=failed_summary["downstream_eval"],
            )
            self._write_run_state(
                status="failed",
                phase="failed",
                rounds_total=rounds,
                completed_rounds=len(all_round_metrics),
                current_round=current_round,
                checkpoint_path=last_checkpoint_path,
                downstream_status=failed_summary["downstream_eval"]["status"],
                last_error=failed_summary["error"],
                resume_requested=resume,
            )
            self.logger.exception("Experiment %s | failed", self.experiment_id)
            raise
        finally:
            if progress is not None:
                progress.close()
            self._restore_signal_handlers()
            close_experiment_file_logger("thesis_platform")




