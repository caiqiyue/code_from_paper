#!/usr/bin/env python3
"""Run a single E1 Aug-PE baseline experiment.

Aug-PE (Xie et al., ICML 2024) is a centralized-DP private-evolution algorithm
that selects high-quality public seeds via a DP histogram over nearest-neighbour
counts. We run its PE step via ``aug_pe_adapter.run_augpe_seed_selection``,
materialize the selected seeds as a curated initialization pool, then defer the
remaining "Stage 2 + downstream eval" steps to round19's existing pipeline
(``stage1_mode: c4_only`` with the curated pool, which random-samples and runs
LLaMA few-shot expansion + GPT-2 eval just like the C4-only baseline).

This script writes a runner-compatible sidecar JSON whose
``runtime_artifacts.eval_summary.metrics.best_top1`` is consumed by the
``round23_dynamic_experiment_runner``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from aug_pe_adapter import run_augpe_seed_selection
from round23_runtime_utils import (
    collect_runtime_artifacts,
    deep_merge,
    load_yaml_with_inherits,
    run_round19_selector_subprocess,
)


ROUND23_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROUND23_ROOT.parent
DEFAULT_AUGPE_REPO_ROOT = REPO_ROOT / "aug-pe"


def _flatten_json_texts(payload: Any) -> list[str]:
    """Mirror PrE-Text/round19's text-flattening contract for JSON corpora."""
    if payload is None:
        return []
    if isinstance(payload, str):
        text = payload.strip()
        return [text] if text else []
    if isinstance(payload, list):
        out: list[str] = []
        for item in payload:
            out.extend(_flatten_json_texts(item))
        return out
    if isinstance(payload, dict):
        out: list[str] = []
        for key in sorted(payload.keys(), key=lambda x: (int(x) if str(x).isdigit() else str(x))):
            out.extend(_flatten_json_texts(payload[key]))
        return out
    text = str(payload).strip()
    return [text] if text else []


def _load_json_texts(path: Path, *, min_words: int = 0) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    texts = _flatten_json_texts(payload)
    if min_words > 0:
        texts = [text for text in texts if len(text.split()) > min_words]
    return texts


def _resolve_repo_path(raw: str) -> Path:
    candidate = Path(str(raw))
    if candidate.is_absolute():
        return candidate
    # Project convention: data paths in configs are relative to the repo root.
    return (REPO_ROOT / candidate).resolve()


def _generate_override_config(
    *,
    original_config_path: Path,
    output_root: Path,
    curated_init_path: Path,
    seed_top_k: int,
    epsilon: float,
    delta: float,
) -> tuple[Path, str, str]:
    original_cfg = load_yaml_with_inherits(original_config_path)
    experiment_id = str(original_cfg.get("meta", {}).get("experiment_id", original_config_path.stem))
    dataset_name = str(original_cfg.get("data", {}).get("dataset_name", "unknown"))
    override = {
        "meta": {
            "augpe_baseline_runtime": {
                "enabled": True,
                "seed_top_k": int(seed_top_k),
                "epsilon": float(epsilon),
                "delta": float(delta),
                "curated_initialization_path": str(curated_init_path.resolve()),
            }
        },
        "paths": {
            "output_root": str(output_root.resolve()),
        },
        "data": {
            # Re-route the init pool to the Aug-PE-curated seed file. Round19's
            # ``c4_only`` path then random-samples from this small curated pool,
            # giving Stage 2 bootstrap exactly the PE-selected seeds.
            "initialization_path": str(curated_init_path.resolve()),
            # Honor the curated pool size; do not subsample further.
            "initialization_limit": int(seed_top_k),
        },
        "pipeline": {
            "stage1_mode": "c4_only",
            "stage2_mode": "pretext_bootstrap",
            "run_eval": True,
        },
        "selector": {
            "seed_top_k": int(seed_top_k),
            "seed_budget_rule": {
                "enabled": False,
                "mode": "hierarchical_shape_routing",
            },
        },
        "round23_controller": {
            "enabled": False,
        },
        "eval": {
            "enabled": True,
            "mode": "pretext_small",
        },
    }
    merged = deep_merge(original_cfg, override)
    override_path = output_root / f"{experiment_id}_augpe_override.yaml"
    override_path.parent.mkdir(parents=True, exist_ok=True)
    with override_path.open("w", encoding="utf-8") as handle:
        yaml.dump(merged, handle)
    return override_path, experiment_id, dataset_name


def _write_runtime_sidecar(
    *,
    output_root: Path,
    experiment_id: str,
    dataset_name: str,
    override_config_path: Path,
    runtime_artifacts: dict[str, Any],
    augpe_payload: dict[str, Any],
) -> Path:
    sidecar = {
        "budget_policy_type": "augpe",
        "dataset_name": dataset_name,
        "experiment_id": experiment_id,
        **augpe_payload,
        "override_config_path": str(override_config_path),
        "runtime_artifacts": runtime_artifacts,
    }
    sidecar_path = output_root / f"{experiment_id}_augpe_runtime.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")
    return sidecar_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run E1 Aug-PE baseline (centralized DP seed selection + LLaMA expansion)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--augpe-repo-root",
        default=str(DEFAULT_AUGPE_REPO_ROOT),
        help="Path to the cloned AI-secure/aug-pe repository.",
    )
    parser.add_argument("--seed-top-k", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=1.29)
    parser.add_argument("--delta", type=float, default=3e-6)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument(
        "--reference-budget",
        type=int,
        default=20,
        help="Accepted for runner compatibility; treated as seed_top_k upper bound.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    source_config = Path(args.config)
    if not source_config.is_absolute():
        source_config = (ROUND23_ROOT / source_config).resolve()
    raw_cfg = load_yaml_with_inherits(source_config)
    meta = raw_cfg.get("meta", {}) or {}
    data_cfg = raw_cfg.get("data", {}) or {}
    seed = int(meta.get("seed", 42))
    dataset_name = str(data_cfg.get("dataset_name", "unknown"))
    experiment_id = str(meta.get("experiment_id", source_config.stem))

    # Pull policy knobs from the config if present (so per-experiment configs can override CLI defaults).
    augpe_cfg = raw_cfg.get("augpe_baseline", {}) or {}
    seed_top_k = int(augpe_cfg.get("seed_top_k", args.seed_top_k))
    epsilon = float(augpe_cfg.get("epsilon", args.epsilon))
    delta = float(augpe_cfg.get("delta", args.delta))

    train_path = _resolve_repo_path(data_cfg["train_path"])
    init_path = _resolve_repo_path(data_cfg["initialization_path"])

    print(
        f"[e1_augpe] experiment_id={experiment_id} dataset={dataset_name} "
        f"seed_top_k={seed_top_k} epsilon={epsilon} delta={delta}"
    )
    print(f"[e1_augpe] train_path={train_path}")
    print(f"[e1_augpe] init_path={init_path}")

    private_texts = _load_json_texts(train_path)
    initialization_min_words = int(data_cfg.get("initialization_min_words", 20))
    init_pool = _load_json_texts(init_path, min_words=initialization_min_words)
    initialization_limit = data_cfg.get("initialization_limit")
    if initialization_limit not in (None, "", 0):
        init_pool = init_pool[: int(initialization_limit)]

    print(f"[e1_augpe] private_texts={len(private_texts)} init_pool={len(init_pool)}")

    selected_seeds = run_augpe_seed_selection(
        private_texts=private_texts,
        init_pool=init_pool,
        seed_top_k=seed_top_k,
        epsilon=epsilon,
        delta=delta,
        seed=seed,
        augpe_repo_root=Path(args.augpe_repo_root),
    )
    if not selected_seeds:
        print("[ERROR] Aug-PE returned zero seeds; aborting.", file=sys.stderr)
        return 2

    curated_init_dir = output_root / "augpe_curated"
    curated_init_dir.mkdir(parents=True, exist_ok=True)
    curated_init_path = curated_init_dir / "initialization.json"
    with curated_init_path.open("w", encoding="utf-8") as handle:
        json.dump(selected_seeds, handle, ensure_ascii=False)
    selected_seeds_log = curated_init_dir / "augpe_selected_seeds.json"
    with selected_seeds_log.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "seed_top_k": seed_top_k,
                "epsilon": epsilon,
                "delta": delta,
                "seed": seed,
                "selected_seeds": selected_seeds,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    override_path, _, _ = _generate_override_config(
        original_config_path=source_config,
        output_root=output_root,
        curated_init_path=curated_init_path,
        seed_top_k=seed_top_k,
        epsilon=epsilon,
        delta=delta,
    )
    print(f"[e1_augpe] Invoking round19 selector subprocess with {override_path}")
    result = run_round19_selector_subprocess(
        config_path=override_path,
        timeout_seconds=args.timeout_seconds,
    )
    if result.returncode != 0:
        print(result.stdout, end="")
        print(f"[ERROR] round19 runtime failed:\n{result.stderr}", file=sys.stderr)
        return int(result.returncode)
    print(result.stdout, end="")

    runtime_artifacts = collect_runtime_artifacts(output_root)
    augpe_payload = {
        "seed_top_k": seed_top_k,
        "epsilon": epsilon,
        "delta": delta,
        "augpe_repo_root": str(Path(args.augpe_repo_root).resolve()),
        "selected_seed_count": len(selected_seeds),
        "curated_initialization_path": str(curated_init_path.resolve()),
        "selected_seeds_log_path": str(selected_seeds_log.resolve()),
    }
    sidecar_path = _write_runtime_sidecar(
        output_root=output_root,
        experiment_id=experiment_id,
        dataset_name=dataset_name,
        override_config_path=override_path,
        runtime_artifacts=runtime_artifacts,
        augpe_payload=augpe_payload,
    )
    print(f"[e1_augpe] Done. Sidecar: {sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
