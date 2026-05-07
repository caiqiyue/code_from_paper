from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..thesis_bridge import load_yaml_config, resolve_repo_root
from .common_eval import run_external_stage1_summary_eval
from .dpga_adapter import build_dpga_stage1_summary
from .wasp_adapter import build_wasp_stage1_summary


def _resolve_relative_to_repo(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (resolve_repo_root() / path).resolve()


def _read_wasp_generated_jsonl(path: Path) -> list[str]:
    texts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        raw_text = record.get("X")
        if raw_text in (None, ""):
            raw_text = record.get("C", "")
        text = str(raw_text).strip()
        if len(text.split()) >= 2:
            texts.append(text)
    return texts


def _read_dpga_epoch_all_json(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    texts: list[str] = []
    for item in payload:
        text = str(item.get("text", "")).strip()
        if len(text.split()) >= 2:
            texts.append(text)
    return texts


def resolve_external_single_run_contract(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    stage1_mode = str(config.get("pipeline", {}).get("stage1_mode", "")).strip().lower()
    external_cfg = dict(config.get("external_baseline", {}))
    if not external_cfg:
        raise ValueError("external_baseline section is required for external single-run configs.")

    source_path = _resolve_relative_to_repo(str(external_cfg["source_artifact_path"]))
    summary_output_path = _resolve_relative_to_repo(
        str(
            external_cfg.get(
                "summary_output_path",
                Path(config.get("paths", {}).get("output_root", "paper-new-round19/outputs")) / "stage1_summary.json",
            )
        )
    )
    budget = int(external_cfg.get("expected_budget", 100))
    output_root = _resolve_relative_to_repo(str(config.get("paths", {}).get("output_root", "")))
    return {
        "config": config,
        "stage1_mode": stage1_mode,
        "external_cfg": external_cfg,
        "source_path": source_path,
        "summary_output_path": summary_output_path,
        "budget": budget,
        "output_root": output_root,
    }


def build_external_stage1_summary_from_config(config_path: str | Path) -> tuple[Path, dict[str, Any]]:
    contract = resolve_external_single_run_contract(config_path)
    stage1_mode = str(contract["stage1_mode"])
    source_path = Path(contract["source_path"])
    summary_output_path = Path(contract["summary_output_path"])
    budget = int(contract["budget"])
    if stage1_mode == "wasp_external":
        texts = _read_wasp_generated_jsonl(source_path)
        payload = build_wasp_stage1_summary(texts=texts, budget=budget)
    elif stage1_mode == "dpga_external":
        texts = _read_dpga_epoch_all_json(source_path)
        payload = build_dpga_stage1_summary(texts=texts, budget=budget)
    else:
        raise ValueError(f"Unsupported external stage1_mode: {stage1_mode}")

    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_output_path, payload


def run_external_single_run_from_config(
    config_path: str | Path,
    *,
    validate_only: bool = False,
) -> dict[str, Any]:
    contract = resolve_external_single_run_contract(config_path)
    config = dict(contract["config"])

    if validate_only:
        return {
            "experiment_id": config.get("meta", {}).get("experiment_id", ""),
            "stage1_mode": str(contract["stage1_mode"]),
            "source_path": str(contract["source_path"]),
            "source_exists": bool(Path(contract["source_path"]).exists()),
            "summary_path": str(contract["summary_output_path"]),
            "expected_budget": int(contract["budget"]),
            "external_baseline": dict(config.get("external_baseline", {})),
        }

    summary_path, _payload = build_external_stage1_summary_from_config(config_path)
    output_root = Path(contract["output_root"])
    eval_summary = run_external_stage1_summary_eval(
        summary_path=summary_path,
        config_path=config_path,
        output_dir=output_root / "eval",
    )
    return {
        "experiment_id": config.get("meta", {}).get("experiment_id", ""),
        "stage1_mode": str(config.get("pipeline", {}).get("stage1_mode", "")),
        "summary_path": str(summary_path),
        "eval": eval_summary,
    }
