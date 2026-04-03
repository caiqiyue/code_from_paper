from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_platform.core.config import ExperimentConfig
from thesis_platform.core.io_utils import read_json, read_jsonl, write_json, ensure_dir
from thesis_platform.evaluation.downstream_eval import DownstreamEvalManager


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_run_dir(*, experiment_id: str | None, run_dir: str | None, output_root: Path) -> Path:
    if run_dir not in (None, ""):
        candidate = Path(str(run_dir)).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"run directory not found: {candidate}")
        return candidate

    if experiment_id in (None, ""):
        raise ValueError("Either --run-dir or --experiment-id must be provided.")

    pointer_path = output_root / f"{experiment_id}_latest.json"
    if not pointer_path.exists():
        raise FileNotFoundError(f"latest pointer not found: {pointer_path}")
    pointer = dict(read_json(pointer_path))
    resolved = Path(str(pointer.get("experiment_dir", ""))).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"experiment_dir recorded in latest pointer does not exist: {resolved}")
    return resolved


def _render_sample_payload(payload: dict) -> str:
    """Render a sample payload dict into a text string."""
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


def _load_run_config(run_dir: Path) -> ExperimentConfig:
    resolved_config_path = run_dir / "resolved_config.json"
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"resolved_config.json not found under run directory: {resolved_config_path}")
    payload = dict(read_json(resolved_config_path))
    config_path = Path(str(payload.get("config_path") or run_dir / "config.yaml")).resolve()
    config_raw = dict(payload.get("config", {}))
    if not config_raw:
        raise ValueError(f"resolved_config.json does not contain a non-empty config payload: {resolved_config_path}")
    return ExperimentConfig(path=config_path, raw=config_raw)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run thesis_platform downstream evaluation on an existing synthetic corpus export."
    )
    parser.add_argument("--experiment-id", help="Experiment id to resolve via outputs/thesis_platform/<id>_latest.json.")
    parser.add_argument("--run-dir", help="Explicit thesis_platform experiment output directory.")
    parser.add_argument(
        "--output-root",
        default=str(_workspace_root() / "outputs" / "thesis_platform"),
        help="Root directory containing thesis_platform experiment outputs.",
    )
    parser.add_argument(
        "--large-eval-mode",
        choices=["auto", "peft_lora", "full_finetune", "gpt2_xl"],
        default=None,
        help="Optional override for large_eval_mode.",
    )
    parser.add_argument(
        "--with-small-eval",
        action="store_true",
        help="Also run small downstream eval in addition to large eval.",
    )
    args = parser.parse_args()

    output_root = Path(str(args.output_root)).expanduser().resolve()
    run_dir = _resolve_run_dir(
        experiment_id=args.experiment_id,
        run_dir=args.run_dir,
        output_root=output_root,
    )
    thesis_config = _load_run_config(run_dir)
    downstream_dir = run_dir / "downstream_eval"
    stage2_dir = downstream_dir / "stage2"
    export_filename = str(thesis_config.downstream_eval.get("export_filename", "llama7b_text_syn.json"))
    corpus_path = stage2_dir / export_filename
    synthetic_texts: list[str] = []
    if corpus_path.exists():
        synthetic_texts = read_json(corpus_path)
    else:
        fallback_path = stage2_dir / "llama7b_text_syn.json"
        if fallback_path.exists():
            synthetic_texts = read_json(fallback_path)
        else:
            # Fallback: restore from last completed round's client_assigned_samples
            rounds = int(thesis_config.federation.get("rounds", 1))
            for candidate in range(rounds - 1, -1, -1):
                client_assigned_path = run_dir / f"round_{candidate:03d}" / "client_assigned_samples.jsonl"
                if client_assigned_path.exists():
                    try:
                        synthetic_texts = [
                            _render_sample_payload(dict(row))
                            for row in read_jsonl(client_assigned_path)
                        ]
                        synthetic_texts = [t for t in synthetic_texts if t]
                    except Exception:
                        synthetic_texts = []
                    if synthetic_texts:
                        ensure_dir(stage2_dir)
                        write_json(fallback_path, synthetic_texts)
                        break

            if not synthetic_texts:
                raise FileNotFoundError(
                    f"Synthetic corpus not found at {corpus_path} and no fallback corpus "
                    f"could be restored from round artifacts under {run_dir}."
                )
    if not isinstance(synthetic_texts, list):
        raise ValueError(f"Synthetic corpus must be a JSON array of texts: {corpus_path}")

    downstream_cfg = dict(thesis_config.raw.get("downstream_eval", {}))
    downstream_cfg["enabled"] = True
    downstream_cfg["kind"] = str(downstream_cfg.get("kind", "pretext_large_eval"))
    downstream_cfg["run_large_eval"] = True
    downstream_cfg["run_small_eval"] = bool(args.with_small_eval)
    if args.large_eval_mode is not None:
        downstream_cfg["large_eval_mode"] = args.large_eval_mode
    thesis_config.raw["downstream_eval"] = downstream_cfg

    summary = DownstreamEvalManager(
        thesis_config,
        experiment_id=str(thesis_config.meta.get("experiment_id", run_dir.name)),
        output_dir=downstream_dir,
    ).run([str(item) for item in synthetic_texts if str(item).strip()])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
