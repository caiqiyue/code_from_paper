from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_platform.core.config import ExperimentConfig
from thesis_platform.core.io_utils import read_json, read_jsonl, write_json, ensure_dir
from thesis_platform.evaluation.downstream_eval import run_pretext_glue_eval


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
        description="Run GLUE-style downstream evaluation on one thesis_platform synthetic corpus."
    )
    parser.add_argument("--experiment-id", help="Experiment id to resolve via outputs/thesis_platform/<id>_latest.json.")
    parser.add_argument("--run-dir", help="Explicit thesis_platform experiment output directory.")
    parser.add_argument(
        "--output-root",
        default=str(_workspace_root() / "outputs" / "thesis_platform"),
        help="Root directory containing thesis_platform experiment outputs.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["sst2"],
        help="One or more tasks from: sst2 qqp qnli imdb rotten_tomatoes.",
    )
    parser.add_argument("--epochs", type=int, help="Optional override for classifier epochs.")
    parser.add_argument("--batch-size", type=int, help="Optional override for batch size.")
    parser.add_argument("--lr", type=float, help="Optional override for learning rate.")
    args = parser.parse_args()

    output_root = Path(str(args.output_root)).expanduser().resolve()
    run_dir = _resolve_run_dir(
        experiment_id=args.experiment_id,
        run_dir=args.run_dir,
        output_root=output_root,
    )
    thesis_config = _load_run_config(run_dir)
    stage2_dir = run_dir / "downstream_eval" / "stage2"
    corpus_path = stage2_dir / "llama7b_text_syn.json"

    synthetic_texts: list[str] = []
    if corpus_path.exists():
        synthetic_texts = [str(item).strip() for item in read_json(corpus_path) if str(item).strip()]
    else:
        # Fallback: restore from last completed round's client_assigned_samples
        rounds = int(thesis_config.federation.get("rounds", 1))
        last_round = rounds - 1
        for candidate in range(last_round, -1, -1):
            client_assigned_path = run_dir / f"round_{candidate:03d}" / "client_assigned_samples.jsonl"
            if client_assigned_path.exists():
                try:
                    fallback_texts = [
                        _render_sample_payload(dict(row))
                        for row in read_jsonl(client_assigned_path)
                    ]
                    synthetic_texts = [t for t in fallback_texts if t]
                except Exception:
                    synthetic_texts = []
                break

        if not synthetic_texts:
            raise FileNotFoundError(
                f"Synthetic corpus not found at {corpus_path} and no fallback corpus could be restored "
                f"from round artifacts under {run_dir}."
            )

        # Write fallback texts to stage2_dir so downstream eval can read them
        ensure_dir(stage2_dir)
        write_json(corpus_path, synthetic_texts)

    overrides: dict[str, object] = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["learning_rate"] = args.lr

    summary = run_pretext_glue_eval(
        thesis_config,
        stage2_dir=stage2_dir,
        output_dir=run_dir / "glue_eval",
        tasks=list(args.tasks),
        overrides=overrides,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
