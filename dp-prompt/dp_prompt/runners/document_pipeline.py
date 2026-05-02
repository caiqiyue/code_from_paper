from __future__ import annotations

from pathlib import Path
from typing import Any

from dp_prompt.attacks.text_attacks import run_text_attacks
from dp_prompt.data.loader import load_document_dataset
from dp_prompt.decoding.privacy import build_privacy_controls_summary
from dp_prompt.evaluation.utility import run_utility_evaluation
from dp_prompt.generation.backend import LocalTransformersGenerator
from dp_prompt.prompting.templates import render_document_prompt
from dp_prompt.utils.io import ensure_dir, write_json


def build_pipeline_components(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "privacy_controls": build_privacy_controls_summary(config.get("privacy", {})),
        "template_name": config.get("prompting", {}).get("template_name", "review_paraphrase"),
        "dataset_name": config.get("dataset", {}).get("name", "unknown"),
        "model_backend": config.get("model", {}).get("backend", "local_transformers"),
    }


def _build_output_dir(config: dict[str, Any]) -> Path:
    output_root = Path(config["runtime"]["output_root"]).expanduser()
    experiment_id = config["experiment"]["id"]
    return output_root / experiment_id


def run_document_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    bundle = load_document_dataset(config)
    dataframe = bundle.dataframe.copy()
    components = build_pipeline_components(config)
    output_dir = ensure_dir(_build_output_dir(config))

    dataframe["prompt"] = dataframe["text"].apply(
        lambda text: render_document_prompt(
            text,
            template_name=components["template_name"],
        )
    )

    generator = LocalTransformersGenerator.from_config(config["model"])
    generated = generator.generate_batch(
        dataframe["prompt"].tolist(),
        config.get("privacy", {}),
    )
    dataframe["sanitized_text"] = [entry["sanitized_text"] for entry in generated]

    sanitized_records = []
    for row, generated_item in zip(
        dataframe.to_dict(orient="records"),
        generated,
        strict=True,
    ):
        sanitized_records.append(
            {
                "sample_id": row["sample_id"],
                "text": row["text"],
                "sanitized_text": row["sanitized_text"],
                "label": row["label"],
                "author_id": row["author_id"],
                "split": row["split"],
                "prompt": row["prompt"],
                "generation_metadata": generated_item["generation_request"],
            }
        )
    write_json(output_dir / "sanitized_corpus.json", sanitized_records)

    utility_summary = run_utility_evaluation(dataframe, config)
    write_json(output_dir / "utility_summary.json", utility_summary)

    privacy_attack_summary = run_text_attacks(dataframe, config)
    write_json(output_dir / "privacy_attack_summary.json", privacy_attack_summary)

    privacy_controls_summary = components["privacy_controls"]
    write_json(output_dir / "privacy_controls_summary.json", privacy_controls_summary)

    experiment_summary = {
        "experiment_id": config["experiment"]["id"],
        "dataset": components["dataset_name"],
        "model_backend": components["model_backend"],
        "model_path": config["model"]["model_path"],
        "temperature": privacy_controls_summary["temperature"],
        "document_count": int(len(dataframe)),
        "artifact_paths": {
            "sanitized_corpus": str(output_dir / "sanitized_corpus.json"),
            "utility_summary": str(output_dir / "utility_summary.json"),
            "privacy_attack_summary": str(output_dir / "privacy_attack_summary.json"),
            "privacy_controls_summary": str(output_dir / "privacy_controls_summary.json"),
        },
        "utility": utility_summary,
        "privacy_attacks": privacy_attack_summary,
    }
    write_json(output_dir / "experiment_summary.json", experiment_summary)
    return experiment_summary
