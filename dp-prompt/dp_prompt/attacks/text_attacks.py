from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline


def build_attack_views(dataframe: pd.DataFrame, attack_mode: str) -> dict[str, dict[str, Any]]:
    if attack_mode not in {"static", "adaptive"}:
        raise ValueError(f"Unsupported attack mode: {attack_mode}")

    if attack_mode == "static":
        return {
            "train": {"frame": dataframe[dataframe["split"] == "train"].copy(), "text_field": "text"},
            "validation": {
                "frame": dataframe[dataframe["split"] == "validation"].copy(),
                "text_field": "text",
            },
            "test": {
                "frame": dataframe[dataframe["split"] == "test"].copy(),
                "text_field": "sanitized_text",
            },
        }

    return {
        split: {
            "frame": dataframe[dataframe["split"] == split].copy(),
            "text_field": "sanitized_text",
        }
        for split in ("train", "validation", "test")
    }


def _run_single_attack(view_bundle: dict[str, dict[str, Any]]) -> dict[str, Any]:
    train_view = view_bundle["train"]
    val_view = view_bundle["validation"]
    test_view = view_bundle["test"]

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1),
        LogisticRegression(max_iter=1000),
    )
    model.fit(
        train_view["frame"][train_view["text_field"]].tolist(),
        train_view["frame"]["author_id"].tolist(),
    )

    def evaluate(frame: pd.DataFrame, text_field: str) -> dict[str, Any]:
        truth = frame["author_id"].tolist()
        predictions = model.predict(frame[text_field].tolist())
        return {
            "accuracy": accuracy_score(truth, predictions),
            "macro_f1": f1_score(truth, predictions, average="macro"),
            "num_examples": int(len(frame)),
        }

    return {
        "backend": "sklearn_tfidf_logreg",
        "split_semantics": {
            "train": train_view["text_field"],
            "validation": val_view["text_field"],
            "test": test_view["text_field"],
        },
        "validation": evaluate(val_view["frame"], val_view["text_field"]),
        "test": evaluate(test_view["frame"], test_view["text_field"]),
        "train_text_field": train_view["text_field"],
        "validation_text_field": val_view["text_field"],
        "test_text_field": test_view["text_field"],
    }


def run_text_attacks(dataframe: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    modes = config.get("attacks", {}).get("text_attack_modes", ["static", "adaptive"])
    results: dict[str, Any] = {}
    for mode in modes:
        views = build_attack_views(dataframe, attack_mode=mode)
        results[mode] = _run_single_attack(views)
    return results
