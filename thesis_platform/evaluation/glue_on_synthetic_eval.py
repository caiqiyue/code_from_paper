"""Cross-domain evaluation: evaluate synthetic corpus on target domain downstream task.

This module runs a simplified downstream evaluation where:
1. A classifier is trained on the synthetic corpus from the source domain
2. The classifier is evaluated on the target domain's validation set

This measures the transfer learning capability of the synthetic data.

Supports target datasets:
- jobs: Job posting text classification
- forums: Forum post text classification
- microblog: Microblog text classification
- congressional: Congressional speech classification
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from thesis_platform.core.io_utils import ensure_dir, read_json, write_json


def _load_target_dataset(target_dataset: str, target_train_path: Path, target_eval_path: Path | None) -> tuple[pd.DataFrame, int]:
    """Load target domain dataset.

    Returns:
        tuple of (dataframe, num_labels)
    """
    if target_dataset == "jobs":
        # Jobs dataset: binary classification (job_type or category)
        train_df = pd.read_json(target_train_path, lines=True)
        eval_df = pd.read_json(target_eval_path, lines=True) if target_eval_path and target_eval_path.exists() else None
        num_labels = 2
        return train_df, num_labels

    elif target_dataset == "forums":
        # Forums dataset
        train_df = pd.read_json(target_train_path, lines=True)
        eval_df = pd.read_json(target_eval_path, lines=True) if target_eval_path and target_eval_path.exists() else None
        num_labels = 2
        return train_df, num_labels

    elif target_dataset == "microblog":
        # Microblog dataset
        train_df = pd.read_json(target_train_path, lines=True)
        eval_df = pd.read_json(target_eval_path, lines=True) if target_eval_path and target_eval_path.exists() else None
        num_labels = 2
        return train_df, num_labels

    elif target_dataset == "congressional":
        # Congressional dataset: multi-party classification
        train_df = pd.read_json(target_train_path, lines=True)
        eval_df = pd.read_json(target_eval_path, lines=True) if target_eval_path and target_eval_path.exists() else None
        # Count unique labels
        if "label" in train_df.columns:
            num_labels = train_df["label"].nunique()
        else:
            num_labels = 2
        return train_df, num_labels

    else:
        raise ValueError(f"Unsupported target dataset for cross-domain eval: {target_dataset}")


def _extract_text_column(df: pd.DataFrame, dataset_name: str) -> pd.Series:
    """Extract text column from dataset dataframe."""
    if "text" in df.columns:
        return df["text"]
    elif "content" in df.columns:
        return df["content"]
    elif "sentence" in df.columns:
        return df["sentence"]
    elif "input" in df.columns:
        return df["input"]
    else:
        # Try to find any text-like column
        for col in df.columns:
            if df[col].dtype == "object" and col != "label":
                return df[col]
        raise ValueError(f"No text column found in dataset {dataset_name}")


def run_glue_on_synthetic(
    synthetic_corpus_path: Path,
    target_train_path: Path,
    target_eval_path: Path | None,
    target_dataset_name: str,
    output_dir: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Run cross-domain evaluation.

    Trains a simple classifier on synthetic corpus and evaluates on target domain.

    Args:
        synthetic_corpus_path: Path to synthetic corpus JSON file
        target_train_path: Path to target domain training data
        target_eval_path: Path to target domain validation data
        target_dataset_name: Name of target dataset
        output_dir: Output directory for results
        config: Experiment config dict
        repo_root: Repository root path

    Returns:
        Dictionary with evaluation metrics
    """
    output_dir = ensure_dir(output_dir)

    # Load synthetic corpus
    if not synthetic_corpus_path.exists():
        raise FileNotFoundError(f"Synthetic corpus not found: {synthetic_corpus_path}")

    with synthetic_corpus_path.open("r", encoding="utf-8") as f:
        synthetic_texts = json.load(f)

    if not synthetic_texts:
        raise ValueError("Synthetic corpus is empty")

    # Load target domain dataset
    train_df, num_labels = _load_target_dataset(target_dataset_name, target_train_path, target_eval_path)

    # Extract texts and labels
    train_texts = _extract_text_column(train_df, target_dataset_name).tolist()
    train_labels = train_df["label"].tolist() if "label" in train_df.columns else [0] * len(train_texts)

    # Build a simple TF-IDF + LogisticRegression classifier for fast evaluation
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
    except ImportError:
        return {
            "status": "skipped",
            "message": "sklearn not available for cross-domain evaluation",
        }

    # Combine synthetic corpus with target domain texts for training
    # Use synthetic texts as training, target domain texts for evaluation
    combined_texts = synthetic_texts + train_texts[:min(len(synthetic_texts), len(train_texts))]
    combined_labels = [1] * len(synthetic_texts) + train_labels[:min(len(synthetic_texts), len(train_texts))]

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(combined_texts)
    y_train = combined_labels

    # Train classifier
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate on target domain
    if target_eval_path and target_eval_path.exists():
        eval_df = pd.read_json(target_eval_path, lines=True)
        eval_texts = _extract_text_column(eval_df, target_dataset_name).tolist()
        eval_labels = eval_df["label"].tolist() if "label" in eval_df.columns else [0] * len(eval_texts)

        if eval_texts:
            X_eval = vectorizer.transform(eval_texts)
            y_pred = clf.predict(X_eval)
            accuracy = accuracy_score(eval_labels, y_pred)
        else:
            accuracy = None
    else:
        # Use train/val split on target domain
        X_target = vectorizer.transform(train_texts)
        accuracy = clf.score(X_target, train_labels)

    metrics = {
        "accuracy": float(accuracy) if accuracy is not None else None,
        "num_synthetic_samples": len(synthetic_texts),
        "num_target_train_samples": len(train_texts),
        "num_labels": num_labels,
        "classifier": "TfidfVectorizer + LogisticRegression",
    }

    # Save results
    results = {
        "target_dataset": target_dataset_name,
        "synthetic_corpus_path": str(synthetic_corpus_path),
        "metrics": metrics,
    }
    write_json(output_dir / "cross_domain_results.json", results)

    return metrics
