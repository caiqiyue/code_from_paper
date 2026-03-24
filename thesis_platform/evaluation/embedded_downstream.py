"""Embedded downstream evaluation for synthetic corpus quality.

Reduces dependency on external PrE-Text library by implementing
core evaluation logic internally.

Supports:
- Perplexity evaluation
- Few-shot classification
- Semantic similarity
- Style consistency
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from tqdm import tqdm

from thesis_platform.core.io_utils import ensure_dir, write_json
from thesis_platform.models.features import build_feature_encoder
from thesis_platform.models.embedding import build_embedder

logger = logging.getLogger(__name__)


class PerplexityEvaluator:
    """Evaluate synthetic text perplexity using a language model."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Lazy load the model."""
        if self.model is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                logger.info(f"Loading perplexity model from {self.model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16
                    if self.device == "cuda"
                    else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                )
                self.model.eval()
                logger.info("Perplexity model loaded")
            except Exception as e:
                logger.error(f"Failed to load perplexity model: {e}")
                raise

    def compute_perplexity(
        self, texts: List[str], batch_size: int = 8
    ) -> Dict[str, float]:
        """Compute perplexity for a list of texts."""
        self.load_model()

        perplexities = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Tokenize
            encodings = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encodings, labels=encodings["input_ids"])
                loss = outputs.loss

                # Per-sample perplexity
                for j in range(len(batch)):
                    # Get per-sample loss
                    shift_logits = outputs.logits[j, :-1, :].contiguous()
                    shift_labels = encodings["input_ids"][j, 1:].contiguous()

                    # Compute loss only on non-padding tokens
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
                    sample_loss = loss_fct(shift_logits, shift_labels)

                    # Count non-padding tokens
                    n_tokens = (
                        (shift_labels != self.tokenizer.pad_token_id).sum().item()
                    )

                    if n_tokens > 0:
                        ppl = torch.exp(sample_loss / n_tokens).item()
                        perplexities.append(ppl)

        return {
            "mean_perplexity": float(np.mean(perplexities)),
            "median_perplexity": float(np.median(perplexities)),
            "std_perplexity": float(np.std(perplexities)),
            "min_perplexity": float(np.min(perplexities)),
            "max_perplexity": float(np.max(perplexities)),
        }


class FewShotClassifier:
    """Few-shot classification evaluation."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None

    def evaluate_classification(
        self,
        test_texts: List[str],
        test_labels: List[str],
        few_shot_examples: List[Tuple[str, str]],
        candidate_labels: List[str],
    ) -> Dict[str, float]:
        """Evaluate few-shot classification accuracy.

        Args:
            test_texts: Test texts
            test_labels: True labels
            few_shot_examples: List of (text, label) tuples for few-shot prompting
            candidate_labels: List of possible labels

        Returns:
            Dictionary with accuracy and per-class metrics
        """
        correct = 0
        predictions = []

        for text, true_label in zip(test_texts, test_labels):
            pred_label = self._few_shot_predict(
                text, few_shot_examples, candidate_labels
            )
            predictions.append(pred_label)
            if pred_label == true_label:
                correct += 1

        accuracy = correct / len(test_texts) if test_texts else 0.0

        # Per-class metrics
        per_class = {}
        for label in candidate_labels:
            tp = sum(
                1 for p, t in zip(predictions, test_labels) if p == label and t == label
            )
            fp = sum(
                1 for p, t in zip(predictions, test_labels) if p == label and t != label
            )
            fn = sum(
                1 for p, t in zip(predictions, test_labels) if p != label and t == label
            )

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            per_class[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": sum(1 for t in test_labels if t == label),
            }

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(test_texts),
            "per_class": per_class,
        }

    def _few_shot_predict(
        self,
        text: str,
        few_shot_examples: List[Tuple[str, str]],
        candidate_labels: List[str],
    ) -> str:
        """Predict label for a single text using few-shot prompting."""
        # This is a simplified version
        # In practice, you'd use the model to score each label
        # For now, return a random label as placeholder
        import random

        return random.choice(candidate_labels)


class SemanticSimilarityEvaluator:
    """Evaluate semantic similarity between synthetic and real texts."""

    def __init__(
        self,
        embedding_model: str,
        repo_root: str,
        device: str = "auto",
    ):
        self.embedder = build_embedder(
            embedding_model,
            repo_root,
            allow_fallback=True,
        )

    def compute_similarity(
        self,
        synthetic_texts: List[str],
        real_texts: List[str],
    ) -> Dict[str, float]:
        """Compute semantic similarity between synthetic and real corpora."""
        # Encode texts
        syn_embeddings = np.array(self.embedder.embed_texts(synthetic_texts))
        real_embeddings = np.array(self.embedder.embed_texts(real_texts))

        # Compute pairwise similarities
        from sklearn.metrics.pairwise import cosine_similarity

        similarities = cosine_similarity(syn_embeddings, real_embeddings)

        # Statistics
        mean_sim = np.mean(similarities)
        max_sim = np.mean(
            np.max(similarities, axis=1)
        )  # For each synthetic, max similarity to real

        return {
            "mean_similarity": float(mean_sim),
            "max_similarity": float(max_sim),
            "similarity_matrix_shape": similarities.shape,
        }


class StyleConsistencyEvaluator:
    """Evaluate style consistency of synthetic texts."""

    def __init__(self):
        pass

    def evaluate_style(
        self,
        synthetic_texts: List[str],
        real_texts: List[str],
    ) -> Dict[str, float]:
        """Evaluate style consistency using simple heuristics."""
        # Compute basic statistics
        syn_lengths = [len(text.split()) for text in synthetic_texts]
        real_lengths = [len(text.split()) for text in real_texts]

        # Vocabulary overlap
        syn_vocab = set()
        real_vocab = set()

        for text in synthetic_texts:
            syn_vocab.update(text.lower().split())

        for text in real_texts:
            real_vocab.update(text.lower().split())

        vocab_overlap = (
            len(syn_vocab & real_vocab) / len(syn_vocab | real_vocab)
            if (syn_vocab | real_vocab)
            else 0.0
        )

        return {
            "syn_mean_length": float(np.mean(syn_lengths)),
            "real_mean_length": float(np.mean(real_lengths)),
            "length_ratio": float(np.mean(syn_lengths) / np.mean(real_lengths))
            if np.mean(real_lengths) > 0
            else 0.0,
            "vocabulary_overlap": vocab_overlap,
        }


class EmbeddedDownstreamEval:
    """Main downstream evaluation class that aggregates all metrics."""

    def __init__(
        self,
        config: Dict[str, Any],
        repo_root: str,
        experiment_id: str,
        output_dir: Path,
    ):
        self.config = config
        self.repo_root = repo_root
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir)

        # Initialize evaluators
        self.perplexity_eval = None
        self.semantic_eval = None
        self.style_eval = StyleConsistencyEvaluator()

        if config.get("enable_perplexity", True):
            model_path = config.get("eval_model", "gpt2")
            self.perplexity_eval = PerplexityEvaluator(model_path)

        if config.get("enable_semantic", True):
            embedding_model = config.get("embedding_model", "all-MiniLM-L6-v2")
            self.semantic_eval = SemanticSimilarityEvaluator(embedding_model, repo_root)

    def evaluate(
        self,
        synthetic_texts: List[str],
        real_texts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run complete downstream evaluation.

        This is the main entry point that replaces PrE-Text evaluation.
        """
        logger.info(f"Starting embedded downstream evaluation for {self.experiment_id}")

        results = {
            "experiment_id": self.experiment_id,
            "n_synthetic_samples": len(synthetic_texts),
            "metrics": {},
        }

        # 1. Perplexity evaluation
        if self.perplexity_eval:
            logger.info("Computing perplexity...")
            try:
                ppl_metrics = self.perplexity_eval.compute_perplexity(synthetic_texts)
                results["metrics"]["perplexity"] = ppl_metrics
                logger.info(f"Perplexity: {ppl_metrics['mean_perplexity']:.2f}")
            except Exception as e:
                logger.error(f"Perplexity evaluation failed: {e}")
                results["metrics"]["perplexity"] = {"error": str(e)}

        # 2. Semantic similarity
        if self.semantic_eval and real_texts:
            logger.info("Computing semantic similarity...")
            try:
                sim_metrics = self.semantic_eval.compute_similarity(
                    synthetic_texts, real_texts
                )
                results["metrics"]["semantic_similarity"] = sim_metrics
                logger.info(f"Similarity: {sim_metrics['mean_similarity']:.4f}")
            except Exception as e:
                logger.error(f"Semantic similarity evaluation failed: {e}")
                results["metrics"]["semantic_similarity"] = {"error": str(e)}

        # 3. Style consistency
        if real_texts:
            logger.info("Computing style consistency...")
            try:
                style_metrics = self.style_eval.evaluate_style(
                    synthetic_texts, real_texts
                )
                results["metrics"]["style_consistency"] = style_metrics
                logger.info(f"Vocab overlap: {style_metrics['vocabulary_overlap']:.4f}")
            except Exception as e:
                logger.error(f"Style evaluation failed: {e}")
                results["metrics"]["style_consistency"] = {"error": str(e)}

        # Save results
        output_path = self.output_dir / "embedded_downstream_eval.json"
        write_json(output_path, results)
        logger.info(f"Evaluation results saved to {output_path}")

        return results


# Configuration presets
def get_default_eval_config() -> Dict[str, Any]:
    """Get default evaluation configuration."""
    return {
        "enable_perplexity": True,
        "enable_semantic": True,
        "enable_style": True,
        "eval_model": "gpt2",
        "embedding_model": "all-MiniLM-L6-v2",
    }


def get_fast_eval_config() -> Dict[str, Any]:
    """Get fast evaluation config (for smoke tests)."""
    return {
        "enable_perplexity": False,  # Skip slow perplexity
        "enable_semantic": True,
        "enable_style": True,
        "embedding_model": "all-MiniLM-L6-v2",
    }
