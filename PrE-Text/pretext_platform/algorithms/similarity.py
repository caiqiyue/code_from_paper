"""Embedding helpers for private texts and synthetic candidates."""

from __future__ import annotations

import numpy as np

from pretext_platform.algorithms.variation import Variation


class Similarity:
    """Build the embeddings used by nearest-neighbor quality scoring."""

    @staticmethod
    def sentence_embedding(texts, embedding_model, device="cuda"):
        sentence_embeddings = embedding_model.encode(texts, device=device)
        return np.vstack(sentence_embeddings)

    @staticmethod
    def concat_embedding(texts, config):
        return Similarity.sentence_embedding(
            texts,
            config["mpnet"],
            device=config["device"],
        )

    @staticmethod
    def lookahead_embedding(parent_set, attention_mask, mlm_probability, config):
        tokenizer = config["tokenizer"]
        embeddings_list = []
        for _ in range(config["lookahead"]):
            curr_variation = Variation.produce_variation(
                {"input_ids": parent_set, "attention_mask": attention_mask},
                mlm_probability,
                config,
            )["input_ids"]
            curr_variation_texts = tokenizer.batch_decode(curr_variation, skip_special_tokens=True)
            curr_variation_embedding = Similarity.concat_embedding(curr_variation_texts, config)[None, :, :]
            embeddings_list.append(curr_variation_embedding)
        embeddings_cat = np.concatenate(embeddings_list, axis=0)
        embeddings_mean = np.mean(embeddings_cat, axis=0)
        return embeddings_mean
