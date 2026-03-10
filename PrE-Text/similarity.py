"""Embedding helpers for private texts and synthetic candidates."""

import numpy as np

from variation import Variation


class Similarity:
    """Build the embeddings used by nearest-neighbor quality scoring."""

    @staticmethod
    def sentence_embedding(texts, embedding_model, device="cuda"):
        """Encode raw texts into sentence embeddings with the chosen encoder."""
        sentence_embeddings = embedding_model.encode(texts, device=device)  # Run batched embedding inference.
        return np.vstack(sentence_embeddings)

    @staticmethod
    def concat_embedding(texts, config):
        """Compute the sentence-transformer embeddings used by the DP histogram."""
        mpnet_embeds = Similarity.sentence_embedding(
            texts,
            config["mpnet"],
            device=config["device"],
        )
        return mpnet_embeds

    @staticmethod
    def lookahead_embedding(parent_set, attention_mask, mlm_probability, config):
        """Average embeddings over several future variations of the same parent set."""
        tokenizer = config["tokenizer"]
        embeddings_list = []
        for _ in range(config["lookahead"]):
            curr_variation = Variation.produce_variation(
                {"input_ids": parent_set, "attention_mask": attention_mask},
                mlm_probability,
                config,
            )["input_ids"]
            curr_variation_texts = tokenizer.batch_decode(
                curr_variation,
                skip_special_tokens=True,
            )  # Convert the sampled future candidates back to text before embedding.
            curr_variation_embedding = Similarity.concat_embedding(curr_variation_texts, config)[None, :, :]
            embeddings_list.append(curr_variation_embedding)
        embeddings_cat = np.concatenate(embeddings_list, axis=0)  # Stack lookahead samples along the Monte Carlo axis.
        embeddings_mean = np.mean(embeddings_cat, axis=0)  # Average future embeddings to reduce score variance.
        return embeddings_mean
