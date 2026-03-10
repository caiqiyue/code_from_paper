"""Differentially private nearest-neighbor histogram computation."""

import sys
import time

import faiss
import numpy as np
import torch

from similarity import Similarity


class NN_Histogram:
    """Compute noisy nearest-neighbor vote histograms for candidate texts."""

    @staticmethod
    def split_given_size(a, size):
        """Split an array into fixed-size chunks for batched FAISS search."""
        return np.split(a, np.arange(size, len(a), size))

    @staticmethod
    def dp_nn_histogram(
        private_embeddings,
        parent_set,
        attention_mask,
        mlm_probability,
        config,
    ):
        """Build a thresholded noisy vote histogram over synthetic candidates."""
        sigma = config["sigma"]
        H = config["H"]
        embed_dim = config["embed_dim"]
        accelerator = config["accelerator"]
        nearest_neighbors_print = config["nearest_neighbors_print"]

        t0 = time.time()
        lookahead_embeddings = Similarity.lookahead_embedding(
            parent_set,
            attention_mask,
            mlm_probability,
            config,
        )  # Score candidates by the average embedding of their lookahead descendants.

        t1 = time.time()
        accelerator.print("Time for synthetic embeddings:", t1 - t0, file=sys.stderr)
        index_flat = faiss.IndexFlatL2(embed_dim)
        index_flat.add(lookahead_embeddings)  # Index the candidate embeddings once for repeated nearest-neighbor queries.
        t0 = time.time()

        private_embeds_split = NN_Histogram.split_given_size(private_embeddings, 50000)  # Bound memory use during FAISS search.
        res_D = []
        res_I = []
        for curr_embedding in private_embeds_split:
            len_curr_batch = curr_embedding.shape[0]
            closest_multiple = (len_curr_batch // config["num_workers"]) * config["num_workers"]
            randidx = np.random.choice(len_curr_batch, closest_multiple, replace=False)
            curr_embedding_subset = curr_embedding[randidx, :]
            process_split = np.split(curr_embedding_subset, config["num_workers"], axis=0)
            with accelerator.split_between_processes(process_split) as private_search_embed:
                D, I = index_flat.search(private_search_embed[0], 1)  # Find each private sample's nearest synthetic candidate.
                D = torch.from_numpy(D).to("cuda:{0}".format(accelerator.process_index))
                I = torch.from_numpy(I).to("cuda:{0}".format(accelerator.process_index))
            D = accelerator.gather(D).cpu().numpy()  # Gather search results from every process.
            I = accelerator.gather(I).cpu().numpy()
            res_D.append(D)
            res_I.append(I)
        I = np.concatenate(res_I, axis=0)
        D = np.concatenate(res_D, axis=0)
        accelerator.print(I.shape, file=sys.stderr)
        resulting_histogram, _ = np.histogram(
            I[:, 0],
            bins=[-0.5] + [x + 0.5 for x in range(lookahead_embeddings.shape[0])],
        )  # Count how many private texts map to each candidate.
        resulting_mean_distance = np.mean(D, axis=0).squeeze()
        first_nearest_neighbors_idx = I[:nearest_neighbors_print, 0]
        resulting_histogram_noised = (
            resulting_histogram.astype(np.float32)
            + np.random.standard_normal(size=resulting_histogram.shape) * sigma
        )  # Inject Gaussian noise for differential privacy.
        accelerator.print("Histogram before thresholding", np.sum(np.maximum(resulting_histogram_noised, 0)))
        noised_histogram_thresh = np.maximum(resulting_histogram_noised - H, 0.0)  # Discard weak noisy counts after thresholding.
        accelerator.print("Histogram after thresholding", np.sum(noised_histogram_thresh))
        t1 = time.time()
        accelerator.print("Time for nearest neighbor calculation:", t1 - t0, file=sys.stderr)
        return (
            noised_histogram_thresh,
            resulting_mean_distance,
            first_nearest_neighbors_idx,
        )
