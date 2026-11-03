"""
src/amr/similarity.py
GPU-accelerated AMR similarity retrieval engine supporting Top-K query matching,
batched tiled all-pairs computation, and latent embedding similarity.
"""

import logging
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch

from src.amr.device import get_gpu_manager
from src.amr.kernels.wrappers import gpu_compute_similarity

logger = logging.getLogger("amr.similarity")


class AMRSimilarityEngine:
    """
    Parallel AMR similarity and nearest-neighbor retrieval system using Apple Metal.
    """

    def __init__(self, feature_matrix: np.ndarray, metric: str = "euclidean"):
        self.feature_matrix = np.ascontiguousarray(feature_matrix, dtype=np.float32)
        self.num_records, self.num_features = self.feature_matrix.shape
        self.metric = metric
        self.gpu = get_gpu_manager()

    def query_top_k(
        self,
        query: np.ndarray,
        k: int = 10,
        metric: Optional[str] = None,
        use_metal_kernel: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Execute Metal similarity kernel and retrieve top-K most similar isolate indices and distances.
        Returns:
            top_k_indices: int64 array of shape [K]
            top_k_distances: float32 array of shape [K]
        """
        chosen_metric = metric or self.metric
        k = min(k, self.num_records)

        # 1. Compute distances across all N records via GPU kernel
        scores = gpu_compute_similarity(
            query=query,
            feature_matrix=self.feature_matrix,
            metric=chosen_metric,
            use_metal_kernel=use_metal_kernel
        )

        # 2. GPU Top-K selection using PyTorch MPS
        scores_t = torch.from_numpy(scores).to(self.gpu.torch_device)
        if chosen_metric == "euclidean":
            # Smallest distance is most similar
            vals, idxs = torch.topk(scores_t, k=k, largest=False)
        else:
            # Largest cosine similarity is most similar
            vals, idxs = torch.topk(scores_t, k=k, largest=True)

        self.gpu.synchronize()
        return idxs.cpu().numpy(), vals.cpu().numpy()

    def query_embedding_top_k(
        self,
        query: np.ndarray,
        autoencoder: torch.nn.Module,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Execute similarity search in the learned latent embedding space of Phase 4 Autoencoder.
        """
        autoencoder = autoencoder.to(self.gpu.torch_device)
        autoencoder.eval()

        with torch.no_grad():
            q_t = torch.from_numpy(query.reshape(1, -1)).to(self.gpu.torch_device)
            z_q = autoencoder.encode(q_t).cpu().numpy().flatten()

            # Encode all records in batches
            batch_size = 4096
            z_list = []
            for start in range(0, self.num_records, batch_size):
                end = min(start + batch_size, self.num_records)
                x_b = torch.from_numpy(self.feature_matrix[start:end]).to(self.gpu.torch_device)
                z_b = autoencoder.encode(x_b)
                z_list.append(z_b.cpu().numpy())
            self.gpu.synchronize()
            Z_all = np.vstack(z_list)

        # Query in embedding space
        engine_latent = AMRSimilarityEngine(Z_all, metric="euclidean")
        return engine_latent.query_top_k(z_q, k=k, metric="euclidean")

    def compute_batched_all_pairs(
        self,
        batch_size: int = 2048,
        metric: str = "euclidean"
    ) -> np.ndarray:
        """
        Tiled batched all-pairs similarity matrix computation for memory safety (Report Section 16 & 28).
        Calculates pairwise distance matrix D in [N x N] tiles on GPU MPS.
        """
        N = self.num_records
        D_matrix = np.zeros((N, N), dtype=np.float32)
        X_t = torch.from_numpy(self.feature_matrix).to(self.gpu.torch_device)

        logger.info(f"Computing batched all-pairs similarity ({N} records, tile_size={batch_size})...")

        with torch.no_grad():
            for i in range(0, N, batch_size):
                i_end = min(i + batch_size, N)
                tile_i = X_t[i:i_end]

                for j in range(0, N, batch_size):
                    j_end = min(j + batch_size, N)
                    tile_j = X_t[j:j_end]

                    if metric == "euclidean":
                        # Pairwise tile distance
                        diff = tile_i.unsqueeze(1) - tile_j.unsqueeze(0)  # [B_i, B_j, D]
                        tile_dist = torch.sqrt(torch.sum(diff * diff, dim=2))
                    else:
                        norm_i = torch.norm(tile_i, p=2, dim=1, keepdim=True).clamp(min=1e-7)
                        norm_j = torch.norm(tile_j, p=2, dim=1, keepdim=True).clamp(min=1e-7)
                        tile_dist = torch.matmul(tile_i, tile_j.t()) / (norm_i * norm_j.t())

                    D_matrix[i:i_end, j:j_end] = tile_dist.cpu().numpy()

        self.gpu.synchronize()
        return D_matrix


def format_worked_example(
    query_idx: int,
    top_k_indices: np.ndarray,
    top_k_distances: np.ndarray,
    metadata_df: pd.DataFrame,
    feature_names: List[str],
    feature_matrix: np.ndarray,
    target_names: List[str],
    Y_labels: np.ndarray
) -> pd.DataFrame:
    """
    Construct detailed worked example table for similarity retrieval.
    """
    rows = []
    q_genes = [feature_names[i].replace("gene_", "") for i, v in enumerate(feature_matrix[query_idx]) if v > 0 and feature_names[i].startswith("gene_")]

    for rank, (idx, dist) in enumerate(zip(top_k_indices, top_k_distances)):
        is_query = (idx == query_idx)
        meta = metadata_df.iloc[idx]
        iso_id = meta.get("isolate_id", f"iso_{idx}")
        country = meta.get("country", "Unknown")
        year = meta.get("year", "N/A")
        source = meta.get("source", "Unknown")

        # Extract active resistance determinants
        r_genes = [feature_names[i].replace("gene_", "") for i, v in enumerate(feature_matrix[idx]) if v > 0 and feature_names[i].startswith("gene_")]
        shared_genes = sorted(list(set(q_genes).intersection(set(r_genes))))

        # Antibiotic resistance status
        res_drugs = [target_names[d] for d in range(len(target_names)) if Y_labels[idx, d] == 1.0]

        rows.append({
            "Rank": "Query (Self)" if is_query else f"Top-{rank}",
            "Isolate ID": iso_id,
            "Distance": f"{dist:.4f}",
            "Year": year,
            "Country": country,
            "Source": source,
            "Shared AMR Determinants": ", ".join(shared_genes[:5]) + (f" (+{len(shared_genes)-5})" if len(shared_genes) > 5 else ""),
            "Resistant Drugs": ", ".join(res_drugs[:4]) if res_drugs else "None / Susceptible"
        })

    return pd.DataFrame(rows)
