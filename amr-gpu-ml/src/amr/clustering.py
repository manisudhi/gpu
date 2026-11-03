"""
src/amr/clustering.py
GPU-accelerated K-Means clustering using Apple Metal compute kernels for assignment,
GPU reduction for centroid updates, convergence handling, and cluster interpretation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score

from src.amr.device import get_gpu_manager
from src.amr.kernels.wrappers import gpu_kmeans_assign

logger = logging.getLogger("amr.clustering")


class GPUKMeans:
    """
    K-Means clustering executed on Apple Silicon GPU.
    - Nearest-centroid assignment: Metal compute shader (`kmeans_assign.metal`) or MPS.
    - Centroid update reduction: GPU PyTorch scatter-add.
    - Initialization: K-means++ on GPU.
    """

    def __init__(
        self,
        n_clusters: int = 5,
        max_iter: int = 100,
        tol: float = 1e-4,
        seed: int = 42,
        use_metal_kernel: bool = True
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.use_metal_kernel = use_metal_kernel
        self.gpu = get_gpu_manager()

        self.centroids: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = float("inf")
        self.n_iter_: int = 0

    def _init_kmeans_plus_plus(self, X: np.ndarray) -> np.ndarray:
        """GPU-accelerated K-means++ centroid initialization."""
        np.random.seed(self.seed)
        N, D = X.shape
        centroids = np.zeros((self.n_clusters, D), dtype=np.float32)

        # First centroid chosen uniformly at random
        first_idx = np.random.randint(0, N)
        centroids[0] = X[first_idx]

        X_t = torch.from_numpy(X).to(self.gpu.torch_device)
        curr_dists = torch.full((N,), float("inf"), device=self.gpu.torch_device)

        for k in range(1, self.n_clusters):
            # Compute distance to last added centroid
            c_t = torch.from_numpy(centroids[k - 1]).to(self.gpu.torch_device)
            diff = X_t - c_t.unsqueeze(0)
            dist_k = torch.sum(diff * diff, dim=1)
            curr_dists = torch.minimum(curr_dists, dist_k)

            # Sample next centroid proportional to squared distance
            probs = (curr_dists / (curr_dists.sum() + 1e-7)).cpu().numpy()
            probs = np.clip(probs, 0.0, None)
            probs /= probs.sum()
            next_idx = np.random.choice(N, p=probs)
            centroids[k] = X[next_idx]

        self.gpu.synchronize()
        return centroids

    def fit(self, X: np.ndarray) -> "GPUKMeans":
        """Fit K-Means clustering on GPU feature matrix."""
        N, D = X.shape
        X = np.ascontiguousarray(X, dtype=np.float32)

        self.centroids = self._init_kmeans_plus_plus(X)
        prev_inertia = float("inf")

        for iteration in range(1, self.max_iter + 1):
            self.n_iter_ = iteration

            # 1. GPU Assignment Step (Metal Kernel or MPS)
            assignments, min_sq_dists = gpu_kmeans_assign(
                feature_matrix=X,
                centroids=self.centroids,
                use_metal_kernel=self.use_metal_kernel
            )

            # Current inertia
            current_inertia = float(np.sum(min_sq_dists))

            # 2. GPU Reduction for Centroid Updates
            X_t = torch.from_numpy(X).to(self.gpu.torch_device)
            assign_t = torch.from_numpy(assignments.astype(np.int64)).to(self.gpu.torch_device)

            # Accumulate feature sums and cluster counts per cluster
            cluster_sums = torch.zeros((self.n_clusters, D), dtype=torch.float32, device=self.gpu.torch_device)
            cluster_counts = torch.zeros((self.n_clusters, 1), dtype=torch.float32, device=self.gpu.torch_device)

            ones = torch.ones((N, 1), dtype=torch.float32, device=self.gpu.torch_device)
            cluster_sums.index_add_(0, assign_t, X_t)
            cluster_counts.index_add_(0, assign_t, ones)

            self.gpu.synchronize()
            counts_np = cluster_counts.cpu().numpy().flatten()
            sums_np = cluster_sums.cpu().numpy()

            # 3. Handle Empty Clusters & Update
            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                if counts_np[k] > 0:
                    new_centroids[k] = sums_np[k] / counts_np[k]
                else:
                    # Reseed empty cluster with the record with max distance
                    farthest_idx = np.argmax(min_sq_dists)
                    new_centroids[k] = X[farthest_idx]
                    logger.warning(f"K-Means iteration {iteration}: cluster {k} was empty; reseeded with isolate {farthest_idx}")

            # 4. Check Convergence
            centroid_shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            self.labels_ = assignments
            self.inertia_ = current_inertia

            if centroid_shift < self.tol or abs(prev_inertia - current_inertia) < self.tol:
                logger.info(f"K-Means converged at iteration {iteration} (shift={centroid_shift:.2e}, inertia={self.inertia_:.2f})")
                break
            prev_inertia = current_inertia

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign records to nearest centroids on GPU."""
        assignments, _ = gpu_kmeans_assign(X, self.centroids, use_metal_kernel=self.use_metal_kernel)
        return assignments


def evaluate_cluster_k_range(
    X: np.ndarray,
    k_values: List[int] = [2, 3, 4, 5, 6, 8, 10],
    subsample_size: int = 5000,
    seed: int = 42
) -> Tuple[pd.DataFrame, int]:
    """
    Evaluate K selection via Elbow (Inertia), Silhouette Score, and Davies-Bouldin index.
    Returns:
        metrics_df: summary DataFrame across all K
        best_k: suggested K based on silhouette and elbow balance
    """
    np.random.seed(seed)
    N = len(X)
    sub_idx = np.random.choice(N, size=min(subsample_size, N), replace=False)
    X_sub = X[sub_idx]

    records = []
    best_sil = -1.0
    suggested_k = k_values[0]

    for k in k_values:
        model = GPUKMeans(n_clusters=k, seed=seed)
        model.fit(X)
        labels_sub = model.predict(X_sub)

        sil = float(silhouette_score(X_sub, labels_sub))
        db = float(davies_bouldin_score(X_sub, labels_sub))

        records.append({
            "K": k,
            "Inertia": model.inertia_,
            "Silhouette_Score": sil,
            "Davies_Bouldin_Index": db,
            "Iterations_to_Converge": model.n_iter_
        })
        if sil > best_sil:
            best_sil = sil
            suggested_k = k

    df = pd.DataFrame(records)
    logger.info(f"K evaluation completed. Recommended K={suggested_k} (Silhouette={best_sil:.4f})")
    return df, suggested_k


def plot_k_selection_metrics(metrics_df: pd.DataFrame, output_path: Path):
    """Plot Elbow curve, Silhouette, and Davies-Bouldin scores across K."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

    # Elbow (Inertia)
    ax1.plot(metrics_df["K"], metrics_df["Inertia"], marker="o", color="#1f77b4", linewidth=2)
    ax1.set_title("Elbow Method (Inertia)")
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Sum of Squared Distances (Inertia)")

    # Silhouette Score
    ax2.plot(metrics_df["K"], metrics_df["Silhouette_Score"], marker="s", color="#2ca02c", linewidth=2)
    ax2.set_title("Silhouette Score (Higher is Better)")
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Mean Silhouette Coefficient")

    # Davies-Bouldin Index
    ax3.plot(metrics_df["K"], metrics_df["Davies_Bouldin_Index"], marker="^", color="#d62728", linewidth=2)
    ax3.set_title("Davies-Bouldin Index (Lower is Better)")
    ax3.set_xlabel("Number of Clusters (K)")
    ax3.set_ylabel("Davies-Bouldin Index")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved K selection curves to {output_path}")


def gpu_pca_projection(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Compute PCA dimensionality reduction using PyTorch MPS GPU SVD."""
    gpu = get_gpu_manager()
    X_t = torch.from_numpy(X).to(gpu.torch_device)
    mean_t = torch.mean(X_t, dim=0, keepdim=True)
    X_centered = X_t - mean_t

    # SVD on GPU: X = U S V^T -> projection = X_centered @ V[:, :n_components]
    _, _, V = torch.linalg.svd(X_centered, full_matrices=False)
    proj_t = torch.matmul(X_centered, V[:n_components, :].t())
    gpu.synchronize()
    return proj_t.cpu().numpy()


def plot_cluster_pca(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    output_path: Path
):
    """Plot 2D GPU PCA projection of clusters and centroids."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Project data and centroids together
    all_data = np.vstack([X, centroids])
    all_proj = gpu_pca_projection(all_data, n_components=2)
    N = len(X)
    X_proj = all_proj[:N]
    c_proj = all_proj[N:]

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        X_proj[:, 0],
        X_proj[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.6,
        s=18,
        edgecolors="none"
    )
    # Plot centroids
    ax.scatter(
        c_proj[:, 0],
        c_proj[:, 1],
        c="black",
        marker="X",
        s=160,
        edgecolors="white",
        linewidths=1.5,
        label="Centroids"
    )
    ax.set_title("GPU PCA Projection of AMR Clusters (Apple Metal K-Means)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    fig.colorbar(scatter, ax=ax, label="Cluster Assignment")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved cluster PCA plot to {output_path}")


def profile_clusters(
    labels: np.ndarray,
    metadata_df: pd.DataFrame,
    feature_matrix: np.ndarray,
    feature_names: List[str],
    mdr_labels: np.ndarray
) -> pd.DataFrame:
    """Interpret clusters by size, MDR rate, dominant genes, and top country/sources."""
    records = []
    unique_labels = sorted(list(np.unique(labels)))

    for k in unique_labels:
        mask = (labels == k)
        cluster_size = int(mask.sum())
        mdr_rate = float(mdr_labels[mask].mean() * 100) if len(mdr_labels) == len(labels) else 0.0

        # Dominant genes
        cluster_X = feature_matrix[mask]
        gene_means = cluster_X.mean(axis=0)
        gene_indices = [i for i, name in enumerate(feature_names) if name.startswith("gene_")]
        top_gene_idx = sorted(gene_indices, key=lambda idx: gene_means[idx], reverse=True)[:5]
        top_genes = [f"{feature_names[i].replace('gene_', '')} ({gene_means[i]*100:.1f}%)" for i in top_gene_idx if gene_means[i] > 0.05]

        # Top source & country
        cluster_meta = metadata_df.iloc[mask]
        top_source = cluster_meta["source"].value_counts().index[0] if len(cluster_meta) > 0 else "N/A"
        top_country = cluster_meta["country"].value_counts().index[0] if len(cluster_meta) > 0 else "N/A"

        records.append({
            "Cluster": f"Cluster {k}",
            "Isolates_Count": cluster_size,
            "MDR_Rate_Pct": f"{mdr_rate:.1f}%",
            "Top_Source": top_source,
            "Top_Country": top_country,
            "Dominant_Resistance_Genes": ", ".join(top_genes) if top_genes else "None dominant"
        })

    return pd.DataFrame(records)
