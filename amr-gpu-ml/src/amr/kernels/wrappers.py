"""
src/amr/kernels/wrappers.py
Python dispatch wrappers for Apple Metal Shading Language compute kernels,
featuring MLX JIT execution and equivalent pure-tensor PyTorch MPS fallbacks.
"""

import os
import math
import logging
from pathlib import Path
from typing import Tuple, Optional, Union

import numpy as np
import torch

from src.amr.device import get_gpu_manager, MLX_AVAILABLE

if MLX_AVAILABLE:
    import mlx.core as mx

logger = logging.getLogger("amr.kernels")

THREADGROUP_SIZE = 256


class MetalKernels:
    """Manages compiled Apple Metal compute shaders."""

    def __init__(self):
        self.kernels_dir = Path(__file__).parent
        self._similarity_kernel = None
        self._kmeans_kernel = None
        self._anomaly_kernel = None
        self._init_mlx_kernels()

    def _init_mlx_kernels(self):
        if not MLX_AVAILABLE:
            return

        # 1. Similarity Kernel (Euclidean)
        sim_src = """
            uint gid = thread_position_in_grid.x;
            if (gid >= num_records) {
                return;
            }
            uint row_offset = gid * num_features;
            float sum_sq = 0.0f;
            for (uint f = 0; f < num_features; ++f) {
                float diff = query[f] - feature_matrix[row_offset + f];
                sum_sq += diff * diff;
            }
            out_distances[gid] = sqrt(sum_sq);
        """
        try:
            self._similarity_kernel = mx.fast.metal_kernel(
                name="amr_similarity_euclidean",
                input_names=["query", "feature_matrix", "num_records", "num_features"],
                output_names=["out_distances"],
                source=sim_src,
                header="""
                    #include <metal_stdlib>
                    using namespace metal;
                """
            )
        except Exception as e:
            logger.warning(f"Could not compile MLX similarity kernel: {e}")

        # 2. K-Means Assignment Kernel
        kmeans_src = """
            uint gid = thread_position_in_grid.x;
            if (gid >= num_records) {
                return;
            }
            uint record_offset = gid * num_features;
            uint best_cluster = 0;
            float min_dist_sq = 1e30f;

            for (uint k = 0; k < num_clusters; ++k) {
                uint centroid_offset = k * num_features;
                float current_dist_sq = 0.0f;
                for (uint f = 0; f < num_features; ++f) {
                    float diff = feature_matrix[record_offset + f] - centroids[centroid_offset + f];
                    current_dist_sq += diff * diff;
                }
                if (current_dist_sq < min_dist_sq) {
                    min_dist_sq = current_dist_sq;
                    best_cluster = k;
                }
            }
            cluster_assignments[gid] = best_cluster;
            min_sq_distances[gid] = min_dist_sq;
        """
        try:
            self._kmeans_kernel = mx.fast.metal_kernel(
                name="amr_kmeans_assign",
                input_names=["feature_matrix", "centroids", "num_records", "num_features", "num_clusters"],
                output_names=["cluster_assignments", "min_sq_distances"],
                source=kmeans_src,
                header="""
                    #include <metal_stdlib>
                    using namespace metal;
                """
            )
        except Exception as e:
            logger.warning(f"Could not compile MLX K-means assignment kernel: {e}")

        # 3. Anomaly Score Kernel
        anomaly_src = """
            uint gid = thread_position_in_grid.x;
            if (gid >= num_records) {
                return;
            }
            uint assigned_k = cluster_assignments[gid];
            uint record_offset = gid * num_features;
            uint centroid_offset = assigned_k * num_features;

            float sum_sq = 0.0f;
            for (uint f = 0; f < num_features; ++f) {
                float diff = feature_matrix[record_offset + f] - centroids[centroid_offset + f];
                sum_sq += diff * diff;
            }
            out_anomaly_scores[gid] = sqrt(sum_sq);
        """
        try:
            self._anomaly_kernel = mx.fast.metal_kernel(
                name="amr_anomaly_score",
                input_names=["feature_matrix", "centroids", "cluster_assignments", "num_records", "num_features"],
                output_names=["out_anomaly_scores"],
                source=anomaly_src,
                header="""
                    #include <metal_stdlib>
                    using namespace metal;
                """
            )
        except Exception as e:
            logger.warning(f"Could not compile MLX anomaly kernel: {e}")


_KERNELS_INSTANCE = None

def get_metal_kernels() -> MetalKernels:
    global _KERNELS_INSTANCE
    if _KERNELS_INSTANCE is None:
        _KERNELS_INSTANCE = MetalKernels()
    return _KERNELS_INSTANCE


# =====================================================================
# 1. Similarity Kernel Dispatch
# =====================================================================

def gpu_compute_similarity(
    query: np.ndarray,
    feature_matrix: np.ndarray,
    metric: str = "euclidean",
    use_metal_kernel: bool = True
) -> np.ndarray:
    """
    Compute distance / similarity between a single query vector and all records on the GPU.
    Returns: float32 numpy array of shape [N].
    """
    num_records, num_features = feature_matrix.shape
    query = np.ascontiguousarray(query.flatten(), dtype=np.float32)
    feature_matrix = np.ascontiguousarray(feature_matrix, dtype=np.float32)

    kernels = get_metal_kernels()
    if use_metal_kernel and MLX_AVAILABLE and kernels._similarity_kernel is not None and metric == "euclidean":
        try:
            q_mx = mx.array(query)
            x_mx = mx.array(feature_matrix)
            grid = (num_records, 1, 1)
            threadgroup = (min(THREADGROUP_SIZE, num_records), 1, 1)

            out = kernels._similarity_kernel(
                inputs=[
                    q_mx,
                    x_mx,
                    mx.array(num_records, dtype=mx.uint32),
                    mx.array(num_features, dtype=mx.uint32),
                ],
                template=[],
                grid=grid,
                threadgroup=threadgroup,
                output_shapes=[(num_records,)],
                output_dtypes=[mx.float32],
            )
            mx.eval(out)
            return np.array(out[0], copy=False)
        except Exception as e:
            logger.debug(f"Metal kernel dispatch exception ({e}), falling back to PyTorch MPS.")

    # Pure-Tensor PyTorch MPS Implementation
    gpu = get_gpu_manager()
    with torch.no_grad():
        q_t = torch.from_numpy(query).to(gpu.torch_device)
        x_t = torch.from_numpy(feature_matrix).to(gpu.torch_device)

        if metric == "euclidean":
            # ||q - x_i||_2
            diff = x_t - q_t.unsqueeze(0)
            dist_t = torch.sqrt(torch.sum(diff * diff, dim=1))
            gpu.synchronize()
            return dist_t.cpu().numpy()
        elif metric == "cosine":
            # Cosine similarity: (q . x) / (||q|| * ||x||)
            q_norm = torch.norm(q_t, p=2).clamp(min=1e-7)
            x_norm = torch.norm(x_t, p=2, dim=1).clamp(min=1e-7)
            dot = torch.matmul(x_t, q_t)
            sim_t = dot / (q_norm * x_norm)
            gpu.synchronize()
            return sim_t.cpu().numpy()
        else:
            raise ValueError(f"Unsupported metric: {metric}")


# =====================================================================
# 2. K-Means Assignment Dispatch
# =====================================================================

def gpu_kmeans_assign(
    feature_matrix: np.ndarray,
    centroids: np.ndarray,
    use_metal_kernel: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute nearest-centroid assignment for all records on the GPU.
    Returns:
        assignments: uint32 array of shape [N]
        min_sq_distances: float32 array of shape [N]
    """
    num_records, num_features = feature_matrix.shape
    num_clusters = centroids.shape[0]

    feature_matrix = np.ascontiguousarray(feature_matrix, dtype=np.float32)
    centroids = np.ascontiguousarray(centroids, dtype=np.float32)

    kernels = get_metal_kernels()
    if use_metal_kernel and MLX_AVAILABLE and kernels._kmeans_kernel is not None:
        try:
            x_mx = mx.array(feature_matrix)
            c_mx = mx.array(centroids)
            grid = (num_records, 1, 1)
            threadgroup = (min(THREADGROUP_SIZE, num_records), 1, 1)

            out = kernels._kmeans_kernel(
                inputs=[
                    x_mx,
                    c_mx,
                    mx.array(num_records, dtype=mx.uint32),
                    mx.array(num_features, dtype=mx.uint32),
                    mx.array(num_clusters, dtype=mx.uint32),
                ],
                template=[],
                grid=grid,
                threadgroup=threadgroup,
                output_shapes=[(num_records,), (num_records,)],
                output_dtypes=[mx.uint32, mx.float32],
            )
            mx.eval(out)
            return np.array(out[0], copy=False), np.array(out[1], copy=False)
        except Exception as e:
            logger.debug(f"Metal K-means dispatch exception ({e}), falling back to PyTorch MPS.")

    # Pure-Tensor PyTorch MPS Implementation
    gpu = get_gpu_manager()
    with torch.no_grad():
        x_t = torch.from_numpy(feature_matrix).to(gpu.torch_device)
        c_t = torch.from_numpy(centroids).to(gpu.torch_device)

        # Pairwise squared Euclidean distance: ||x - c||^2 = ||x||^2 - 2 x c^T + ||c||^2
        x_sq = torch.sum(x_t * x_t, dim=1, keepdim=True)  # [N, 1]
        c_sq = torch.sum(c_t * c_t, dim=1, keepdim=True)  # [K, 1]
        dists = x_sq - 2.0 * torch.matmul(x_t, c_t.t()) + c_sq.t()  # [N, K]
        dists = torch.clamp(dists, min=0.0)

        min_dists, assignments = torch.min(dists, dim=1)
        gpu.synchronize()
        return assignments.cpu().numpy().astype(np.uint32), min_dists.cpu().numpy()


# =====================================================================
# 3. Anomaly Score Dispatch
# =====================================================================

def gpu_anomaly_scores(
    feature_matrix: np.ndarray,
    centroids: np.ndarray,
    cluster_assignments: np.ndarray,
    use_metal_kernel: bool = True
) -> np.ndarray:
    """
    Compute distance-to-assigned-centroid anomaly scores on the GPU.
    Returns: float32 array of shape [N]
    """
    num_records, num_features = feature_matrix.shape
    feature_matrix = np.ascontiguousarray(feature_matrix, dtype=np.float32)
    centroids = np.ascontiguousarray(centroids, dtype=np.float32)
    cluster_assignments = np.ascontiguousarray(cluster_assignments, dtype=np.uint32)

    kernels = get_metal_kernels()
    if use_metal_kernel and MLX_AVAILABLE and kernels._anomaly_kernel is not None:
        try:
            x_mx = mx.array(feature_matrix)
            c_mx = mx.array(centroids)
            a_mx = mx.array(cluster_assignments)
            grid = (num_records, 1, 1)
            threadgroup = (min(THREADGROUP_SIZE, num_records), 1, 1)

            out = kernels._anomaly_kernel(
                inputs=[
                    x_mx,
                    c_mx,
                    a_mx,
                    mx.array(num_records, dtype=mx.uint32),
                    mx.array(num_features, dtype=mx.uint32),
                ],
                template=[],
                grid=grid,
                threadgroup=threadgroup,
                output_shapes=[(num_records,)],
                output_dtypes=[mx.float32],
            )
            mx.eval(out)
            return np.array(out[0], copy=False)
        except Exception as e:
            logger.debug(f"Metal anomaly score dispatch exception ({e}), falling back to PyTorch MPS.")

    # Pure-Tensor PyTorch MPS Implementation
    gpu = get_gpu_manager()
    with torch.no_grad():
        x_t = torch.from_numpy(feature_matrix).to(gpu.torch_device)
        c_t = torch.from_numpy(centroids).to(gpu.torch_device)
        a_t = torch.from_numpy(cluster_assignments.astype(np.int64)).to(gpu.torch_device)

        assigned_c = c_t[a_t]  # [N, D]
        diff = x_t - assigned_c
        scores = torch.sqrt(torch.sum(diff * diff, dim=1))
        gpu.synchronize()
        return scores.cpu().numpy()
