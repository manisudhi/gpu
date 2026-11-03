"""
tests/test_kernels.py
Numerical tolerance validation of Metal compute shaders and PyTorch MPS routines
against CPU NumPy / scikit-learn reference implementations on subsamples.
"""

import pytest
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from src.amr.device import get_gpu_manager
from src.amr.kernels.wrappers import (
    gpu_compute_similarity,
    gpu_kmeans_assign,
    gpu_anomaly_scores,
)

TOLERANCE = 1e-4


@pytest.fixture
def sample_data():
    np.random.seed(42)
    N = 250
    D = 64
    K = 5
    X = np.random.randn(N, D).astype(np.float32)
    q = np.random.randn(D).astype(np.float32)
    centroids = np.random.randn(K, D).astype(np.float32)
    assignments = np.random.randint(0, K, size=(N,)).astype(np.uint32)
    return {"X": X, "q": q, "centroids": centroids, "assignments": assignments, "N": N, "D": D, "K": K}


def test_gpu_device_initialization():
    gpu = get_gpu_manager()
    assert gpu.torch_device.type == "mps", "Torch device must be mps"
    hw = gpu.get_hardware_info()
    assert "Apple" in hw["chip"] or "arm64" in hw["machine"], "Must run on Apple Silicon"


def test_similarity_euclidean_validation(sample_data):
    X = sample_data["X"]
    q = sample_data["q"]

    # CPU reference (NumPy)
    cpu_ref = np.linalg.norm(X - q, axis=1)

    # GPU computation
    gpu_out = gpu_compute_similarity(q, X, metric="euclidean")

    # Correctness check: numerical agreement within tolerance
    max_diff = np.max(np.abs(gpu_out - cpu_ref))
    assert max_diff < TOLERANCE, f"Euclidean similarity error {max_diff} exceeded tolerance {TOLERANCE}"


def test_similarity_cosine_validation(sample_data):
    X = sample_data["X"]
    q = sample_data["q"]

    # CPU reference (scikit-learn cosine similarity)
    # Cosine distance = 1 - cosine_similarity
    cpu_sim = 1.0 - pairwise_distances(X, q.reshape(1, -1), metric="cosine").flatten()

    # GPU computation
    gpu_out = gpu_compute_similarity(q, X, metric="cosine")

    max_diff = np.max(np.abs(gpu_out - cpu_sim))
    assert max_diff < TOLERANCE, f"Cosine similarity error {max_diff} exceeded tolerance {TOLERANCE}"


def test_kmeans_assign_validation(sample_data):
    X = sample_data["X"]
    centroids = sample_data["centroids"]

    # CPU reference: pairwise squared distances
    diffs = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # [N, K, D]
    cpu_sq_dists = np.sum(diffs * diffs, axis=2)  # [N, K]
    cpu_assigns = np.argmin(cpu_sq_dists, axis=1).astype(np.uint32)
    cpu_min_sq = np.min(cpu_sq_dists, axis=1)

    # GPU computation
    gpu_assigns, gpu_min_sq = gpu_kmeans_assign(X, centroids)

    assert np.array_equal(gpu_assigns, cpu_assigns), "K-Means cluster assignments mismatch against CPU reference"
    max_sq_diff = np.max(np.abs(gpu_min_sq - cpu_min_sq))
    assert max_sq_diff < TOLERANCE, f"K-Means min squared distance error {max_sq_diff} exceeded tolerance {TOLERANCE}"


def test_anomaly_score_validation(sample_data):
    X = sample_data["X"]
    centroids = sample_data["centroids"]
    assignments = sample_data["assignments"]

    # CPU reference
    assigned_centroids = centroids[assignments]
    cpu_scores = np.linalg.norm(X - assigned_centroids, axis=1)

    # GPU computation
    gpu_scores = gpu_anomaly_scores(X, centroids, assignments)

    max_diff = np.max(np.abs(gpu_scores - cpu_scores))
    assert max_diff < TOLERANCE, f"Anomaly score error {max_diff} exceeded tolerance {TOLERANCE}"
