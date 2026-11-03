"""
tests/test_clustering.py
Tests for GPU K-Means convergence, assignment validity, and deterministic reruns.
"""

import pytest
import numpy as np

from src.amr.clustering import GPUKMeans


def test_kmeans_convergence_and_determinism():
    np.random.seed(42)
    N = 200
    D = 16
    K = 4
    X = np.random.randn(N, D).astype(np.float32)

    model1 = GPUKMeans(n_clusters=K, max_iter=50, seed=42)
    model1.fit(X)

    assert model1.labels_ is not None
    assert len(model1.labels_) == N
    assert set(model1.labels_).issubset(set(range(K)))
    assert model1.n_iter_ > 0
    assert model1.inertia_ > 0.0

    # Deterministic rerun check
    model2 = GPUKMeans(n_clusters=K, max_iter=50, seed=42)
    model2.fit(X)

    assert np.array_equal(model1.labels_, model2.labels_), "K-Means reruns with same seed must be deterministic"
    assert np.allclose(model1.centroids, model2.centroids, atol=1e-5), "Centroids must match exactly on rerun"
