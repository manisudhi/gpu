"""
tests/test_models.py
Verification of PyTorch MPS model architectures, forward passes, and training convergence.
"""

import pytest
import numpy as np
import torch

from src.amr.device import get_gpu_manager
from src.amr.models import (
    LogisticRegressionModel,
    MLPClassifier,
    DeepMLPClassifier,
    MultiTaskAMRNet,
    AMRAutoencoder,
    train_classifier,
    train_autoencoder,
)


@pytest.fixture
def dummy_train_data():
    np.random.seed(42)
    N = 100
    D = 32
    T = 4
    X = np.random.randn(N, D).astype(np.float32)
    Y = (np.random.rand(N, T) > 0.5).astype(np.float32)
    Y_mask = (np.random.rand(N, T) > 0.2).astype(bool)
    return {"X": X, "Y": Y, "Y_mask": Y_mask, "N": N, "D": D, "T": T}


def test_models_forward_pass_on_mps(dummy_train_data):
    gpu = get_gpu_manager()
    X = torch.from_numpy(dummy_train_data["X"]).to(gpu.torch_device)
    D = dummy_train_data["D"]
    T = dummy_train_data["T"]

    # 1. Logistic Regression
    lr_model = LogisticRegressionModel(D, T).to(gpu.torch_device)
    out_lr = lr_model(X)
    assert out_lr.shape == (dummy_train_data["N"], T)

    # 2. MLP
    mlp_model = MLPClassifier(D, [64, 32], T).to(gpu.torch_device)
    out_mlp = mlp_model(X)
    assert out_mlp.shape == (dummy_train_data["N"], T)

    # 3. Deep MLP
    deep_mlp = DeepMLPClassifier(D, [64, 32], T).to(gpu.torch_device)
    out_deep = deep_mlp(X)
    assert out_deep.shape == (dummy_train_data["N"], T)

    # 4. Multi-Task Net
    mt_model = MultiTaskAMRNet(D, T, shared_dim=64).to(gpu.torch_device)
    out_ab, out_mdr = mt_model(X)
    assert out_ab.shape == (dummy_train_data["N"], T)
    assert out_mdr.shape == (dummy_train_data["N"], 1)

    # 5. Autoencoder
    ae_model = AMRAutoencoder(D, latent_dim=8).to(gpu.torch_device)
    recon, z = ae_model(X)
    assert recon.shape == (dummy_train_data["N"], D)
    assert z.shape == (dummy_train_data["N"], 8)


def test_classifier_training_convergence(dummy_train_data):
    X = dummy_train_data["X"]
    Y = dummy_train_data["Y"]
    Y_mask = dummy_train_data["Y_mask"]
    D = dummy_train_data["D"]
    T = dummy_train_data["T"]

    model = MLPClassifier(D, [32], T)
    trained_model, history = train_classifier(
        model,
        X_train=X[:80],
        Y_train=Y[:80],
        Y_mask_train=Y_mask[:80],
        X_val=X[80:],
        Y_val=Y[80:],
        Y_mask_val=Y_mask[80:],
        epochs=5,
        batch_size=32,
    )
    assert len(history["train_loss"]) == 5
    assert history["train_loss"][-1] <= history["train_loss"][0] + 0.1
