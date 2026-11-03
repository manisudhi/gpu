"""
src/amr/models.py
PyTorch GPU (MPS) neural models for antimicrobial resistance prediction,
multi-task learning, and unsupervised representation learning.
"""

import os
import copy
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from src.amr.device import get_gpu_manager

logger = logging.getLogger("amr.models")


# =====================================================================
# Model Architectures
# =====================================================================

class LogisticRegressionModel(nn.Module):
    """(a) Single/Multi-target Logistic Regression baseline on GPU MPS."""
    def __init__(self, in_features: int, out_targets: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPClassifier(nn.Module):
    """(b) Standard Multi-Layer Perceptron classifier."""
    def __init__(self, in_features: int, hidden_dims: List[int], out_targets: int, dropout: float = 0.2):
        super().__init__()
        layers = []
        curr_dim = in_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, out_targets))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepMLPClassifier(nn.Module):
    """(c) Deep MLP with Batch Normalization, Dropout, and Residual-style depth."""
    def __init__(self, in_features: int, hidden_dims: List[int] = [256, 128, 64], out_targets: int = 10, dropout: float = 0.3):
        super().__init__()
        layers = []
        curr_dim = in_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, out_targets))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskAMRNet(nn.Module):
    """(d) Multi-Task Network predicting multiple antibiotics + MDR jointly."""
    def __init__(self, in_features: int, num_antibiotics: int, shared_dim: int = 128):
        super().__init__()
        # Shared trunk representation
        self.shared_trunk = nn.Sequential(
            nn.Linear(in_features, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(shared_dim, shared_dim // 2),
            nn.BatchNorm1d(shared_dim // 2),
            nn.GELU(),
        )
        # Dedicated task heads
        self.antibiotic_head = nn.Linear(shared_dim // 2, num_antibiotics)
        self.mdr_head = nn.Linear(shared_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared_trunk(x)
        ab_logits = self.antibiotic_head(features)
        mdr_logits = self.mdr_head(features)
        return ab_logits, mdr_logits


class AMRAutoencoder(nn.Module):
    """(e) Unsupervised Autoencoder for learned low-dimensional genomic embeddings."""
    def __init__(self, in_features: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, in_features),
            nn.Sigmoid(),  # for reconstruction of multi-hot feature values
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


# =====================================================================
# GPU Training Routines
# =====================================================================

def train_classifier(
    model: nn.Module,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    Y_mask_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    Y_mask_val: np.ndarray,
    epochs: int = 25,
    batch_size: int = 128,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    patience: int = 5,
    pos_weights: Optional[np.ndarray] = None,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train a classification network on Apple Silicon MPS GPU.
    Handles masked targets (untested antibiotics) and class imbalance.
    """
    gpu = get_gpu_manager()
    model = model.to(gpu.torch_device)

    # Class weighting tensor for BCEWithLogitsLoss
    pw_tensor = None
    if pos_weights is not None:
        pw_tensor = torch.from_numpy(pos_weights.astype(np.float32)).to(gpu.torch_device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor, reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    # Dataloader
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(Y_train),
        torch.from_numpy(Y_mask_train)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(Y_val),
        torch.from_numpy(Y_mask_val)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    history = {"train_loss": [], "val_loss": []}
    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_items = 0

        for b_x, b_y, b_mask in train_loader:
            b_x = b_x.to(gpu.torch_device)
            b_y = b_y.to(gpu.torch_device)
            b_mask = b_mask.to(gpu.torch_device)

            optimizer.zero_grad()
            logits = model(b_x)
            loss_raw = criterion(logits, b_y)
            # Mask out untested antibiotics
            loss = (loss_raw * b_mask.float()).sum() / (b_mask.float().sum() + 1e-7)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(b_x)
            train_items += len(b_x)

        gpu.synchronize()
        train_epoch_loss = train_loss_sum / max(train_items, 1)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_items = 0
        with torch.no_grad():
            for b_x, b_y, b_mask in val_loader:
                b_x = b_x.to(gpu.torch_device)
                b_y = b_y.to(gpu.torch_device)
                b_mask = b_mask.to(gpu.torch_device)

                logits = model(b_x)
                loss_raw = criterion(logits, b_y)
                loss = (loss_raw * b_mask.float()).sum() / (b_mask.float().sum() + 1e-7)

                val_loss_sum += loss.item() * len(b_x)
                val_items += len(b_x)

        gpu.synchronize()
        val_epoch_loss = val_loss_sum / max(val_items, 1)
        scheduler.step(val_epoch_loss)

        history["train_loss"].append(train_epoch_loss)
        history["val_loss"].append(val_epoch_loss)

        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def train_multi_task(
    model: MultiTaskAMRNet,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    Y_mask_train: np.ndarray,
    MDR_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    Y_mask_val: np.ndarray,
    MDR_val: np.ndarray,
    epochs: int = 25,
    batch_size: int = 128,
    lr: float = 0.001,
) -> Tuple[MultiTaskAMRNet, Dict[str, List[float]]]:
    """Train Multi-Task network predicting both antibiotic resistance and MDR jointly."""
    gpu = get_gpu_manager()
    model = model.to(gpu.torch_device)

    crit_ab = nn.BCEWithLogitsLoss(reduction="none")
    crit_mdr = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(Y_train),
        torch.from_numpy(Y_mask_train),
        torch.from_numpy(MDR_train.reshape(-1, 1))
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(Y_val),
        torch.from_numpy(Y_mask_val),
        torch.from_numpy(MDR_val.reshape(-1, 1))
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss = 0.0
        for b_x, b_y, b_mask, b_mdr in train_loader:
            b_x, b_y, b_mask, b_mdr = (
                b_x.to(gpu.torch_device),
                b_y.to(gpu.torch_device),
                b_mask.to(gpu.torch_device),
                b_mdr.to(gpu.torch_device),
            )
            optimizer.zero_grad()
            ab_logits, mdr_logits = model(b_x)

            ab_loss_raw = crit_ab(ab_logits, b_y)
            ab_loss = (ab_loss_raw * b_mask.float()).sum() / (b_mask.float().sum() + 1e-7)
            mdr_loss = crit_mdr(mdr_logits, b_mdr)

            total_loss = ab_loss + 0.5 * mdr_loss
            total_loss.backward()
            optimizer.step()
            t_loss += total_loss.item()

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for b_x, b_y, b_mask, b_mdr in val_loader:
                b_x, b_y, b_mask, b_mdr = (
                    b_x.to(gpu.torch_device),
                    b_y.to(gpu.torch_device),
                    b_mask.to(gpu.torch_device),
                    b_mdr.to(gpu.torch_device),
                )
                ab_logits, mdr_logits = model(b_x)
                ab_loss = (crit_ab(ab_logits, b_y) * b_mask.float()).sum() / (b_mask.float().sum() + 1e-7)
                mdr_loss = crit_mdr(mdr_logits, b_mdr)
                v_loss += (ab_loss + 0.5 * mdr_loss).item()

        history["train_loss"].append(t_loss / len(train_loader))
        history["val_loss"].append(v_loss / len(val_loader))

    return model, history


def train_autoencoder(
    model: AMRAutoencoder,
    X_train: np.ndarray,
    X_val: np.ndarray,
    epochs: int = 25,
    batch_size: int = 128,
    lr: float = 0.001,
) -> Tuple[AMRAutoencoder, Dict[str, List[float]]]:
    """Train Autoencoder on MPS GPU for unsupervised representation learning."""
    gpu = get_gpu_manager()
    model = model.to(gpu.torch_device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val)), batch_size=batch_size, shuffle=False)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss = 0.0
        for (b_x,) in train_loader:
            b_x = b_x.to(gpu.torch_device)
            optimizer.zero_grad()
            recon, _ = model(b_x)
            loss = criterion(recon, b_x)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for (b_x,) in val_loader:
                b_x = b_x.to(gpu.torch_device)
                recon, _ = model(b_x)
                v_loss += criterion(recon, b_x).item()

        history["train_loss"].append(t_loss / len(train_loader))
        history["val_loss"].append(v_loss / len(val_loader))

    return model, history


def compute_gene_importance(
    model: nn.Module,
    X_test: np.ndarray,
    feature_names: List[str],
    target_names: List[str],
    top_k: int = 5
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Compute gradient-based attribution on GPU to extract top resistance determinants per drug.
    """
    gpu = get_gpu_manager()
    model = model.to(gpu.torch_device)
    model.eval()

    X_t = torch.from_numpy(X_test).to(gpu.torch_device)
    X_t.requires_grad = True

    logits = model(X_t)  # [N, num_targets]
    attributions = {}

    for t_idx, target in enumerate(target_names):
        if X_t.grad is not None:
            X_t.grad.zero_()
        
        target_logit = logits[:, t_idx].sum()
        target_logit.backward(retain_graph=True)

        # Saliency map: mean absolute gradient per feature
        grads = X_t.grad.abs().mean(dim=0).cpu().numpy()
        top_indices = np.argsort(grads)[::-1]
        
        # Filter to gene features only
        gene_ranked = []
        for idx in top_indices:
            feat = feature_names[idx]
            if feat.startswith("gene_"):
                gene_ranked.append((feat.replace("gene_", ""), float(grads[idx])))
            if len(gene_ranked) >= top_k:
                break
        attributions[target] = gene_ranked

    return attributions
