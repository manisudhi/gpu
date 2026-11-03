"""
src/amr/metrics.py
Comprehensive evaluation metrics, calibration curves, confusion matrices,
and statistical reporting for AMR resistance prediction models.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve

logger = logging.getLogger("amr.metrics")


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive clinical and machine learning metrics for binary resistance.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    # Basic counts
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": 0.5,
        "pr_auc": 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

    # Only compute AUC if both classes exist in y_true
    if len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = 0.5
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            metrics["pr_auc"] = 0.0

    return metrics


def plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str],
    output_path: Path
):
    """Plot multi-target grid of confusion matrices."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_targets = len(target_names)
    cols = 3
    rows = int(np.ceil(num_targets / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = np.array(axes).reshape(-1)

    for i, target in enumerate(target_names):
        ax = axes[i]
        cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["S (0)", "R (1)"],
            yticklabels=["S (0)", "R (1)"]
        )
        ax.set_title(f"{target.capitalize()}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True Ground Truth")

    for j in range(num_targets, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved confusion matrices to {output_path}")


def plot_calibration_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    target_names: List[str],
    output_path: Path
):
    """Plot reliability calibration curves for model probability calibration."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    for i, target in enumerate(target_names):
        try:
            prob_true, prob_pred = calibration_curve(y_true[:, i], y_probs[:, i], n_bins=8, strategy="uniform")
            ax.plot(prob_pred, prob_true, marker="o", label=target.capitalize())
        except Exception as e:
            logger.debug(f"Could not compute calibration for {target}: {e}")

    ax.set_xlabel("Mean Predicted Resistance Probability")
    ax.set_ylabel("Observed Fraction of Resistant Isolates")
    ax.set_title("Probability Calibration Curves (Reliability Diagram)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved calibration curves to {output_path}")


def format_metrics_table(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Format evaluation metrics into a pandas DataFrame."""
    df = pd.DataFrame(results_list)
    return df
