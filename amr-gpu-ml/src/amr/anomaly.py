"""
src/amr/anomaly.py
GPU-accelerated anomaly detection for AMR profiles using Metal distance-to-centroid
compute kernels and Autoencoder reconstruction error with documented statistical thresholds.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from src.amr.device import get_gpu_manager
from src.amr.kernels.wrappers import gpu_anomaly_scores

logger = logging.getLogger("amr.anomaly")


class AMRAnomalyDetector:
    """
    Identifies atypical resistance profiles and outlier isolates.
    Supports distance-to-centroid scoring (Metal kernel) and autoencoder reconstruction error.
    """

    def __init__(
        self,
        percentile_threshold: float = 99.0,
        mad_multiplier: float = 3.5
    ):
        self.percentile_threshold = percentile_threshold
        self.mad_multiplier = mad_multiplier
        self.gpu = get_gpu_manager()

    def score_distance_to_centroid(
        self,
        X: np.ndarray,
        centroids: np.ndarray,
        assignments: np.ndarray,
        use_metal_kernel: bool = True
    ) -> np.ndarray:
        """Compute Euclidean distance to assigned cluster centroid using Metal kernel."""
        return gpu_anomaly_scores(
            feature_matrix=X,
            centroids=centroids,
            cluster_assignments=assignments,
            use_metal_kernel=use_metal_kernel
        )

    def score_autoencoder_reconstruction(
        self,
        X: np.ndarray,
        autoencoder: torch.nn.Module,
        batch_size: int = 4096
    ) -> np.ndarray:
        """Compute Autoencoder reconstruction error on GPU MPS."""
        autoencoder = autoencoder.to(self.gpu.torch_device)
        autoencoder.eval()

        N = len(X)
        errors = np.zeros(N, dtype=np.float32)

        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                x_b = torch.from_numpy(X[start:end]).to(self.gpu.torch_device)
                recon_b, _ = autoencoder(x_b)
                diff = x_b - recon_b
                b_err = torch.sqrt(torch.sum(diff * diff, dim=1))
                errors[start:end] = b_err.cpu().numpy()

        self.gpu.synchronize()
        return errors

    def compute_statistical_threshold(
        self,
        scores: np.ndarray,
        method: str = "mad"
    ) -> Tuple[float, str]:
        """
        Documented statistical threshold rules (Report Section 18):
        - 'mad': Median + multiplier * (MAD / 0.6745)
        - 'percentile': e.g., 99th percentile
        """
        if method == "mad":
            median = float(np.median(scores))
            mad = float(np.median(np.abs(scores - median)))
            sigma = mad / 0.6745 if mad > 0 else 1.0
            threshold = median + self.mad_multiplier * sigma
            rule_desc = f"MAD Rule (Median={median:.3f} + {self.mad_multiplier}*sigma={threshold:.3f})"
        else:
            threshold = float(np.percentile(scores, self.percentile_threshold))
            rule_desc = f"{self.percentile_threshold}th Percentile Rule (Threshold={threshold:.3f})"

        # If threshold exceeds max score, fallback to 99th percentile
        if threshold >= np.max(scores):
            threshold = float(np.percentile(scores, 99.0))
            rule_desc = f"99th Percentile Rule (Threshold={threshold:.3f})"

        logger.info(f"Anomaly threshold established via {rule_desc}")
        return threshold, rule_desc

    @staticmethod
    def inspect_top_anomalies(
        scores: np.ndarray,
        threshold: float,
        metadata_df: pd.DataFrame,
        feature_matrix: np.ndarray,
        feature_names: List[str],
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Inspect highest scoring anomalous isolates and explain potential epidemiological
        or genomic factors driving their divergence.
        """
        flagged_mask = (scores >= threshold)
        flagged_count = int(flagged_mask.sum())
        logger.info(f"Flagged {flagged_count:,} isolates ({flagged_count/len(scores)*100:.2f}%) above threshold {threshold:.3f}")

        # Sort by anomaly score descending
        sorted_indices = np.argsort(scores)[::-1][:top_n]
        rows = []

        mean_gene_count = np.mean(metadata_df["gene_count"]) if "gene_count" in metadata_df else 5.0

        for rank, idx in enumerate(sorted_indices, 1):
            score = scores[idx]
            meta = metadata_df.iloc[idx]
            iso_id = meta.get("isolate_id", f"iso_{idx}")
            country = meta.get("country", "Unknown")
            year = meta.get("year", "N/A")
            source = meta.get("source", "Unknown")
            gene_cnt = meta.get("gene_count", 0)

            # Active rare genes in this record
            x_row = feature_matrix[idx]
            active_genes = [
                feature_names[i].replace("gene_", "")
                for i, v in enumerate(x_row)
                if v > 0 and feature_names[i].startswith("gene_")
            ]

            # Explanation rationale
            reasons = []
            if gene_cnt > mean_gene_count * 2.0:
                reasons.append(f"Hyper-determinant carriage ({gene_cnt} genes vs mean {mean_gene_count:.1f})")
            if any("blaNDM" in g or "blaKPC" in g or "blaOXA-48" in g for g in active_genes) and any("mcr" in g or "colistin" in g.lower() for g in active_genes):
                reasons.append("Co-carriage of high-level carbapenemase and colistin determinant")
            if source in ["Environmental", "Wound_Tissue"]:
                reasons.append(f"Uncommon ecological reservoir ({source})")
            if not reasons:
                reasons.append("Rare multidrug determinant permutation divergent from centroid")

            rows.append({
                "Rank": rank,
                "Isolate_ID": iso_id,
                "Anomaly_Score": f"{score:.4f}",
                "Year": year,
                "Country": country,
                "Source": source,
                "Gene_Count": gene_cnt,
                "Distinctive_Genotypes": ", ".join(active_genes[:4]) + (f" (+{len(active_genes)-4})" if len(active_genes) > 4 else ""),
                "Analytical_Hypothesis": "; ".join(reasons)
            })

        return pd.DataFrame(rows)


def plot_anomaly_distribution(
    scores: np.ndarray,
    threshold: float,
    rule_desc: str,
    output_path: Path
):
    """Plot distribution of anomaly scores with statistical threshold boundary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))

    sns.histplot(scores, bins=50, kde=True, color="#4c72b0", ax=ax, edgecolor="black")
    ax.axvline(threshold, color="#d62728", linestyle="--", linewidth=2.5, label=f"Threshold ({threshold:.3f})\n{rule_desc}")

    ax.set_title("Distribution of Distance-Based Anomaly Scores (Apple Metal Kernel)")
    ax.set_xlabel("Euclidean Distance to Assigned Cluster Centroid")
    ax.set_ylabel("Isolate Count")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved anomaly distribution plot to {output_path}")
