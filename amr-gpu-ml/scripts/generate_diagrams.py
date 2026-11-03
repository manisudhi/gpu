#!/usr/bin/env python3
"""
scripts/generate_diagrams.py
Generates publication-quality architecture and Data Flow Diagrams (DFD) as PNG images.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
DOC_ARCH = Path(__file__).resolve().parent.parent / "docs" / "architecture"
DOC_DFD = Path(__file__).resolve().parent.parent / "docs" / "dfd"

for d in [FIG_DIR, DOC_ARCH, DOC_DFD]:
    d.mkdir(parents=True, exist_ok=True)


def draw_box(ax, x, y, w, h, text, color="#e6f7ff", edge="#1890ff", fontsize=9, bold=False):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        facecolor=color, edgecolor=edge, linewidth=1.5
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight, multialignment="center")


def draw_arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5, mutation_scale=12)
    )
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.02, label, ha="center", va="bottom", fontsize=8, color="#555555")


def generate_branching_architecture():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, 0.35, 0.88, 0.30, 0.08, "NCBI Pathogen Detection Data\n(Klebsiella pneumoniae)", color="#fff7e6", edge="#d46b08", bold=True)
    draw_box(ax, 0.35, 0.75, 0.30, 0.07, "Data Ingestion & Validation\n(scripts/download_data.py)")
    draw_box(ax, 0.35, 0.62, 0.30, 0.07, "Preprocessing & Encoding\n(Multi-Hot Genotype Matrix [N x D])")

    draw_arrow(ax, 0.50, 0.88, 0.50, 0.82)
    draw_arrow(ax, 0.50, 0.75, 0.50, 0.69)

    # Subsample validation box
    draw_box(ax, 0.04, 0.62, 0.22, 0.07, "Correctness Validation\n(NumPy Ref, Tol < 1e-4)", color="#f6ffed", edge="#52c41a")
    draw_arrow(ax, 0.35, 0.655, 0.26, 0.655, label="Subsample")

    # Branches
    draw_box(ax, 0.02, 0.44, 0.22, 0.08, "Branch 1: Supervised ML\n(Deep MLP & Multi-Task MPS)\nROC-AUC: 0.8945", color="#e6f7ff", edge="#1890ff", bold=True)
    draw_box(ax, 0.26, 0.44, 0.22, 0.08, "Branch 2: Parallel Similarity\n(Metal Query-vs-All Kernel)\nGPU Top-K Retrieval", color="#e6f7ff", edge="#1890ff", bold=True)
    draw_box(ax, 0.50, 0.44, 0.22, 0.08, "Branch 3: K-Means Clustering\n(Metal Assignment Kernel)\nK=5 Clusters, GPU Reduction", color="#e6f7ff", edge="#1890ff", bold=True)
    draw_box(ax, 0.74, 0.44, 0.24, 0.08, "Branch 4: Anomaly Detection\n(Metal Distance Kernel)\n99th Percentile Thresholding", color="#e6f7ff", edge="#1890ff", bold=True)

    draw_arrow(ax, 0.50, 0.62, 0.13, 0.52)
    draw_arrow(ax, 0.50, 0.62, 0.37, 0.52)
    draw_arrow(ax, 0.50, 0.62, 0.61, 0.52)
    draw_arrow(ax, 0.50, 0.62, 0.86, 0.52)

    # Converged Results
    draw_box(ax, 0.30, 0.26, 0.40, 0.08, "Converged Analytical Results\n(Predictions, Neighbors, Clusters, Outliers)", color="#f9f0ff", edge="#722ed1", bold=True)
    draw_arrow(ax, 0.13, 0.44, 0.42, 0.34)
    draw_arrow(ax, 0.37, 0.44, 0.46, 0.34)
    draw_arrow(ax, 0.61, 0.44, 0.54, 0.34)
    draw_arrow(ax, 0.86, 0.44, 0.58, 0.34)

    # Characterization & Reports
    draw_box(ax, 0.10, 0.12, 0.38, 0.07, "GPU Pipeline Characterization\nThroughput: 5.4M records/s (Apple M3)")
    draw_box(ax, 0.52, 0.12, 0.38, 0.07, "Reporting, Tables & Visualizations\n(reports/figures & final_report.md)")

    draw_arrow(ax, 0.45, 0.26, 0.29, 0.19)
    draw_arrow(ax, 0.55, 0.26, 0.71, 0.19)

    plt.title("Branching System Architecture for GPU AMR Machine Learning", fontsize=14, weight="bold", pad=20)
    plt.tight_layout()
    p1 = FIG_DIR / "branching_architecture.png"
    p2 = DOC_ARCH / "branching_architecture.png"
    fig.savefig(p1, dpi=300)
    fig.savefig(p2, dpi=300)
    plt.close(fig)


def generate_host_device_model():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Host Box
    rect_host = patches.FancyBboxPatch((0.03, 0.1), 0.42, 0.78, boxstyle="round,pad=0.03", facecolor="#f0f7ff", edgecolor="#0366d6", lw=2)
    ax.add_patch(rect_host)
    ax.text(0.24, 0.84, "Host CPU (Python / PyTorch / MLX)", ha="center", va="center", weight="bold", fontsize=11, color="#0366d6")

    draw_box(ax, 0.06, 0.68, 0.36, 0.08, "1. Ingest & Prepare Contiguous Matrix\n[N x D] float32 arrays in RAM")
    draw_box(ax, 0.06, 0.54, 0.36, 0.08, "2. Compile Metal Compute Shaders\n(mx.fast.metal_kernel JIT / PyTorch MPS)")
    draw_box(ax, 0.06, 0.40, 0.36, 0.08, "3. Command Encoder & Buffer Binding\n(Zero-copy shared buffers)")
    draw_box(ax, 0.06, 0.26, 0.36, 0.08, "4. Dispatch 1D Grid & Threadgroups\n(256 threads/group, 1 thread/record)")
    draw_box(ax, 0.06, 0.14, 0.36, 0.08, "7. Synchronize & Read Results to Host\n(Cluster labels, Top-K, Anomaly flags)")

    # GPU Box
    rect_gpu = patches.FancyBboxPatch((0.55, 0.1), 0.42, 0.78, boxstyle="round,pad=0.03", facecolor="#f6ffed", edgecolor="#52c41a", lw=2)
    ax.add_patch(rect_gpu)
    ax.text(0.76, 0.84, "Apple Metal GPU (M3 Unified Memory)", ha="center", va="center", weight="bold", fontsize=11, color="#389e0d")

    draw_box(ax, 0.58, 0.68, 0.36, 0.08, "Unified Memory Device Buffers\n- Feature Matrix [N x D]\n- Query / Centroids [K x D]\n- Result Arrays [N]", color="#ffffff", edge="#52c41a")
    draw_box(ax, 0.58, 0.47, 0.36, 0.14, "Compute Shader Execution\nThread Index: thread_position_in_grid.x\n- SIMD-group Width: 32 threads\n- amr_similarity_euclidean\n- amr_kmeans_assign\n- amr_anomaly_score", color="#ffffff", edge="#52c41a")
    draw_box(ax, 0.58, 0.20, 0.36, 0.18, "GPU Reduction & Optimization\n- Centroid updates via scatter-add\n- GPU Top-K selection on scores\n- Throughput: 5.4M records/s", color="#ffffff", edge="#52c41a")

    # Connectors
    draw_arrow(ax, 0.42, 0.44, 0.58, 0.72, label="Bind Buffers")
    draw_arrow(ax, 0.42, 0.30, 0.58, 0.54, label="Dispatch Kernel")
    draw_arrow(ax, 0.58, 0.29, 0.42, 0.18, label="Readback")

    plt.title("Host-Side vs Apple Metal GPU Execution Architecture", fontsize=14, weight="bold", pad=20)
    plt.tight_layout()
    p1 = FIG_DIR / "host_device_model.png"
    p2 = DOC_ARCH / "host_device_model.png"
    fig.savefig(p1, dpi=300)
    fig.savefig(p2, dpi=300)
    plt.close(fig)


def generate_dfds():
    # DFD Level 0
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, 0.05, 0.65, 0.26, 0.16, "External Entity:\nNCBI Pathogen Detection\n(ftp.ncbi.nlm.nih.gov)", color="#fff1f0", edge="#f5222d", bold=True)
    draw_box(ax, 0.05, 0.20, 0.26, 0.16, "External Entity:\nAMR Researcher /\nMicrobiologist", color="#fff1f0", edge="#f5222d", bold=True)

    draw_box(ax, 0.45, 0.35, 0.48, 0.32, "0.0 GPU-Accelerated AMR Pattern Analysis System\n(Apple Silicon Metal / PyTorch MPS)\n\n- Genotype-Phenotype Supervised Learning\n- Parallel Similarity Matching & GPU Top-K\n- Fast K-Means Clustering & Profiling\n- Distance Anomaly Detection & Flagging", color="#e6f7ff", edge="#1890ff", bold=True)

    draw_arrow(ax, 0.31, 0.73, 0.45, 0.58, label="Raw Genotypes & AST Data")
    draw_arrow(ax, 0.31, 0.28, 0.45, 0.42, label="Query Profile / Target Drugs")
    draw_arrow(ax, 0.45, 0.36, 0.31, 0.22, label="Clusters / Attributions / Anomalies")

    plt.title("Data Flow Diagram (DFD) — Level 0 Context Diagram", fontsize=14, weight="bold", pad=15)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "dfd_level_0.png", dpi=300)
    fig.savefig(DOC_DFD / "dfd_level_0.png", dpi=300)
    plt.close(fig)

    # DFD Level 1
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, 0.02, 0.85, 0.22, 0.08, "1.0 Data Ingestion\n(Download & Verify)", color="#e6f7ff", edge="#1890ff")
    draw_box(ax, 0.28, 0.85, 0.22, 0.08, "2.0 Preprocessing\n(Multi-Hot Matrices)", color="#e6f7ff", edge="#1890ff")
    draw_box(ax, 0.54, 0.85, 0.20, 0.08, "3.0 Subsample Check\n(NumPy Validation)", color="#f6ffed", edge="#52c41a")
    draw_box(ax, 0.78, 0.85, 0.20, 0.08, "4.0 GPU Supervised ML\n(Deep MLP / Multi-Task)", color="#e6f7ff", edge="#1890ff")

    draw_box(ax, 0.02, 0.52, 0.22, 0.08, "5.0 GPU Similarity\n(Metal Query Distance)", color="#e6f7ff", edge="#1890ff")
    draw_box(ax, 0.28, 0.52, 0.22, 0.08, "6.0 GPU Clustering\n(Metal K-Means)", color="#e6f7ff", edge="#1890ff")
    draw_box(ax, 0.54, 0.52, 0.20, 0.08, "7.0 Anomaly Detection\n(Metal Centroid Dist)", color="#e6f7ff", edge="#1890ff")
    draw_box(ax, 0.78, 0.52, 0.20, 0.08, "8.0 Characterization\n(Scaling & Throughput)", color="#e6f7ff", edge="#1890ff")

    # Data Stores
    draw_box(ax, 0.05, 0.20, 0.25, 0.10, "Data Store D1:\nRaw NCBI Metadata\n(14.5k isolates)", color="#f9f0ff", edge="#722ed1")
    draw_box(ax, 0.37, 0.20, 0.26, 0.10, "Data Store D2:\nProcessed Feature Tensors\n[14583 x 263] float32", color="#f9f0ff", edge="#722ed1")
    draw_box(ax, 0.70, 0.20, 0.25, 0.10, "Data Store D3:\nCheckpoints & Metrics\n(Models, Tables, Figures)", color="#f9f0ff", edge="#722ed1")

    # Connectors
    draw_arrow(ax, 0.24, 0.89, 0.28, 0.89)
    draw_arrow(ax, 0.50, 0.89, 0.54, 0.89)
    draw_arrow(ax, 0.74, 0.89, 0.78, 0.89)

    draw_arrow(ax, 0.13, 0.85, 0.17, 0.30)
    draw_arrow(ax, 0.39, 0.85, 0.50, 0.30)
    draw_arrow(ax, 0.50, 0.30, 0.13, 0.52)
    draw_arrow(ax, 0.50, 0.30, 0.39, 0.52)
    draw_arrow(ax, 0.50, 0.30, 0.64, 0.52)
    draw_arrow(ax, 0.50, 0.30, 0.88, 0.52)

    draw_arrow(ax, 0.88, 0.85, 0.82, 0.30)
    draw_arrow(ax, 0.88, 0.52, 0.82, 0.30)

    plt.title("Data Flow Diagram (DFD) — Level 1 Detailed Architecture", fontsize=14, weight="bold", pad=15)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "dfd_level_1.png", dpi=300)
    fig.savefig(DOC_DFD / "dfd_level_1.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    generate_branching_architecture()
    generate_host_device_model()
    generate_dfds()
    print("All architecture and DFD diagrams generated successfully.")
