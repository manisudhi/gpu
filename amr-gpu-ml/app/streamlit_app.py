"""
app/streamlit_app.py
Interactive Exploration Dashboard for GPU-Accelerated AMR Pattern Analysis.
Allows researchers to query isolate similarity, view clusters, inspect anomalies,
and explore GPU scaling characteristics on Apple Silicon.
"""

import sys
import json
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Setup path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.amr.similarity import AMRSimilarityEngine
from src.amr.device import GPUManager

st.set_page_config(
    page_title="AMR Pattern Analysis (Apple Metal GPU)",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 GPU-Accelerated AMR Pattern Analysis & Similarity Detection")
st.markdown("""
**Target Hardware:** Apple Silicon M3 (Metal GPU) | **Organism:** *Klebsiella pneumoniae*  
*Research Prototype Only — Strictly Not for Clinical Diagnostic Decisions.*
""")

# Load artifacts
data_dir = root_dir / "data" / "processed"
reports_dir = root_dir / "reports"
figures_dir = reports_dir / "figures"
tables_dir = reports_dir / "tables"

@st.cache_resource
def load_data():
    X = np.load(data_dir / "X.npy")
    Y = np.load(data_dir / "Y.npy")
    MDR = np.load(data_dir / "mdr.npy")
    meta_df = pd.read_csv(data_dir / "isolates_metadata.csv")
    with open(data_dir / "feature_dictionary.json") as f:
        feat_dict = json.load(f)
    return X, Y, MDR, meta_df, feat_dict

try:
    X, Y, MDR, meta_df, feat_dict = load_data()
    feature_names = feat_dict["feature_names"]
    target_antibiotics = feat_dict["target_antibiotics"]
except Exception as e:
    st.error(f"Error loading pipeline artifacts: {e}. Please run `make all` first.")
    st.stop()

# Sidebar Navigation
tab = st.sidebar.radio("Navigation", [
    "1. Overview & EDA",
    "2. Supervised Resistance Models",
    "3. Parallel Similarity Search",
    "4. K-Means Clustering",
    "5. Outlier & Anomaly Detection",
    "6. GPU Pipeline Scaling"
])

if tab == "1. Overview & EDA":
    st.header("Phase 1: Dataset Overview & Exploratory Data Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Curated Isolates", f"{len(X):,}")
    col2.metric("Engineered Features", f"{X.shape[1]:,}")
    col3.metric("MDR Prevalence Rate", f"{MDR.mean()*100:.2f}%")

    st.subheader("Epidemiological & Genomic Profiles")
    c1, c2 = st.columns(2)
    if (figures_dir / "eda_isolates_per_year.png").exists():
        c1.image(str(figures_dir / "eda_isolates_per_year.png"), caption="Temporal Distribution")
    if (figures_dir / "eda_top_amr_genes.png").exists():
        c2.image(str(figures_dir / "eda_top_amr_genes.png"), caption="Top AMR Resistance Genes")

    c3, c4 = st.columns(2)
    if (figures_dir / "eda_ast_label_balance.png").exists():
        c3.image(str(figures_dir / "eda_ast_label_balance.png"), caption="AST Clinical Label Balance")
    if (figures_dir / "eda_coresistance_heatmap.png").exists():
        c4.image(str(figures_dir / "eda_coresistance_heatmap.png"), caption="Co-Resistance Heatmap")

elif tab == "2. Supervised Resistance Models":
    st.header("Phase 4: Supervised Resistance & MDR Prediction on MPS GPU")
    st.markdown("""
    Models trained on Apple Silicon GPU MPS using PyTorch. Handles extreme class imbalance
    via positive-class weighting and temporal / stratified cross-validation.
    """)
    if (tables_dir / "supervised_models_evaluation.csv").exists():
        eval_df = pd.read_csv(tables_dir / "supervised_models_evaluation.csv")
        st.dataframe(eval_df.style.format({
            "accuracy": "{:.2%}", "balanced_accuracy": "{:.2%}",
            "roc_auc": "{:.4f}", "pr_auc": "{:.4f}", "mcc": "{:.4f}"
        }))

    c1, c2 = st.columns(2)
    if (figures_dir / "confusion_matrices.png").exists():
        c1.image(str(figures_dir / "confusion_matrices.png"), caption="Confusion Matrices")
    if (figures_dir / "calibration_curves.png").exists():
        c2.image(str(figures_dir / "calibration_curves.png"), caption="Calibration Curves (Reliability Diagram)")

elif tab == "3. Parallel Similarity Search":
    st.header("Phase 5: Parallel AMR Similarity Search (Apple Metal Kernel)")
    st.markdown("Select an isolate index to retrieve nearest genomic neighbors via GPU Metal distance kernel.")

    selected_idx = st.number_input("Select Isolate Index", min_value=0, max_value=len(X)-1, value=1059)
    top_k = st.slider("Top-K Neighbors", min_value=3, max_value=20, value=5)

    if st.button("Run GPU Similarity Query"):
        engine = AMRSimilarityEngine(X, metric="euclidean")
        idxs, dists = engine.query_top_k(X[selected_idx], k=top_k)
        
        res_rows = []
        for rank, (i, d) in enumerate(zip(idxs, dists)):
            row = meta_df.iloc[i].to_dict()
            row["Rank"] = "Query (Self)" if i == selected_idx else f"Top-{rank}"
            row["Distance"] = f"{d:.4f}"
            res_rows.append(row)
        st.dataframe(pd.DataFrame(res_rows))

elif tab == "4. K-Means Clustering":
    st.header("Phase 6: GPU-Based K-Means Clustering")
    st.markdown("Clustering performed via custom Metal assignment compute shader and GPU reduction.")
    c1, c2 = st.columns(2)
    if (figures_dir / "kmeans_k_selection.png").exists():
        c1.image(str(figures_dir / "kmeans_k_selection.png"), caption="K Selection (Elbow, Silhouette, Davies-Bouldin)")
    if (figures_dir / "cluster_pca_projection.png").exists():
        c2.image(str(figures_dir / "cluster_pca_projection.png"), caption="GPU PCA Cluster Projection")

    if (tables_dir / "cluster_profiles.csv").exists():
        st.subheader("Cluster Biological Profiles")
        st.dataframe(pd.read_csv(tables_dir / "cluster_profiles.csv"))

elif tab == "5. Outlier & Anomaly Detection":
    st.header("Phase 7: Distance-Based Anomaly Detection")
    st.markdown("""
    Flags unusual genomic resistance permutations using distance to assigned cluster centroid
    evaluated against a documented 99th percentile statistical threshold.
    """)
    if (figures_dir / "anomaly_distribution.png").exists():
        st.image(str(figures_dir / "anomaly_distribution.png"), caption="Anomaly Score Distribution & Threshold")

    if (tables_dir / "top_anomalies_inspection.csv").exists():
        st.subheader("Top Flagged Anomalous Isolates")
        st.dataframe(pd.read_csv(tables_dir / "top_anomalies_inspection.csv"))

elif tab == "6. GPU Pipeline Scaling":
    st.header("Phase 8: GPU Pipeline Characterization & Scaling (Apple M3)")
    st.markdown("""
    Measures GPU-only throughput, kernel latency, and workload scaling without CPU comparisons.
    Scaling beyond 14,583 records uses controlled synthetic scaling for computational testing.
    """)
    if (figures_dir / "gpu_scaling_curves.png").exists():
        st.image(str(figures_dir / "gpu_scaling_curves.png"), caption="GPU Throughput & Latency Scaling (Log-Log)")

    if (tables_dir / "gpu_scaling_characterization.csv").exists():
        st.subheader("Empirical Hardware Benchmark Results (MacBook Air M3)")
        st.dataframe(pd.read_csv(tables_dir / "gpu_scaling_characterization.csv"))
