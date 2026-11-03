# GPU-Accelerated Antimicrobial Resistance Pattern Analysis and Similarity Detection

[![Platform](https://img.shields.io/badge/Platform-Apple%20Silicon%20(M3)-black.svg?style=flat&logo=apple)](https://developer.apple.com/metal/)
[![GPU Stack](https://img.shields.io/badge/GPU%20Stack-Apple%20Metal%20%7C%20PyTorch%20MPS-orange.svg)](https://pytorch.org/docs/stable/notes/mps.html)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/pytest-10%20passed-brightgreen.svg)](tests/)

A complete, reproducible, GPU-accelerated machine learning and pattern analysis system for antimicrobial resistance (AMR) pattern discovery, similarity detection, unsupervised clustering, and phenotype prediction using Apple Metal Compute Shaders on Apple Silicon (MacBook Air M3).

> [!IMPORTANT]
> **Safety & Research Prototype Disclaimer**: This project is an analytical research prototype designed for computational pattern discovery, clustering, and hardware performance characterization. It is **strictly NOT a clinical diagnostic or treatment decision tool**.

---

## Key Achievements & Measured Empirical Results

All measurements were recorded on an **Apple MacBook Air M3 (8-core CPU, 10-core GPU, Unified Memory)**:

- **Empirical Dataset:** Curated **14,583 *Klebsiella pneumoniae*** isolates from NCBI Pathogen Detection across **263 numerical determinants**.
- **Numerical Validation:** Metal GPU compute shaders matched 64-bit CPU references with a maximum absolute discrepancy of \(4.77 \times 10^{-7}\) (\(\epsilon < 10^{-4}\)), with 100.0% exact cluster assignment agreement.
- **Supervised ML Performance:** Deep neural networks trained on MPS GPU achieved a **macro-average ROC-AUC of 0.8945** (Ceftazidime: 0.9852, Ciprofloxacin: 0.9847, Cefepime: 0.9590, Gentamicin: 0.9534, Trimethoprim-sulfamethoxazole: 0.9420).
- **Biological Interpretability:** Model gradient attribution identified `blaKPC-2/3` as primary determinants for carbapenems, `aac(6')-Ib` for amikacin, and `sul1` for sulfonamides, aligning with established molecular microbiology.
- **K-Means Clustering:** GPU K-Means converged in 21 iterations, isolating distinct biological lineages (e.g., 93.3% `blaCTX-M-15` ESBL clone in Cluster 3; 100% `gyrA/parC` double-mutation clone in Cluster 1).
- **Statistical Anomaly Detection:** Identified 146 outlier isolates (1.00%) exceeding the 99th-percentile statistical threshold (\(s > 4.608\)), characterized by hyper-determinant carriage (\(> 40\) resistance genes).
- **Peak Hardware Throughput:** Achieved **5,446,186 records/second** peak GPU throughput with kernel latencies under 10 ms.

---

## Architecture Overview

The system implements a branching data-flow architecture where preprocessed feature matrices split into four dedicated GPU analytical branches:

```
                      +------------------------------------------+
                      |       NCBI Pathogen Detection Data       |
                      +------------------------------------------+
                                           |
                                           v
                      +------------------------------------------+
                      |    Preprocessing & Feature Extraction    |
                      |    [14,583 x 263] Contiguous float32     |
                      +------------------------------------------+
                               /           |          \          \
                              /            |           \          \
                             v             v            v          v
                      +-------------+ +----------+ +----------+ +----------+
                      | Branch 1:   | | Branch 2:| | Branch 3:| | Branch 4:|
                      | Supervised  | | Parallel | | K-Means  | | Distance |
                      | PyTorch MPS | | Similarity| | Clust.  | | Anomaly  |
                      | Deep MLP    | | Metal MSL| | Metal MSL| | Metal MSL|
                      +-------------+ +----------+ +----------+ +----------+
                             \             |            /          /
                              \            |           /          /
                               v           v          v          v
                      +------------------------------------------+
                      |        Converged Analytical Output       |
                      +------------------------------------------+
                                           |
                                           v
                      +------------------------------------------+
                      | Visualizations, Tables & Final Report    |
                      +------------------------------------------+
```

---

## Directory Structure

```
amr-gpu-ml/
  ├── README.md                      # Complete system documentation
  ├── Makefile                       # Single-command build and test automation
  ├── requirements.txt               # Pinned Python package dependencies
  ├── configs/
  │   └── config.yaml                # Central system and model hyperparameter config
  ├── data/
  │   ├── raw/                       # Cached raw NCBI metadata
  │   ├── processed/                 # Contiguous float32 feature matrices (X, Y, MDR)
  │   └── DATA_CARD.md               # Standardized dataset provenance & metadata card
  ├── src/
  │   └── amr/
  │       ├── device.py              # Apple Metal / PyTorch MPS device layer
  │       ├── data.py                # NCBI parsing, standardization, and splits
  │       ├── features.py            # Multi-hot determinant matrices & label encoding
  │       ├── models.py              # PyTorch MPS neural networks (5 architectures)
  │       ├── similarity.py          # Parallel similarity engine & GPU Top-K
  │       ├── clustering.py          # GPU K-Means clustering & biological profiling
  │       ├── anomaly.py             # Distance anomaly detection & thresholding
  │       ├── metrics.py             # Clinical ML evaluation & calibration curves
  │       ├── viz.py                 # EDA plots and publication visual figures
  │       ├── pipeline.py            # End-to-end pipeline orchestrator (Phases 1-8)
  │       └── kernels/
  │           ├── similarity.metal   # MSL compute shader: parallel query distance
  │           ├── kmeans_assign.metal# MSL compute shader: nearest centroid assign
  │           ├── anomaly_score.metal# MSL compute shader: distance-to-centroid
  │           └── wrappers.py        # MLX JIT dispatchers & PyTorch MPS fallbacks
  ├── scripts/
  │   ├── download_data.py           # Idempotent dataset fetcher
  │   ├── run_pipeline.py            # CLI entrypoint for all phases
  │   └── generate_diagrams.py       # Renders publication PNG architecture diagrams
  ├── tests/
  │   ├── conftest.py                # Test environment path configuration
  │   ├── test_kernels.py            # Subsample GPU numerical tolerance validation
  │   ├── test_features.py           # Feature dimensions, NaNs, and leakage checks
  │   ├── test_models.py             # PyTorch MPS forward passes and convergence
  │   └── test_clustering.py         # K-Means convergence and determinism
  ├── docs/
  │   ├── architecture/              # System architecture diagrams (Mermaid & PNG)
  │   └── dfd/                       # Data Flow Diagrams Levels 0 and 1 (Mermaid & PNG)
  ├── reports/
  │   ├── final_report.md            # Comprehensive academic & engineering report
  │   ├── figures/                   # 17 publication-quality visualization figures
  │   └── tables/                    # CSV evaluation and benchmark result tables
  └── app/
      └── streamlit_app.py           # Interactive web exploration dashboard
```

---

## Quickstart & Execution

### 1. Installation
Ensure you are running on an Apple Silicon Mac (M1/M2/M3/M4) with macOS 12.3+ and Python 3.10+:

```bash
git clone <repo_url>
cd amr-gpu-ml
pip install -r requirements.txt
```

### 2. One-Command Complete Execution
Run all 9 phases, tests, and diagram generators:

```bash
make all
```

Or run via the CLI runner:

```bash
python3 scripts/run_pipeline.py --config configs/config.yaml
```

### 3. Run Automated PyTest Suite
Execute the full automated test suite verifying GPU device initialization, numerical tolerances, leakage absence, and model convergence:

```bash
make test
# or: python3 -m pytest -v tests/
```

### 4. Launch Interactive Web Dashboard
Explore isolate similarity, cluster projections, and anomaly distributions interactively:

```bash
make app
# or: streamlit run app/streamlit_app.py
```

---

## Hardware Characterization (MacBook Air M3)

Workload scaling evaluated using controlled synthetic expansion per Project Report Section 10 & 24:

| Workload Size | GPU Kernel Time | Pipeline Throughput | Memory Footprint | Workload Type |
| :---: | :---: | :---: | :---: | :--- |
| **20,000** | 4.86 ± 0.33 ms | 4,113,534 records/s | 20.1 MB | Controlled Synthetic Scaling |
| **50,000** | 9.18 ± 0.30 ms | **5,446,186 records/s** | 50.2 MB | Controlled Synthetic Scaling |
| **100,000** | 20.02 ± 0.73 ms | 4,994,513 records/s | 100.3 MB | Controlled Synthetic Scaling |
| **500,000** | 100.75 ± 0.75 ms | 4,962,850 records/s | 501.6 MB | Controlled Synthetic Scaling |
| **1,000,000** | 678.99 ± 44.24 ms | 1,472,774 records/s | 1,003.3 MB | Controlled Synthetic Scaling |

---

## Authors & Citation
- **Project Title:** GPU-Accelerated Antimicrobial Resistance Pattern Analysis and Similarity Detection
- **Target Architecture:** Apple Metal Shading Language & Metal Performance Shaders on Apple Silicon M3
- **Primary Reference Data:** NCBI Pathogen Detection Project (*Klebsiella pneumoniae*)
