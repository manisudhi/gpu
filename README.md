# GPU-Accelerated Antimicrobial Resistance Pattern Analysis and Similarity Detection

[![Platform](https://img.shields.io/badge/Platform-Apple%20Silicon%20(M3)-black.svg?style=flat&logo=apple)](https://developer.apple.com/metal/)
[![GPU Stack](https://img.shields.io/badge/GPU%20Stack-Apple%20Metal%20%7C%20PyTorch%20MPS-orange.svg)](https://pytorch.org/docs/stable/notes/mps.html)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/pytest-10%20passed-brightgreen.svg)](tests/)
[![Deployed Site](https://img.shields.io/badge/Live%20Site-GitHub%20Pages-success.svg)](https://manisudhi.github.io/gpu/)
[![Repository](https://img.shields.io/badge/GitHub-manisudhi%2Fgpu-blue.svg?logo=github)](https://github.com/manisudhi/gpu)

A reproducible, high-performance GPU-accelerated machine learning and pattern analysis system for **Antimicrobial Resistance (AMR)** surveillance and genomic phenotype prediction. Engineered natively for **Apple Silicon (MacBook Air M3)** using **Apple Metal Compute Shaders (MSL)** and **PyTorch Metal Performance Shaders (MPS)** in unified memory.

> [!IMPORTANT]
> **Safety & Research Prototype Disclaimer**: This project is an analytical research prototype designed for computational genomics, algorithm optimization, and hardware benchmarking. It is **strictly NOT a clinical diagnostic or treatment decision tool**.

---

## 🌐 Live Interactive Deployment & Code Reference

- **Interactive Web Report & Dashboard:** [https://manisudhi.github.io/gpu/](https://manisudhi.github.io/gpu/)
- **GitHub Code Repository:** [https://github.com/manisudhi/gpu](https://github.com/manisudhi/gpu)
- **Academic Project Report (Markdown):** [amr-gpu-ml/reports/final_report.md](amr-gpu-ml/reports/final_report.md)
- **Formatted Document:** [GPU_AMR_Metal_Project_Report.docx](GPU_AMR_Metal_Project_Report.docx)

---

## 🚀 Key Achievements & Measured Empirical Results

All benchmarks were evaluated on an **Apple MacBook Air M3 (8-core CPU, 10-core GPU, Unified Memory)**:

- **Empirical Genomic Dataset:** Curated **14,583 *Klebsiella pneumoniae*** whole-genome sequencing (WGS) isolates from NCBI Pathogen Detection snapshot `PDG000000012.2550` across **263 multi-hot numerical resistance determinants**.
- **Numerical Correctness:** Apple Metal compute shaders matched 64-bit CPU reference calculations with a maximum absolute discrepancy of \(4.77 \times 10^{-7}\) (\(\epsilon < 10^{-4}\)), with 100.0% exact cluster assignment agreement.
- **Supervised ML Performance:** Deep neural networks trained on MPS GPU achieved a **macro-average ROC-AUC of 0.8945** across 10 critical antibiotics (Ceftazidime: 0.9852, Ciprofloxacin: 0.9847, Cefepime: 0.9590, Gentamicin: 0.9534, Trimethoprim-sulfamethoxazole: 0.9420).
- **Biological Validation:** Gradient attribution confirmed that models independently identify established resistance mechanisms: `blaKPC-2/3` for carbapenems, `aac(6')-Ib` for aminoglycosides, `sul1` for sulfonamides, and `gyrA_S83I`/`parC_S80I` for fluoroquinolones.
- **Unsupervised K-Means Clustering:** GPU K-Means converged in 21 iterations (\(K=5\)), isolating discrete clinical clones (e.g., 93.3% `blaCTX-M-15` ESBL clone in Cluster 3; 100% `gyrA/parC` fluoroquinolone-resistant clone in Cluster 1).
- **Statistical Anomaly Detection:** Detected 146 outlier strains (1.00%) exceeding the 99th-percentile statistical threshold (\(s > 4.608\)), characterized by extreme hyper-determinant plasmid carriage (\(> 40\) resistance genes).
- **Peak Hardware Throughput:** Achieved **5,446,186 records/second** peak GPU kernel throughput with latencies under 10 ms.

---

## 🏗️ Architecture & Branching Data Flow

The system implements a four-way branching GPU data flow where preprocessed contiguous float32 feature matrices feed into parallel GPU streams:

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

## 📂 Repository Directory Layout

```
.
├── README.md                      # Primary repository overview
├── Makefile                       # Top-level build and test automation
├── .gitignore                     # Git ignore rules
├── .github/
│   └── workflows/
│       ├── ci.yml                 # Automated testing CI workflow
│       └── deploy-pages.yml       # Automated GitHub Pages deployment
├── docs/                          # GitHub Pages interactive web report & assets
│   ├── index.html                 # Interactive web dashboard & report
│   ├── figures/                   # 17 Publication figures and diagrams
│   ├── data/                      # Evaluation tables & benchmark CSVs
│   └── reports/                   # Markdown project report
├── data/
│   ├── raw/                       # Cached raw NCBI metadata TSV
│   ├── processed/                 # Contiguous float32 feature matrices (X, Y, MDR)
│   └── DATA_CARD.md               # Standardized dataset provenance & metadata card
├── amr-gpu-ml/                    # Complete Python source codebase
│   ├── README.md                  # Detailed Python module documentation
│   ├── Makefile                   # Execution targets
│   ├── requirements.txt           # Pinned dependencies
│   ├── configs/config.yaml        # Central system and hyperparameter config
│   ├── src/amr/
│   │   ├── device.py              # Apple Metal / PyTorch MPS device layer
│   │   ├── data.py                # Dataset ingestion and parsing
│   │   ├── features.py            # Feature extraction and multi-hot encoding
│   │   ├── models.py              # PyTorch MPS neural networks
│   │   ├── similarity.py          # Parallel similarity engine & GPU Top-K
│   │   ├── clustering.py          # GPU K-Means clustering
│   │   ├── anomaly.py             # Distance anomaly detection
│   │   ├── metrics.py             # Clinical ML evaluation
│   │   ├── viz.py                 # EDA plots and publication figures
│   │   ├── pipeline.py            # End-to-end pipeline orchestrator
│   │   └── kernels/               # Apple Metal MSL Compute Shaders
│   │       ├── similarity.metal   # Parallel query distance MSL kernel
│   │       ├── kmeans_assign.metal# Nearest centroid assign MSL kernel
│   │       └── anomaly_score.metal# Centroid distance MSL kernel
│   ├── scripts/                   # CLI runners & diagram generators
│   ├── tests/                     # 10 automated PyTest unit & integration tests
│   ├── reports/                   # Figures, tables, and final report
│   └── app/
│       └── streamlit_app.py       # Local interactive Streamlit exploration app
```

---

## ⚡ Quickstart & Reproduction

### 1. Installation
Clone the repository and set up a virtual environment on macOS:

```bash
git clone https://github.com/manisudhi/gpu.git
cd gpu/amr-gpu-ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Automated PyTest Suite
Verify GPU device initialization, numerical tolerances (\(\epsilon < 10^{-4}\)), feature dimensions, and model convergence:

```bash
make test
# or: pytest -v tests/
```

### 3. Run Full GPU Analytical Pipeline (Phases 1-8)
```bash
make run
# or: python3 scripts/run_pipeline.py --config configs/config.yaml
```

### 4. Launch Local Interactive Web App
```bash
make app
# or: streamlit run app/streamlit_app.py
```

---

## 📊 Hardware Scaling Benchmarks (MacBook Air M3)

| Workload Size | GPU Kernel Time | Pipeline Throughput | Memory Footprint | Workload Type |
| :---: | :---: | :---: | :---: | :--- |
| **20,000** | 4.02 ± 0.06 ms | 4,977,481 rec/s | 20.1 MB | Controlled Synthetic Scaling |
| **50,000** | 9.00 ± 0.19 ms | **5,558,540 rec/s** | 50.2 MB | Peak GPU Saturation |
| **100,000** | 18.12 ± 0.31 ms | 5,519,758 rec/s | 100.3 MB | Controlled Synthetic Scaling |
| **500,000** | 94.46 ± 2.39 ms | 5,293,384 rec/s | 501.6 MB | Controlled Synthetic Scaling |
| **1,000,000** | 227.96 ± 33.18 ms | 4,386,736 rec/s | 1,003.3 MB | High-Density Scale |

---

## 📄 Citation & Attribution

- **Target Hardware:** Apple MacBook Air M3 (8-core CPU, 10-core GPU, Unified Memory)
- **Primary Frameworks:** Apple Metal Shading Language, Apple Metal Performance Shaders (MPS), PyTorch 2.5+, MLX
- **Data Source:** NCBI Pathogen Detection Project (*Klebsiella pneumoniae*, snapshot `PDG000000012.2550`)
- **Author:** Sudharshini Manikandan ([@manisudhi](https://github.com/manisudhi))
