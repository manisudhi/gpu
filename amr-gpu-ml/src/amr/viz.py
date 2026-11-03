"""
src/amr/viz.py
Visualizations for AMR exploratory data analysis (EDA), model evaluation,
cluster projections, and GPU benchmark scaling.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger("amr.viz")

# Publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 12


def plot_eda_summary(df: pd.DataFrame, output_dir: Path) -> List[Path]:
    """
    Generate all Phase 1 EDA figures:
    1. Isolates per year
    2. Isolates per country/region
    3. Isolates per isolation source and host
    4. Most frequent resistance genes
    5. Missing-data map
    6. AST label balance per antibiotic
    7. Antibiotic co-resistance heatmap
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_figures = []

    # 1. Isolates per year
    fig, ax = plt.subplots(figsize=(9, 5))
    year_counts = df["year"].dropna().astype(int).value_counts().sort_index()
    if len(year_counts) > 0:
        year_counts.plot(kind="bar", color="#2b5c8f", ax=ax, edgecolor="black")
        ax.set_title("Temporal Distribution of Klebsiella pneumoniae Isolates (NCBI)")
        ax.set_xlabel("Collection Year")
        ax.set_ylabel("Isolate Count")
        plt.tight_layout()
        p1 = output_dir / "eda_isolates_per_year.png"
        fig.savefig(p1, dpi=300)
        plt.close(fig)
        generated_figures.append(p1)

    # 2. Isolates per country (top 15)
    fig, ax = plt.subplots(figsize=(10, 6))
    top_countries = df["country"].value_counts().head(15)
    top_countries.plot(kind="barh", color="#1f77b4", ax=ax, edgecolor="black")
    ax.invert_yaxis()
    ax.set_title("Geographic Distribution (Top 15 Countries / Regions)")
    ax.set_xlabel("Isolate Count")
    ax.set_ylabel("Country / Region")
    plt.tight_layout()
    p2 = output_dir / "eda_geographic_distribution.png"
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    generated_figures.append(p2)

    # 3. Isolation Source and Host
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    df["source"].value_counts().head(8).plot(kind="bar", color="#2ca02c", ax=ax1, edgecolor="black")
    ax1.set_title("Isolation Source Distribution")
    ax1.set_xlabel("Clinical / Environmental Source")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis="x", rotation=45)

    df["host"].value_counts().plot(kind="pie", autopct="%1.1f%%", colors=["#1f77b4", "#ff7f0e", "#7f7f7f"], ax=ax2)
    ax2.set_title("Host Reservoir Distribution")
    ax2.set_ylabel("")
    plt.tight_layout()
    p3 = output_dir / "eda_host_and_source.png"
    fig.savefig(p3, dpi=300)
    plt.close(fig)
    generated_figures.append(p3)

    # 4. Most frequent resistance genes (top 20)
    all_genes = [g for sublist in df["genes"] for g in sublist]
    gene_counts = pd.Series(all_genes).value_counts().head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    gene_counts.plot(kind="barh", color="#d62728", ax=ax, edgecolor="black")
    ax.invert_yaxis()
    ax.set_title("Top 20 Most Prevalent AMR Resistance Genes / Mutations")
    ax.set_xlabel("Frequency in Curated Dataset")
    ax.set_ylabel("Determinant (Gene / Mutation)")
    plt.tight_layout()
    p4 = output_dir / "eda_top_amr_genes.png"
    fig.savefig(p4, dpi=300)
    plt.close(fig)
    generated_figures.append(p4)

    # 5. Missing Data Map
    missing_cols = ["year", "country", "source", "host", "genes", "has_ast"]
    missing_df = df[missing_cols].isna()
    fig, ax = plt.subplots(figsize=(8, 5))
    missing_pct = missing_df.mean() * 100
    missing_pct.plot(kind="bar", color="#9467bd", ax=ax, edgecolor="black")
    ax.set_title("Missing Data Percentage by Column")
    ax.set_ylabel("Missing Percentage (%)")
    ax.set_xlabel("Metadata Field")
    ax.set_ylim(0, 100)
    plt.tight_layout()
    p5 = output_dir / "eda_missing_data_profile.png"
    fig.savefig(p5, dpi=300)
    plt.close(fig)
    generated_figures.append(p5)

    # 6. AST Label Balance per Antibiotic
    ast_df = df[df["has_ast"]].copy()
    if len(ast_df) > 0:
        # Extract records for top antibiotics
        all_drugs = {}
        for ast_dict in ast_df["ast_dict"]:
            for drug, val in ast_dict.items():
                if drug not in all_drugs:
                    all_drugs[drug] = {"R": 0, "I": 0, "S": 0}
                if val in all_drugs[drug]:
                    all_drugs[drug][val] += 1
        
        top_drugs = sorted(all_drugs.keys(), key=lambda d: sum(all_drugs[d].values()), reverse=True)[:10]
        if top_drugs:
            chart_data = pd.DataFrame([all_drugs[d] for d in top_drugs], index=top_drugs)
            fig, ax = plt.subplots(figsize=(11, 6))
            chart_data.plot(kind="bar", stacked=True, color=["#d62728", "#ff7f0e", "#2ca02c"], ax=ax, edgecolor="black")
            ax.set_title("AST Clinical Label Balance Across Top 10 Antibiotics")
            ax.set_ylabel("Number of Tested Isolates")
            ax.set_xlabel("Antibiotic")
            ax.legend(["Resistant (R)", "Intermediate (I)", "Susceptible (S)"])
            ax.tick_params(axis="x", rotation=45)
            plt.tight_layout()
            p6 = output_dir / "eda_ast_label_balance.png"
            fig.savefig(p6, dpi=300)
            plt.close(fig)
            generated_figures.append(p6)

            # 7. Antibiotic co-resistance heatmap
            co_res = []
            for ast_dict in ast_df["ast_dict"]:
                row = {}
                for d in top_drugs:
                    # 1 if R or I, 0 if S, NaN if not tested
                    val = ast_dict.get(d)
                    if val in ("R", "I"):
                        row[d] = 1.0
                    elif val == "S":
                        row[d] = 0.0
                    else:
                        row[d] = np.nan
                co_res.append(row)
            co_res_df = pd.DataFrame(co_res)
            corr = co_res_df.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, vmin=-0.2, vmax=1.0)
            ax.set_title("Antibiotic Co-Resistance Correlation Heatmap (Pearson r)")
            plt.tight_layout()
            p7 = output_dir / "eda_coresistance_heatmap.png"
            fig.savefig(p7, dpi=300)
            plt.close(fig)
            generated_figures.append(p7)

    logger.info(f"Generated {len(generated_figures)} EDA figures in {output_dir}")
    return generated_figures
