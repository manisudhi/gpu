"""
src/amr/features.py
Feature engineering, multi-hot gene matrix encoding, rare feature pruning,
and leakage-free label construction for AMR GPU ML.
"""

import os
import json
import logging
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

from src.amr.data import ANTIBIOTIC_CLASS_MAP

logger = logging.getLogger("amr.features")


class FeaturePipeline:
    """
    Transforms parsed AMR isolate records into contiguous float32 feature matrices
    and multi-target resistance labels without label leakage.
    """

    def __init__(
        self,
        rare_gene_threshold_pct: float = 0.002,
        min_gene_freq: int = 10,
        target_antibiotics: Optional[List[str]] = None,
        intermediate_as_resistant: bool = True,
        mdr_min_classes: int = 3,
    ):
        self.rare_gene_threshold_pct = rare_gene_threshold_pct
        self.min_gene_freq = min_gene_freq
        self.target_antibiotics = target_antibiotics or [
            "meropenem", "imipenem", "ertapenem", "ceftazidime", "cefepime",
            "ciprofloxacin", "amikacin", "gentamicin", "colistin",
            "trimethoprim-sulfamethoxazole"
        ]
        self.intermediate_as_resistant = intermediate_as_resistant
        self.mdr_min_classes = mdr_min_classes

        self.gene_vocab: List[str] = []
        self.gene_to_idx: Dict[str, int] = {}
        self.feature_names: List[str] = []
        self.country_vocab: List[str] = []
        self.source_vocab: List[str] = []
        self.year_min: float = 2010.0
        self.year_max: float = 2026.0

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        """Learn vocabularies and scaling parameters from training isolates."""
        logger.info(f"Fitting FeaturePipeline on {len(df):,} records...")

        # 1. Learn Gene Determinants Vocabulary
        gene_counter = Counter()
        for gene_list in df["genes"]:
            gene_counter.update(set(gene_list))

        min_count = max(self.min_gene_freq, int(len(df) * self.rare_gene_threshold_pct))
        self.gene_vocab = sorted([gene for gene, count in gene_counter.items() if count >= min_count])
        self.gene_to_idx = {g: i for i, g in enumerate(self.gene_vocab)}
        logger.info(
            f"Extracted {len(gene_counter):,} distinct genes/mutations; retained {len(self.gene_vocab):,} "
            f"determinants after pruning rare features (threshold={min_count} isolates)."
        )

        # 2. Learn Top Categorical Vocabularies
        self.country_vocab = list(df["country"].value_counts().head(10).index)
        self.source_vocab = list(df["source"].value_counts().head(7).index)

        # 3. Year Normalization Bounds
        valid_years = df["year"].dropna()
        if len(valid_years) > 0:
            self.year_min = float(valid_years.min())
            self.year_max = float(valid_years.max())
            if self.year_max == self.year_min:
                self.year_max += 1.0

        # Build feature names
        self.feature_names = [f"gene_{g}" for g in self.gene_vocab]
        self.feature_names.append("norm_year")
        self.feature_names.append("gene_burden")
        for c in self.country_vocab:
            self.feature_names.append(f"country_{c}")
        self.feature_names.append("country_Other")
        for s in self.source_vocab:
            self.feature_names.append(f"source_{s}")
        self.feature_names.append("source_Other")
        self.feature_names.extend(["host_Human", "host_Animal", "host_Unknown"])

        logger.info(f"Total engineered features per record: {len(self.feature_names):,}")
        return self

    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Build contiguous float32 numerical feature matrix X.
        Guarantees: No NaNs, zero leakage from AST results, contiguous layout.
        """
        num_records = len(df)
        num_features = len(self.feature_names)
        X = np.zeros((num_records, num_features), dtype=np.float32)

        num_genes = len(self.gene_vocab)
        year_idx = num_genes
        burden_idx = num_genes + 1
        country_start = num_genes + 2
        source_start = country_start + len(self.country_vocab) + 1
        host_start = source_start + len(self.source_vocab) + 1

        for i, row in enumerate(df.itertuples(index=False)):
            # 1. Binary multi-hot gene encoding
            for gene in row.genes:
                if gene in self.gene_to_idx:
                    X[i, self.gene_to_idx[gene]] = 1.0

            # 2. Gene burden (total AMR determinant count)
            X[i, burden_idx] = float(row.gene_count) / 20.0

            # 3. Normalized year
            y = row.year if pd.notna(row.year) else (self.year_min + self.year_max) / 2.0
            norm_y = (float(y) - self.year_min) / (self.year_max - self.year_min)
            X[i, year_idx] = float(np.clip(norm_y, 0.0, 1.0))

            # 4. Country one-hot
            c = row.country
            if c in self.country_vocab:
                X[i, country_start + self.country_vocab.index(c)] = 1.0
            else:
                X[i, country_start + len(self.country_vocab)] = 1.0

            # 5. Source one-hot
            s = row.source
            if s in self.source_vocab:
                X[i, source_start + self.source_vocab.index(s)] = 1.0
            else:
                X[i, source_start + len(self.source_vocab)] = 1.0

            # 6. Host one-hot
            h = row.host
            if h == "Human":
                X[i, host_start] = 1.0
            elif h == "Animal":
                X[i, host_start + 1] = 1.0
            else:
                X[i, host_start + 2] = 1.0

        # Enforce C-contiguous layout for direct Metal buffer binding
        X = np.ascontiguousarray(X, dtype=np.float32)
        assert not np.isnan(X).any(), "FATAL: Feature matrix contains NaN values!"
        return X

    def transform_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build resistance ground-truth labels and masks.
        Returns:
            Y: float32 array [N, num_target_antibiotics], 1=Resistant, 0=Susceptible
            Y_mask: bool array [N, num_target_antibiotics], True if antibiotic was tested
            MDR: float32 array [N], 1=Multidrug resistant (>= 3 classes), 0=Non-MDR
        """
        num_records = len(df)
        num_targets = len(self.target_antibiotics)

        Y = np.zeros((num_records, num_targets), dtype=np.float32)
        Y_mask = np.zeros((num_records, num_targets), dtype=bool)
        MDR = np.zeros((num_records,), dtype=np.float32)

        for i, row in enumerate(df.itertuples(index=False)):
            ast = row.ast_dict
            resistant_classes = set()

            for t_idx, drug in enumerate(self.target_antibiotics):
                if drug in ast:
                    val = ast[drug]
                    Y_mask[i, t_idx] = True
                    if val == "R" or (self.intermediate_as_resistant and val == "I"):
                        Y[i, t_idx] = 1.0
                        drug_class = ANTIBIOTIC_CLASS_MAP.get(drug)
                        if drug_class:
                            resistant_classes.add(drug_class)
                    else:
                        Y[i, t_idx] = 0.0

            # Check MDR across all tested antibiotics in ast
            for drug, val in ast.items():
                if val == "R" or (self.intermediate_as_resistant and val == "I"):
                    d_class = ANTIBIOTIC_CLASS_MAP.get(drug)
                    if d_class:
                        resistant_classes.add(d_class)

            if len(resistant_classes) >= self.mdr_min_classes:
                MDR[i] = 1.0

        return (
            np.ascontiguousarray(Y, dtype=np.float32),
            np.ascontiguousarray(Y_mask, dtype=bool),
            np.ascontiguousarray(MDR, dtype=np.float32),
        )

    def save_feature_dictionary(self, out_path: Path):
        """Save feature dictionary for audit and interpretability."""
        meta = {
            "num_features": len(self.feature_names),
            "gene_vocab_size": len(self.gene_vocab),
            "feature_names": self.feature_names,
            "gene_vocab": self.gene_vocab,
            "target_antibiotics": self.target_antibiotics,
            "country_vocab": self.country_vocab,
            "source_vocab": self.source_vocab,
            "year_min": self.year_min,
            "year_max": self.year_max,
            "intermediate_as_resistant": self.intermediate_as_resistant,
            "mdr_min_classes": self.mdr_min_classes,
        }
        with open(out_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved feature dictionary to {out_path}")


def process_and_save_features(
    raw_path: Path,
    output_dir: Path,
    config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, FeaturePipeline]:
    """
    End-to-end execution of Phase 2 preprocessing and feature extraction.
    """
    from src.amr.data import load_raw_dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    clean_df = load_raw_dataset(raw_path)

    # Initialize and fit pipeline
    pipeline = FeaturePipeline(
        rare_gene_threshold_pct=config["data"].get("rare_gene_threshold_pct", 0.002),
        target_antibiotics=config["data"].get("target_antibiotics"),
        intermediate_as_resistant=config["data"].get("intermediate_as_resistant", True),
        mdr_min_classes=config["data"].get("mdr_min_classes", 3),
    )
    pipeline.fit(clean_df)

    # Transform features and labels
    X = pipeline.transform_features(clean_df)
    Y, Y_mask, MDR = pipeline.transform_labels(clean_df)

    # Persist processed artifacts
    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "Y.npy", Y)
    np.save(output_dir / "Y_mask.npy", Y_mask)
    np.save(output_dir / "mdr.npy", MDR)
    pipeline.save_feature_dictionary(output_dir / "feature_dictionary.json")

    # Save metadata summary
    clean_df[["isolate_id", "year", "country", "source", "host", "gene_count", "has_ast"]].to_csv(
        output_dir / "isolates_metadata.csv", index=False
    )

    logger.info(
        f"Phase 2 complete: Feature matrix shape {X.shape}, Labels shape {Y.shape}, "
        f"MDR positive rate: {MDR.mean()*100:.2f}% across {len(MDR):,} isolates."
    )
    return X, Y, Y_mask, MDR, pipeline
