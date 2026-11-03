"""
tests/test_features.py
Tests for feature matrix dimensions, no NaNs, label construction, and absence of data leakage.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.amr.features import FeaturePipeline


@pytest.fixture
def sample_isolates_df():
    data = [
        {
            "isolate_id": f"iso_{i}",
            "year": 2015 + (i % 8),
            "country": "USA" if i % 2 == 0 else "UK",
            "source": "Blood" if i % 3 == 0 else "Urine",
            "host": "Human",
            "genes": ["blaKPC-2", "emrD", "fosA"] if i % 2 == 0 else ["blaCTX-M-15", "fosA"],
            "gene_count": 3 if i % 2 == 0 else 2,
            "ast_dict": {
                "meropenem": "R" if i % 2 == 0 else "S",
                "ceftazidime": "R",
                "ciprofloxacin": "R" if i % 3 == 0 else "S"
            },
            "has_ast": True
        }
        for i in range(100)
    ]
    return pd.DataFrame(data)


def test_feature_pipeline_fit_and_transform(sample_isolates_df):
    pipeline = FeaturePipeline(rare_gene_threshold_pct=0.01, min_gene_freq=2)
    pipeline.fit(sample_isolates_df)

    assert len(pipeline.gene_vocab) > 0
    assert len(pipeline.feature_names) > len(pipeline.gene_vocab)

    X = pipeline.transform_features(sample_isolates_df)
    assert X.shape[0] == 100
    assert X.shape[1] == len(pipeline.feature_names)
    assert X.dtype == np.float32
    assert X.flags.c_contiguous
    assert not np.isnan(X).any(), "Feature matrix must not contain any NaNs"


def test_label_construction_and_no_leakage(sample_isolates_df):
    pipeline = FeaturePipeline(rare_gene_threshold_pct=0.01, min_gene_freq=2)
    pipeline.fit(sample_isolates_df)

    X = pipeline.transform_features(sample_isolates_df)
    Y, Y_mask, MDR = pipeline.transform_labels(sample_isolates_df)

    assert Y.shape[0] == 100
    assert Y_mask.shape[0] == 100
    assert MDR.shape[0] == 100
    assert not np.isnan(Y).any()
    assert not np.isnan(MDR).any()

    # Leakage check: verify no feature name contains any target antibiotic name
    for feat_name in pipeline.feature_names:
        for target in pipeline.target_antibiotics:
            assert feat_name != target, f"Feature {feat_name} leaks target label {target}"
            assert feat_name != f"label_{target}", f"Feature {feat_name} leaks target label {target}"
