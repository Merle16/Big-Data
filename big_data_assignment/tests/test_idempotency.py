"""Test that key pipeline transforms are idempotent."""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_add_base_features_idempotent():
    """Applying add_base_features twice should give the same result as once."""
    from pipeline.feature_engineering.f1_candidate_features import add_base_features

    df = pd.DataFrame({
        "primaryTitle":   ["The Matrix", "Inception", "Interstellar"],
        "originalTitle":  ["The Matrix", "Inception", "Interstellar"],
        "runtimeMinutes": [136.0, 148.0, None],
        "startYear":      [1999.0, 2010.0, 2014.0],
        "numVotes":       [1_800_000.0, 2_200_000.0, 1_600_000.0],
    })

    once  = add_base_features(df)
    twice = add_base_features(once)

    # Columns produced on the first call should survive the second call unchanged
    shared_cols = [c for c in once.columns if c in twice.columns]
    for col in shared_cols:
        if pd.api.types.is_numeric_dtype(once[col]):
            pd.testing.assert_series_equal(
                once[col].reset_index(drop=True),
                twice[col].reset_index(drop=True),
                check_names=False,
            )


def test_label_column_preserved():
    """Feature engineering must not drop or alter the label column."""
    p = Path(__file__).resolve().parents[1] / "pipeline/outputs/features/features_train_prepped.parquet"
    if not p.exists():
        pytest.skip("features_train_prepped.parquet not found — run pipeline first")
    df = pd.read_parquet(p)
    assert "label" in df.columns, "label column missing from features_train_prepped"
    assert df["label"].isin([0, 1]).all(), "label contains values other than 0/1"
    assert df["label"].isna().sum() == 0, "label has nulls"


def test_tconst_uniqueness_in_features():
    """tconst should be unique in feature matrix (no fanout from joins)."""
    p = Path(__file__).resolve().parents[1] / "pipeline/outputs/features/features_train_prepped.parquet"
    if not p.exists():
        pytest.skip("features_train_prepped.parquet not found — run pipeline first")
    df = pd.read_parquet(p)
    assert df["tconst"].nunique() == len(df), "duplicate tconst in feature matrix"
