"""Unit tests for data contract validation."""
import pytest
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.data_cleaning.schema import COLUMN_CONTRACTS, validate_contracts, SCHEMA_VERSION


def test_schema_version_exists():
    assert SCHEMA_VERSION is not None
    assert isinstance(SCHEMA_VERSION, str)


def test_valid_df_passes():
    df = pd.DataFrame({
        "tconst": ["tt0000001", "tt0000002"],
        "startYear": [1990.0, 2005.0],
        "label": [0, 1],
    })
    warnings = validate_contracts(df, split="test")
    assert warnings == [], f"Expected no warnings, got: {warnings}"


def test_invalid_label_flagged():
    df = pd.DataFrame({
        "tconst": ["tt0000001"],
        "label": [2],  # invalid: not 0 or 1
    })
    warnings = validate_contracts(df, split="test")
    assert any("label" in w for w in warnings)


def test_null_tconst_flagged():
    df = pd.DataFrame({
        "tconst": [None, "tt0000002"],
        "label": [0, 1],
    })
    warnings = validate_contracts(df, split="test")
    assert any("tconst" in w for w in warnings)


def test_out_of_range_year_flagged():
    df = pd.DataFrame({
        "startYear": [1800.0],  # below 1880 min
    })
    warnings = validate_contracts(df, split="test")
    assert any("startYear" in w for w in warnings)
