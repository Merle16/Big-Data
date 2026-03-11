"""
Modular DuckDB-based data cleaning pipeline.
Input:  data/raw/csv/* and data/raw/IMDB_external_csv/*
Output: data/processed/joined_clean.parquet

Pipeline order
--------------
s0. s0_enforce_schema.py  — single source of truth: keys, IDs, UUID regex, drop cols
s1. s1_missing.py         — replace disguised missing tokens (e.g. "\\N") with NULL
s2. s2_dtypes.py          — cast all columns to correct types
s3. s3_standardization.py — NFKD Unicode; string & categorical standardization
s4. s4_deduplication.py   — drop duplicate rows by key (from schema)
s5. s5_join.py            — schema-specific joins (1:1 + aggregated M:1 with person metadata)
s6. s6_normalization.py   — distribution transforms (log, sqrt) for MICE prep
s7. s7_imputation.py      — MICE imputation for remaining missing values
s8. s8_save_output.py     — export to parquet
"""

from pathlib import Path

import duckdb

from .s0_enforce_schema import get_drop_cols
from .s1_missing import MissingTokenReplacer
from .s2_dtypes import DTypeEnforcer
from .s3_standardization import StringStandardizer
from .s4_deduplication import Deduplicator
from .s5_join import JoinBuilder

_ROOT = Path(__file__).resolve().parents[2]

# ── Human-in-the-loop configurations after quality-report checks!
DISGUISED_TOKENS = ("\\N", "\\\\N")


def run_pipeline(con: duckdb.DuckDBPyConnection, table: str) -> str:
    """Steps 1–4: clean one table. Returns the final view name."""
    drop  = get_drop_cols(table)
    table = MissingTokenReplacer(tokens=DISGUISED_TOKENS, drop_cols=drop).transform(con, table)
    table = DTypeEnforcer().transform(con, table)
    table = StringStandardizer().transform(con, table)
    table = Deduplicator().transform(con, table)
    # 5. normalization
    # 6. imputation
    return table
