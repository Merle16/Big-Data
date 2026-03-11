"""
Modular DuckDB-based data cleaning pipeline.
Input:  data/raw/csv/* and data/raw/IMDB_external_csv/*
Output: data/processed/joined_clean.parquet

Pipeline order
--------------
1. missing.py         — replace disguised missing tokens (e.g. "\\N") with NULL
2. dtypes.py          — cast all columns to correct types
3. standardization.py — NFKD Unicode; string & categorical standardization
4. deduplication.py   — drop duplicate rows by UUID
4½. join.py           — schema-specific joins (1:1 + aggregated M:1 with person metadata)
5. normalization.py   — distribution transforms (log, sqrt) for MICE prep
6. imputation.py      — MICE imputation for remaining missing values

Run
---
    python -m pipeline.data_cleaning
"""

from pathlib import Path

import duckdb

from .missing import MissingTokenReplacer
from .dtypes import DTypeEnforcer
from .standardization import StringStandardizer
from .join import JoinBuilder

_ROOT = Path(__file__).resolve().parents[2]

# ── Human-in-the-loop configurations 
DISGUISED_TOKENS = ("\\N", "\\\\N")
TRAIN_DROP_COLS  = ("endYear",)      # 90%+ missing in train CSVs, near-zero signal


def run_pipeline(con: duckdb.DuckDBPyConnection, table: str, is_train: bool = True) -> str:
    """Steps 1–4: clean one table. Returns the final view name."""
    drop  = TRAIN_DROP_COLS if is_train else ()
    table = MissingTokenReplacer(tokens=DISGUISED_TOKENS, drop_cols=drop).transform(con, table)
    table = DTypeEnforcer().transform(con, table)
    table = StringStandardizer().transform(con, table)
    # 4. deduplication
    # 5. normalization
    # 6. imputation
    return table
