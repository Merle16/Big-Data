"""
Modular DuckDB-based data cleaning pipeline.
Input:  data/raw/csv/* and data/raw/IMDB_external_csv/*
Output: data/processed/csv/*

Pipeline order
--------------
1. missing.py         — replace disguised missing tokens (e.g. "\\N", "N/A") with NaN
2. dtypes.py          — cast columns to correct types (numeric, year, UUID); assert schema
3. standardization.py — NFKD Unicode; string & categorical standardization
4. deduplication.py   — drop duplicate rows by UUID (same title + start year = same movie)
5. normalization.py   — distribution transforms (log, sqrt) for MICE prep
6. imputation.py      — MICE imputation for remaining missing values
"""

from .missing import MissingTokenReplacer
from .dtypes import DTypeEnforcer
from .standardization import StringStandardizer


# before running the pipeline, run the quality report to detect quality errors.
# human-in-the-loop configurations

DISGUISED_TOKENS = ("\\N", "\\\\N")
TRAIN_DROP_COLS  = ("endYear",)
YEAR_RANGE       = (1900, 2026)


def run_pipeline(con, table: str, is_train: bool = True) -> str:
    """Run all cleaning steps sequentially. Returns final view name."""
    drop = TRAIN_DROP_COLS if is_train else ()
    table = MissingTokenReplacer(tokens=DISGUISED_TOKENS, drop_cols=drop).transform(con, table)
    table = DTypeEnforcer(year_range=YEAR_RANGE).transform(con, table)
    table = StringStandardizer().transform(con, table)
    # 4. deduplication
    # 5. normalization
    # 6. imputation
    return table
