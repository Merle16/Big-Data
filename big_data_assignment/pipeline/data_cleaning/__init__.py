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
