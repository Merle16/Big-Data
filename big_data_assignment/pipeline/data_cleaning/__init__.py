"""
Modular DuckDB-based data cleaning pipeline.
Input:  data/raw/csv/* and data/raw/IMDB_external_csv/*
Output: data/processed/{train,validation_hidden,test_hidden}_clean.parquet

Pipeline order
--------------
s0. s0_enforce_schema.py  — single source of truth: keys, IDs, UUID regex, drop cols
s1. s1_missing.py         — replace disguised missing tokens (e.g. "\\N") with NULL
s2. s2_dtypes.py          — cast all columns to correct types
s3. s3_standardization.py — NFKD Unicode; string & categorical standardization
s4. s4_deduplication.py   — drop duplicate rows by key (from schema)
s5. s5_join.py            — schema-specific joins (1:1 + aggregated M:1 with person metadata)
s6. s6_normalization.py   — log1p transform for heavy-tailed numerics (numVotes)
s7. s7_imputation.py      — median imputation for remaining missing values
s8. s8_save_output.py     — quality gate assertions + export to parquet
"""

from pathlib import Path

import duckdb

from .s0_enforce_schema import get_drop_cols, validate
from .s1_missing import DISGUISED_TOKENS, MissingTokenReplacer
from .s2_dtypes import DTypeEnforcer
from .s3_standardization import StringStandardizer
from .s4_deduplication import Deduplicator
from .s5_join import JoinBuilder
from .s6_normalization import Normalizer
from .s7_imputation import MICEImputer
from .s8_save_output import assert_quality, save_parquet

_ROOT    = Path(__file__).resolve().parents[2]
RAW_CSV  = _ROOT / "data" / "raw" / "csv"
RAW_EXT  = _ROOT / "data" / "raw" / "IMDB_external_csv"
OUT_DIR  = _ROOT / "data" / "processed"


# ── Per-table cleaning (steps 1–4) ───────────────────────────────────────────

def clean_table(con: duckdb.DuckDBPyConnection, table: str) -> str:
    """Run steps 1–4 on a single table. Returns the final cleaned view name."""
    drop  = get_drop_cols(table)
    table = MissingTokenReplacer(tokens=DISGUISED_TOKENS, drop_cols=drop).transform(con, table)
    table = DTypeEnforcer().transform(con, table)
    table = StringStandardizer().transform(con, table)
    table = Deduplicator().transform(con, table)
    return table


# ── Full end-to-end pipeline ──────────────────────────────────────────────────

def run_pipeline(out_dir: Path | None = None) -> dict[str, Path]:
    """Load, clean, join, normalise, impute, and export all splits.

    Parameters
    ----------
    out_dir : directory for output Parquet files (default: data/processed/)

    Returns
    -------
    dict mapping split name → written Parquet path.
    """
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    con     = duckdb.connect()

    # ── 1. Ingest raw CSVs ───────────────────────────────────────────────────

    # Train: UNION of all 8 CSV splits (read_csv_auto glob).
    # Do NOT exclude column0 here — s1 drops it safely via schema drop_cols,
    # so we never error if a particular CSV happens to lack it.
    train_glob = str(RAW_CSV / "train-*.csv")
    con.execute(f"""
        CREATE OR REPLACE TABLE train AS
        SELECT * FROM read_csv_auto('{train_glob}', header=True, all_varchar=True, ignore_errors=True)
    """)

    for name, fname in [
        ("validation_hidden", "validation_hidden.csv"),
        ("test_hidden",       "test_hidden.csv"),
    ]:
        con.execute(f"""
            CREATE OR REPLACE TABLE {name} AS
            SELECT * FROM read_csv_auto('{RAW_CSV / fname}', header=True, all_varchar=True, ignore_errors=True)
        """)

    # IMDB reference tables
    for stem in ("title_basics", "title_crew", "title_principals", "name_basics"):
        con.execute(f"""
            CREATE OR REPLACE TABLE {stem} AS
            SELECT * FROM read_csv_auto('{RAW_EXT / (stem + ".csv")}', header=True, all_varchar=True)
        """)

    # Edge tables: filter out \N / \\N person IDs at ingestion (same as ilesh F6)
    for stem, person_col in [("movie_directors", "director"), ("movie_writers", "writer")]:
        con.execute(f"""
            CREATE OR REPLACE TABLE {stem} AS
            SELECT * FROM read_csv_auto('{RAW_CSV / (stem + ".csv")}', header=True, all_varchar=True)
            WHERE "{person_col}" IS NOT NULL
              AND TRIM("{person_col}") NOT IN ('\\N', '\\\\N')
              AND TRIM("{person_col}") != ''
        """)

    # ── 2. Clean each table (s1–s4) ─────────────────────────────────────────

    all_tables = [
        "train", "validation_hidden", "test_hidden",
        "title_basics", "title_crew", "title_principals", "name_basics",
        "movie_directors", "movie_writers",
    ]
    cleaned: dict[str, str] = {}
    for tbl in all_tables:
        cleaned[tbl] = clean_table(con, tbl)
        print(f"[clean] {tbl:<25} → {cleaned[tbl]}")

    # ── 3. Schema validation (s0 validate) ──────────────────────────────────

    print("\n[validate] Running schema checks...")
    any_issues = False
    for tbl, view in cleaned.items():
        issues = validate(con, view)
        for msg in issues:
            print(f"  [WARN] {msg}")
            any_issues = True
    if not any_issues:
        print("  All checks passed.")

    # ── 4. Join + normalize each split (s5, s6) ─────────────────────────────

    builder   = JoinBuilder(cleaned)
    normalizer = Normalizer()
    joined_views: dict[str, str] = {}

    for split in ("train", "validation_hidden", "test_hidden"):
        print(f"\n[join] Building wide view for split: {split}")
        base   = cleaned[split]
        joined = builder.transform(con, base, out=f"{split}_joined")
        joined_views[split] = normalizer.transform(con, joined)

    # ── 5. MICE imputation (s7): fit on train only, apply to all splits ──────
    # Fitting on val/test would leak their distributions into imputed values.
    print()
    imputer = MICEImputer()
    imputer.fit(con, joined_views["train"])

    out_paths: dict[str, Path] = {}
    for split in ("train", "validation_hidden", "test_hidden"):
        imputed = imputer.transform(con, joined_views[split], suffix=split)

        # ── 6. Quality gate + save (s8)
        assert_quality(con, imputed)
        path = save_parquet(con, imputed, out_dir / f"{split}_clean.parquet")
        out_paths[split] = path

    con.close()
    return out_paths


if __name__ == "__main__":
    paths = run_pipeline()
    print("\n[pipeline] ALL STEPS COMPLETE.")
    for split, path in paths.items():
        print(f"  {split}: {path}")
