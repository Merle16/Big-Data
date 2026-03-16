"""
Modular DuckDB-based data cleaning pipeline.
Input:  data/raw/csv/* and data/raw/IMDB_external_csv/*
Output: pipeline/outputs/cleaning/{train,validation_hidden,test_hidden}_clean.parquet

Pipeline order
--------------
  schema.py  — single source of truth: keys, IDs, UUID regex, drop cols
  steps.py   — s1 MissingTokenReplacer  → s2 DTypeEnforcer → s3 StringStandardizer
             → s4 Deduplicator → s5 JoinBuilder → s6 Normalizer
             → s7 MICEImputer  → s8 assert_quality / save_parquet
  report.py  — post-pipeline validity checks + figures (calls utils/)
  utils/     — figures.py · audits.py · quality_report.py
"""

from pathlib import Path

import duckdb

from .schema import get_drop_cols, validate
from .steps import (
    DISGUISED_TOKENS, MissingTokenReplacer,
    DTypeEnforcer,
    StringStandardizer,
    Deduplicator,
    JoinBuilder,
    Normalizer,
    MICEImputer,
    assert_quality, save_parquet,
)

_ROOT    = Path(__file__).resolve().parents[2]
RAW_CSV  = _ROOT / "data" / "raw" / "csv"
RAW_EXT  = _ROOT / "data" / "raw" / "IMDB_external_csv"
OUT_DIR  = _ROOT / "pipeline" / "outputs" / "cleaning"


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
    all_issues: list[str] = []
    for tbl, view in cleaned.items():
        all_issues.extend(validate(con, view))
    if all_issues:
        for msg in all_issues:
            print(f"  [FAIL] {msg}")
        raise ValueError("Schema validation failed:\n" + "\n".join(f"  • {m}" for m in all_issues))
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

    # ── 7. Post-pipeline validity checks + figures (s9) ─────────────────────
    _OUTPUTS_CLEAN = _ROOT / "pipeline" / "outputs" / "cleaning"
    from .report import run as _s9_run
    _s9_run(out_paths, RAW_CSV, fig_dir=_OUTPUTS_CLEAN)

    return out_paths


if __name__ == "__main__":
    paths = run_pipeline()
    print("\n[pipeline] ALL STEPS COMPLETE.")
    for split, path in paths.items():
        print(f"  {split}: {path}")
