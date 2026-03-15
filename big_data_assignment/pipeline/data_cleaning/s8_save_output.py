"""Step 8: Quality gate assertions + export to Parquet.

Raises ValueError if any schema violations remain after cleaning.
Then materialises the DuckDB view to a Parquet file.
"""

from pathlib import Path

import duckdb

from .s0_enforce_schema import validate

# After MICE these must have zero NULLs
_IMPUTED_COLS: tuple[str, ...] = ("startYear", "runtimeMinutes", "numVotes")

# Hard numeric range bounds [inclusive]
_RANGE_CHECKS: dict[str, tuple[float, float]] = {
    "startYear":      (1888, 2030),
    "runtimeMinutes": (1,    1440),
}

# Must be non-negative (numVotes is log1p-transformed so 0 is valid)
_NON_NEGATIVE: tuple[str, ...] = ("numVotes", "dir_count", "wri_count")

# Joined columns: warn if NULL fraction exceeds threshold.
# Death-year columns excluded — ~70% NULL is expected (living people).
_JOIN_COVERAGE_WARN: dict[str, float] = {
    "genres":             0.10,
    "titleType":          0.10,
    "isAdult":            0.10,
    "dir_avg_birth_year": 0.60,
    "wri_avg_birth_year": 0.60,
    "dir_professions":    0.20,
    "wri_professions":    0.20,
}


def assert_quality(con: duckdb.DuckDBPyConnection, table: str) -> None:
    """Raise ValueError if the cleaned table still has data-quality violations.

    Hard errors (raise):
      1. UUID format + key uniqueness  (s0_enforce_schema.validate)
      2. Post-imputation nulls         (startYear, runtimeMinutes, numVotes)
      3. Numeric range bounds          (startYear, runtimeMinutes)
      4. Non-negativity                (numVotes, dir_count, wri_count)
      5. Table must not be empty

    Warnings (print only — expected IMDB data gaps, not pipeline failures):
      6. Join-coverage check on joined columns
    """
    issues = validate(con, table)

    existing = {r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()}
    total    = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    # 2. Post-imputation: must be NULL-free
    for col in _IMPUTED_COLS:
        if col not in existing:
            continue
        nulls = con.execute(
            f'SELECT COUNT(*) FROM {table} WHERE "{col}" IS NULL'
        ).fetchone()[0]
        if nulls:
            issues.append(f"{table}.{col}: {nulls} NULLs remain after imputation")

    # 3. Range checks
    for col, (lo, hi) in _RANGE_CHECKS.items():
        if col not in existing:
            continue
        bad = con.execute(f"""
            SELECT COUNT(*) FROM {table}
            WHERE "{col}" IS NOT NULL AND ("{col}" < {lo} OR "{col}" > {hi})
        """).fetchone()[0]
        if bad:
            issues.append(f"{table}.{col}: {bad} rows outside [{lo}, {hi}]")

    # 4. Non-negativity
    for col in _NON_NEGATIVE:
        if col not in existing:
            continue
        bad = con.execute(
            f'SELECT COUNT(*) FROM {table} WHERE "{col}" IS NOT NULL AND "{col}" < 0'
        ).fetchone()[0]
        if bad:
            issues.append(f"{table}.{col}: {bad} negative values")

    # 5. Non-empty
    if total == 0:
        issues.append(f"{table}: table is empty")

    if issues:
        raise ValueError("Quality gate failed:\n" + "\n".join(f"  • {i}" for i in issues))

    # 6. Join-coverage warnings (informational — expected IMDB gaps, not errors)
    for col, max_null_frac in _JOIN_COVERAGE_WARN.items():
        if col not in existing or total == 0:
            continue
        nulls = con.execute(
            f'SELECT COUNT(*) FROM {table} WHERE "{col}" IS NULL'
        ).fetchone()[0]
        frac = nulls / total
        if frac > max_null_frac:
            print(f"  [WARN] {col}: {frac:.1%} NULL (threshold {max_null_frac:.0%}) — IMDB gap")


def save_parquet(con: duckdb.DuckDBPyConnection, view: str, path: Path) -> Path:
    """Materialise a DuckDB view to Parquet, dropping internal __fp_* columns.

    __fp_* columns are fingerprint helpers added by s3_standardization for
    deduplication.  They are pipeline-internal and should not appear in output.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Drop pipeline-internal columns that are not useful as ML features:
    #   __fp_*  — fingerprint helpers from s3 standardization
    #   dir_ids / wri_ids — raw comma-separated nconst strings
    _DROP = {"dir_ids", "wri_ids"}
    all_cols = [r[0] for r in con.execute(f"DESCRIBE {view}").fetchall()]
    keep = [f'"{c}"' for c in all_cols if not c.startswith("__fp_") and c not in _DROP]
    con.execute(f"COPY (SELECT {', '.join(keep)} FROM {view}) TO '{path}' (FORMAT PARQUET)")

    n = con.execute(f"SELECT COUNT(*) FROM '{path}'").fetchone()[0]
    dropped = len(all_cols) - len(keep)
    suffix = f"  (dropped {dropped} __fp_* cols)" if dropped else ""
    print(f"[save] {view:<45} → {path.name}  ({n:,} rows){suffix}")
    return path
