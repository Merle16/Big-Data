"""Step 8: Quality gate assertions + export to Parquet.

Raises ValueError if any schema violations remain after cleaning.
Then materialises the DuckDB view to a Parquet file.
"""

from pathlib import Path

import duckdb

from .s0_enforce_schema import validate


def assert_quality(con: duckdb.DuckDBPyConnection, table: str) -> None:
    """Raise ValueError if the cleaned table still has schema violations."""
    issues = validate(con, table)
    if issues:
        raise ValueError("Quality gate failed:\n" + "\n".join(f"  • {i}" for i in issues))


def save_parquet(con: duckdb.DuckDBPyConnection, view: str, path: Path) -> Path:
    """Materialise a DuckDB view to Parquet, dropping internal __fp_* columns.

    __fp_* columns are fingerprint helpers added by s3_standardization for
    deduplication.  They are pipeline-internal and should not appear in output.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    all_cols = [r[0] for r in con.execute(f"DESCRIBE {view}").fetchall()]
    keep = [f'"{c}"' for c in all_cols if not c.startswith("__fp_")]
    con.execute(f"COPY (SELECT {', '.join(keep)} FROM {view}) TO '{path}' (FORMAT PARQUET)")

    n = con.execute(f"SELECT COUNT(*) FROM '{path}'").fetchone()[0]
    dropped = len(all_cols) - len(keep)
    suffix = f"  (dropped {dropped} __fp_* cols)" if dropped else ""
    print(f"[save] {view:<45} → {path.name}  ({n:,} rows){suffix}")
    return path
