"""Step 4: Drop duplicate rows based on unique or composite keys.

Reads key columns from enforce_schema.py. Falls back to auto-detection
if the table is not declared in the schema.
"""

import re

import duckdb

from .s0_enforce_schema import get_key

_ID_LIKE_RE = re.compile(r"^[a-zA-Z]{1,4}\d+$")


class Deduplicator:
    """Remove duplicate rows by key using DuckDB.

    Looks up key columns from the schema. If the table is not in the schema,
    falls back to auto-detecting a single unique ID column.
    """

    def _detect_id(self, con: duckdb.DuckDBPyConnection, table: str) -> tuple[str, ...] | None:
        """Fallback: return the first column that is ID-like AND actually unique."""
        total = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if total == 0:
            return None
        for col, dtype, *_ in con.execute(f"DESCRIBE {table}").fetchall():
            if "VARCHAR" not in dtype.upper():
                continue
            sample = con.execute(
                f'SELECT CAST("{col}" AS VARCHAR) FROM {table} WHERE "{col}" IS NOT NULL LIMIT 100'
            ).fetchdf().iloc[:, 0].tolist()
            if not sample or sum(_ID_LIKE_RE.match(str(v)) is not None for v in sample) / len(sample) <= 0.9:
                continue
            distinct = con.execute(f'SELECT COUNT(DISTINCT "{col}") FROM {table}').fetchone()[0]
            if distinct == total:
                return (col,)
        return None

    def transform(self, con: duckdb.DuckDBPyConnection, table: str) -> str:
        """Deduplicate on key columns. Returns the new view name."""
        key = get_key(table) or self._detect_id(con, table)
        if key is None:
            return table

        partition = ", ".join(f'"{k}"' for k in key)
        cols      = [f'"{r[0]}"' for r in con.execute(f"DESCRIBE {table}").fetchall()]
        out       = f"{table}_dedup"

        con.execute(f"""
            CREATE OR REPLACE VIEW {out} AS
            SELECT {', '.join(cols)} FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY {partition} ORDER BY {partition}) AS _rn
                FROM {table}
            ) WHERE _rn = 1
        """)

        before = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        after  = con.execute(f"SELECT COUNT(*) FROM {out}").fetchone()[0]
        if before != after:
            print(f"[dedup] {table}: {before - after} duplicates removed on ({', '.join(key)})")

        return out
