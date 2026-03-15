"""Step 2: Cast all columns to their most specific correct type using DuckDB TRY_CAST."""

import duckdb


class DTypeEnforcer:
    """Infer and cast each column to its most specific correct type.

    VARCHAR  → probed for BOOLEAN, INTEGER, DOUBLE (in that order of specificity).
    BIGINT   → downcast to BOOLEAN (0/1 only) or INTEGER (fits range).
    All other types pass through unchanged.
    """

    def _infer_varchar(self, con: duckdb.DuckDBPyConnection, table: str, col: str) -> str | None:
        q, s = f'"{col}"', f'TRIM(CAST("{col}" AS VARCHAR))'
        total, boolean, numeric, whole = con.execute(f"""
            SELECT
                COUNT({q}),
                SUM(CASE WHEN LOWER({s}) IN ('true', 'false') THEN 1 ELSE 0 END),
                SUM(CASE WHEN TRY_CAST({s} AS DOUBLE)  IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN TRY_CAST({s} AS INTEGER) IS NOT NULL THEN 1 ELSE 0 END)
            FROM {table} WHERE {q} IS NOT NULL
        """).fetchone()

        if total == 0:
            return None
        # Check boolean first — "True"/"False" strings are not numeric,
        # so the numeric gate must come after.
        if boolean / total > 0.95:
            return "BOOLEAN"
        if numeric / total < 0.5:
            return None
        return "INTEGER" if whole / total > 0.9 else "DOUBLE"

    def _infer_bigint(self, con: duckdb.DuckDBPyConnection, table: str, col: str) -> str | None:
        q = f'"{col}"'
        total, binary, fits = con.execute(f"""
            SELECT
                COUNT({q}),
                SUM(CASE WHEN {q} IN (0, 1) THEN 1 ELSE 0 END),
                SUM(CASE WHEN {q} BETWEEN -2147483648 AND 2147483647 THEN 1 ELSE 0 END)
            FROM {table} WHERE {q} IS NOT NULL
        """).fetchone()

        if total == 0:
            return None
        if binary / total > 0.99:
            return "BOOLEAN"
        if fits / total > 0.99:
            return "INTEGER"
        return None

    def transform(self, con: duckdb.DuckDBPyConnection, table: str) -> str:
        """Create a typed view. Returns the new view name."""
        cols = con.execute(f"DESCRIBE {table}").fetchall()
        exprs = []
        cast_cols: dict[str, str] = {}   # col → target type, for post-creation audit

        for col, dtype, *_ in cols:
            q, s = f'"{col}"', f'TRIM(CAST("{col}" AS VARCHAR))'
            dt = dtype.upper()

            if "VARCHAR" in dt:
                t = self._infer_varchar(con, table, col)
                if t:
                    exprs.append(f"TRY_CAST({s} AS {t}) AS {q}")
                    cast_cols[col] = t
                else:
                    exprs.append(q)
            elif "BIGINT" in dt:
                t = self._infer_bigint(con, table, col)
                if t:
                    # TRY_CAST (not CAST) so the <1% outside range become NULL
                    # instead of overflowing silently.
                    exprs.append(f"TRY_CAST({q} AS {t}) AS {q}")
                    cast_cols[col] = t
                else:
                    exprs.append(q)
            else:
                exprs.append(q)

        out = f"{table}_typed"
        con.execute(f"CREATE OR REPLACE VIEW {out} AS SELECT {', '.join(exprs)} FROM {table}")

        # Audit: report any nulls introduced by TRY_CAST that weren't null before.
        # These are non-missing values that failed to parse — silent data loss.
        for col, target_type in cast_cols.items():
            q = f'"{col}"'
            before = con.execute(f"SELECT COUNT(*) FROM {table} WHERE {q} IS NOT NULL").fetchone()[0]
            after  = con.execute(f"SELECT COUNT(*) FROM {out}  WHERE {q} IS NOT NULL").fetchone()[0]
            lost   = before - after
            if lost > 0:
                print(
                    f"  [WARN] s2 TRY_CAST {col} → {target_type}: "
                    f"{lost} non-null values failed to parse and became NULL"
                )

        return out
