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

        if total == 0 or numeric / total < 0.5:
            return None
        if boolean / total > 0.95:
            return "BOOLEAN"
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
        for col, dtype, *_ in cols:
            q, s = f'"{col}"', f'TRIM(CAST("{col}" AS VARCHAR))'
            dt = dtype.upper()

            if "VARCHAR" in dt:
                t = self._infer_varchar(con, table, col)
                exprs.append(f"TRY_CAST({s} AS {t}) AS {q}" if t else q)
            elif "BIGINT" in dt:
                t = self._infer_bigint(con, table, col)
                exprs.append(f"CAST({q} AS {t}) AS {q}" if t else q)
            else:
                exprs.append(q)

        out = f"{table}_typed"
        con.execute(f"CREATE OR REPLACE VIEW {out} AS SELECT {', '.join(exprs)} FROM {table}")
        return out
