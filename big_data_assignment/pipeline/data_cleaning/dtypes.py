"""Step 2: Cast VARCHAR columns to the correct numeric or year type using DuckDB TRY_CAST."""

import duckdb

class DTypeEnforcer:
    """Cast VARCHAR columns to DOUBLE (numeric) or INTEGER (year) where the data supports it.
    ID-like and text columns remain VARCHAR. Non-VARCHAR columns pass through unchanged."""

    def __init__(self, year_range: tuple[int, int] = YEAR_RANGE):
        self.year_range = year_range

    def _target_type(self, con: duckdb.DuckDBPyConnection, table: str, col: str) -> str | None:
        """Return 'year', 'double', or None (keep VARCHAR) based on column content."""
        q = f'"{col}"'
        s = f"TRIM(CAST({q} AS VARCHAR))"
        lo, hi = self.year_range

        total, numeric, year = con.execute(f"""
            SELECT
                COUNT({q}),
                SUM(CASE WHEN TRY_CAST({s} AS DOUBLE)  IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN TRY_CAST({s} AS INTEGER) BETWEEN {lo} AND {hi} THEN 1 ELSE 0 END)
            FROM {table} WHERE {q} IS NOT NULL
        """).fetchone()

        if total == 0 or numeric / total < 0.5:
            return None
        return "year" if year / total > 0.9 else "double"

    def transform(self, con: duckdb.DuckDBPyConnection, table: str) -> str:
        """Create a typed view. Returns the new view name."""
        cols = con.execute(f"DESCRIBE {table}").fetchall()
        exprs = []
        for col, dtype, *_ in cols:
            q = f'"{col}"'
            s = f"TRIM(CAST({q} AS VARCHAR))"
            if "VARCHAR" in dtype.upper():
                target = self._target_type(con, table, col)
                if target == "year":
                    exprs.append(f"TRY_CAST({s} AS INTEGER) AS {q}")
                elif target == "double":
                    exprs.append(f"TRY_CAST({s} AS DOUBLE) AS {q}")
                else:
                    exprs.append(q)
            else:
                exprs.append(q)

        out = f"{table}_typed"
        con.execute(f"CREATE OR REPLACE VIEW {out} AS SELECT {', '.join(exprs)} FROM {table}")
        return out
