"""Step 6: Distribution transforms (log1p) for heavy-tailed numeric columns.

Applied before MICE so that imputed values are on a more Gaussian-like scale.
numVotes is extremely right-skewed (range 1k–2.5M), so log1p is standard.
"""

import duckdb

# Columns that benefit from log1p compression before imputation / modelling.
# Only applied if the column exists and has a numeric type.
_LOG1P_COLS: frozenset[str] = frozenset({"numVotes"})

_NUMERIC_TYPES = ("INTEGER", "BIGINT", "DOUBLE", "FLOAT", "HUGEINT", "DECIMAL")


class Normalizer:
    """Apply log1p to declared heavy-tailed columns. All other columns pass through."""

    def transform(self, con: duckdb.DuckDBPyConnection, table: str) -> str:
        cols = con.execute(f"DESCRIBE {table}").fetchall()
        exprs = []
        for col, dtype, *_ in cols:
            q  = f'"{col}"'
            dt = dtype.upper()
            if col in _LOG1P_COLS and any(t in dt for t in _NUMERIC_TYPES):
                exprs.append(f"LN(1 + {q}) AS {q}")
            else:
                exprs.append(q)

        out = f"{table}_norm"
        con.execute(
            f"CREATE OR REPLACE VIEW {out} AS SELECT {', '.join(exprs)} FROM {table}"
        )
        return out
