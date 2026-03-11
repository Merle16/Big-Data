"""Step 1: Replace disguised missing tokens with SQL NULL (NaN-compatible)."""

import duckdb

# ── Human-in-the-loop: add any dataset-specific disguised-missing tokens here.
# Identified via quality-report checks before running the pipeline.
DISGUISED_TOKENS: tuple[str, ...] = ("\\N", "\\\\N")


class MissingTokenReplacer:
    """Replace disguised missing-value tokens with NULL in all columns using DuckDB.
    Optionally drop columns that are too sparse to be useful (e.g. endYear in train CSVs)."""

    def __init__(self, tokens: tuple[str, ...] = (), drop_cols: tuple[str, ...] = ()):
        self.tokens = tokens
        self.drop_cols = set(drop_cols)
        self._in_list = ", ".join(f"'{t.lower()}'" for t in tokens)

    def transform(self, con: duckdb.DuckDBPyConnection, table: str) -> str:
        """Create a view with disguised tokens replaced by NULL. Returns the new view name."""
        cols = con.execute(f"DESCRIBE {table}").fetchall()
        exprs = []
        for name, *_ in cols:
            if name in self.drop_cols:
                continue
            q = f'"{name}"'
            exprs.append(
                f"CASE WHEN LOWER(TRIM(CAST({q} AS VARCHAR))) IN ({self._in_list}) "
                f"THEN NULL ELSE {q} END AS {q}"
            )

        out = f"{table}_no_missing"
        con.execute(
            f"CREATE OR REPLACE VIEW {out} AS SELECT {', '.join(exprs)} FROM {table}"
        )
        return out
