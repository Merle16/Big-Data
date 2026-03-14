"""Step 7: MICE imputation (sklearn IterativeImputer) for missing numeric values.

Fit on the TRAIN joined table only, then transform val/test with the same
fitted model — zero leakage of val/test distributions into imputed values.

Usage:
    imputer = MICEImputer()
    imputer.fit(con, train_view)          # fits on train
    train_out = imputer.transform(con, train_view,  suffix="train")
    val_out   = imputer.transform(con, val_view,    suffix="val")
    test_out  = imputer.transform(con, test_view,   suffix="test")
"""

import duckdb
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

# Numeric columns to impute.  Only applied if the column exists in the table.
_IMPUTE_COLS: tuple[str, ...] = ("startYear", "runtimeMinutes", "numVotes")

# Columns that must be integers after imputation (MICE produces floats).
_ROUND_TO_INT: frozenset[str] = frozenset({"startYear", "runtimeMinutes"})

_NUMERIC_TYPES = ("INTEGER", "BIGINT", "DOUBLE", "FLOAT", "HUGEINT", "DECIMAL")


class MICEImputer:
    """Fit MICE on train, apply to any split — no leakage.

    Only numeric columns listed in _IMPUTE_COLS are imputed.
    All other columns pass through unchanged.
    """

    def __init__(self, max_iter: int = 10, random_state: int = 42) -> None:
        self._imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
        self._cols_to_impute: list[str] = []

    def _numeric_impute_cols(self, con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
        """Return the subset of _IMPUTE_COLS that are present and numeric in table."""
        return [
            col for col, dtype, *_ in con.execute(f"DESCRIBE {table}").fetchall()
            if col in _IMPUTE_COLS and any(t in dtype.upper() for t in _NUMERIC_TYPES)
        ]

    def fit(self, con: duckdb.DuckDBPyConnection, train_table: str) -> "MICEImputer":
        """Fit MICE on the training table. Medians etc. are learnt from train only."""
        self._cols_to_impute = self._numeric_impute_cols(con, train_table)
        if not self._cols_to_impute:
            print("[impute] No numeric columns to impute — skipping fit.")
            return self

        sel = ", ".join(f'"{c}"' for c in self._cols_to_impute)
        df = con.execute(f"SELECT {sel} FROM {train_table}").fetchdf()

        self._imputer.fit(df)
        print(f"[impute] MICE fitted on {train_table}  cols={self._cols_to_impute}")
        return self

    def transform(self, con: duckdb.DuckDBPyConnection, table: str, suffix: str = "") -> str:
        """Impute missing values using the fitted MICE model. Returns new view name."""
        if not self._cols_to_impute:
            return table

        # Fetch only the columns to impute
        present = [c for c in self._cols_to_impute if c in {
            r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()
        }]
        if not present:
            return table

        sel = ", ".join(f'"{c}"' for c in present)
        df = con.execute(f"SELECT {sel} FROM {table}").fetchdf()

        imputed_arr = self._imputer.transform(df)
        df_imputed  = pd.DataFrame(imputed_arr, columns=present)

        # Round integer-valued columns — MICE produces floats (e.g. 1996.222)
        for col in present:
            if col in _ROUND_TO_INT:
                df_imputed[col] = df_imputed[col].round().astype("Int64")

        # Register the imputed columns as a temp table, then join back on rowid
        tmp = f"_mice_imputed_{suffix or table.replace('-', '_')}"
        df_imputed["_rowid"] = range(len(df_imputed))
        con.register(tmp, df_imputed)

        # Build a view: original table with imputed columns overwritten
        all_cols = [(r[0], r[1]) for r in con.execute(f"DESCRIBE {table}").fetchall()]
        exprs = []
        for col, dtype in all_cols:
            if col in present:
                exprs.append(f'{tmp}."{col}" AS "{col}"')
            else:
                exprs.append(f'base."{col}"')

        out = f"{table}_imputed"
        con.execute(f"""
            CREATE OR REPLACE VIEW {out} AS
            SELECT {', '.join(exprs)}
            FROM (SELECT *, ROW_NUMBER() OVER () - 1 AS _rowid FROM {table}) base
            JOIN {tmp} ON base._rowid = {tmp}._rowid
        """)
        print(f"[impute] MICE transformed {table} → {out}  cols={present}")
        return out
