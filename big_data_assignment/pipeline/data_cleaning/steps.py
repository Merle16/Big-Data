"""Pipeline transformer steps — all DuckDB transformer classes in one place.

Each class follows the same interface:
    .transform(con: DuckDBPyConnection, table: str) -> str
        Creates a new DuckDB view and returns its name.

Steps
-----
  s1  MissingTokenReplacer  — replace disguised-null tokens (\\N, etc.) with NULL
  s2  DTypeEnforcer         — infer and cast VARCHAR / BIGINT columns to tighter types
  s3  StringStandardizer    — NFKD Unicode normalisation + fingerprint columns
  s4  Deduplicator          — remove duplicate rows by schema-declared key
  s5  JoinBuilder           — assemble wide analytical view (1 row per tconst)
  s6  Normalizer            — log1p transform for heavy-tailed numerics
  s7  MICEImputer           — MICE imputation (fit on train, apply to all splits)
  s8  assert_quality        — quality gate assertions before export
      save_parquet          — materialise DuckDB view to Parquet
"""
from __future__ import annotations

import re
import unicodedata
import warnings
from pathlib import Path

import duckdb
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from .schema import get_id_cols, get_key, validate


# ══════════════════════════════════════════════════════════════════════════════
# s1 — Missing token replacement
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# s2 — DType enforcement
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# s3 — String standardisation
# ══════════════════════════════════════════════════════════════════════════════

_STRIP_PUNCT = re.compile(r"[^\w\s]")
_COLLAPSE_WS = re.compile(r"\s+")


def _normalize(s: str) -> str | None:
    """NFKD + ASCII + strip punctuation + collapse whitespace. (from ilesh/dq.py)"""
    if not isinstance(s, str):
        return s
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = _STRIP_PUNCT.sub(" ", s)
    return _COLLAPSE_WS.sub(" ", s).strip() or None


def _fingerprint(s: str) -> str:
    """Sort-token fingerprint key for near-duplicate detection. (from quality_report.py)"""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s).lower().encode("ascii", "ignore").decode("ascii")
    return " ".join(sorted(set(_STRIP_PUNCT.sub(" ", s).split())))


class StringStandardizer:
    """Apply NFKD normalisation to VARCHAR text columns.
    ID columns (from schema) are skipped to preserve join keys.
    Adds a __fp_{col} fingerprint column per normalised column for deduplication."""

    def transform(self, con: duckdb.DuckDBPyConnection, table: str) -> str:
        for name, fn in [("_normalize", _normalize), ("_fingerprint", _fingerprint)]:
            try:
                con.create_function(name, fn, [str], str, null_handling="special")
            except (duckdb.CatalogException, duckdb.NotImplementedException):
                pass  # already registered in this connection

        skip = set(get_id_cols(table))
        cols = con.execute(f"DESCRIBE {table}").fetchall()
        exprs = []
        for col, dtype, *_ in cols:
            q = f'"{col}"'
            if "VARCHAR" in dtype.upper() and col not in skip:
                exprs.append(f'_normalize({q}) AS {q}')
                exprs.append(f'_fingerprint({q}) AS "__fp_{col}"')
            else:
                exprs.append(q)

        out = f"{table}_std"
        con.execute(f"CREATE OR REPLACE VIEW {out} AS SELECT {', '.join(exprs)} FROM {table}")
        return out


# ══════════════════════════════════════════════════════════════════════════════
# s4 — Deduplication
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# s5 — Schema-specific joins
# ══════════════════════════════════════════════════════════════════════════════
#
# Assembles a wide analytical view (1 row per tconst) from individually cleaned
# tables so that downstream normalisation and imputation have access to all features.
#
# Relational schema (after steps 1–4 clean each table independently)
# ──────────────────────────────────────────────────────────────────
#     train_all            tconst PK    ← base table (all 8 train splits UNIONed)
#       │
#       ├── 1:1  title_basics     tconst PK    → genres, titleType, isAdult
#       ├── 1:1  title_crew       tconst PK    → directors (csv), writers (csv)
#       │
#       ├── M:1  movie_directors  (tconst, director)   → aggregated to 1 row per tconst
#       │          └── → name_basics  nconst PK         → director person metadata
#       │
#       └── M:1  movie_writers    (tconst, writer)      → aggregated to 1 row per tconst
#                  └── → name_basics  nconst PK         → writer person metadata
#
#     SKIPPED: title_principals (avg 20 rows/tconst → too wide, use in feature engineering)

# ── Editable: columns to SELECT from each 1:1 join table ─────────────────────
ONE_TO_ONE_JOINS = {
    "title_basics": {
        "cols": ["genres", "titleType", "isAdult"],
    },
    # title_crew intentionally omitted — raw director/writer ID strings
    # are superseded by the aggregated person metadata in MANY_TO_MANY_AGGS.
}

# ── Editable: aggregation specs for many-to-many edge tables ─────────────────
MANY_TO_MANY_AGGS = {
    "movie_directors": {
        "key":        "tconst",
        "person_col": "director",
        "agg_prefix": "dir",
    },
    "movie_writers": {
        "key":        "tconst",
        "person_col": "writer",
        "agg_prefix": "wri",
    },
}

PERSON_TABLE = "name_basics"
PERSON_KEY   = "nconst"

_JOIN_KEY_CANDIDATES = ("tconst", "nconst")


class JoinBuilder:
    """Build a single wide analytical VIEW (1 row per tconst) from cleaned tables.

    Expects all table names to be the *cleaned* view names (after steps 1–4).
    Pass them via the `cleaned` dict: {original_csv_stem: cleaned_view_name}.
    """

    def __init__(self, cleaned: dict[str, str]):
        self.cleaned = cleaned

    def _resolve(self, stem: str) -> str:
        if stem not in self.cleaned:
            raise KeyError(f"Table '{stem}' not found in cleaned views: {list(self.cleaned)}")
        return self.cleaned[stem]

    def _detect_join_key(
        self, con: duckdb.DuckDBPyConnection, base: str, tbl: str
    ) -> str | None:
        base_cols = {r[0] for r in con.execute(f"DESCRIBE {base}").fetchall()}
        tbl_cols  = {r[0] for r in con.execute(f"DESCRIBE {tbl}").fetchall()}
        for key in _JOIN_KEY_CANDIDATES:
            if key in base_cols and key in tbl_cols:
                return key
        return None

    def _create_m2m_agg(self, con: duckdb.DuckDBPyConnection, stem: str, spec: dict) -> str:
        """Aggregate a many-to-many edge table to 1 row per tconst with person metadata."""
        edge   = self._resolve(stem)
        person = self._resolve(PERSON_TABLE)
        key    = spec["key"]
        pcol   = spec["person_col"]
        p      = spec["agg_prefix"]

        view = f"_agg_{p}"
        con.execute(f"""
            CREATE OR REPLACE VIEW {view} AS
            SELECT
                e."{key}",
                COUNT(*)                           AS {p}_count,
                STRING_AGG(e."{pcol}", ',')        AS {p}_ids,
                AVG(p."birthYear")                 AS {p}_avg_birth_year,
                MIN(p."birthYear")                 AS {p}_min_birth_year,
                AVG(p."deathYear")                 AS {p}_avg_death_year,
                STRING_AGG(DISTINCT p."primaryProfession", ',')
                                                   AS {p}_professions
            FROM {edge} e
            LEFT JOIN {person} p ON e."{pcol}" = p."{PERSON_KEY}"
            GROUP BY e."{key}"
        """)
        return view

    def transform(self, con: duckdb.DuckDBPyConnection, base: str, out: str = "joined") -> str:
        """Create the wide joined VIEW. Returns its name."""
        select = ["b.*"]
        joins  = []

        for stem, spec in ONE_TO_ONE_JOINS.items():
            tbl = self._resolve(stem)
            key = self._detect_join_key(con, base, tbl)
            if key is None:
                warnings.warn(
                    f"Skipping 1:1 join for '{stem}': no shared key column "
                    f"({', '.join(_JOIN_KEY_CANDIDATES)}) found between '{base}' and '{tbl}'."
                )
                continue
            alias = f"o_{stem}"
            for col in spec["cols"]:
                select.append(f'{alias}."{col}"')
            joins.append(f'LEFT JOIN {tbl} AS {alias} ON b."{key}" = {alias}."{key}"')

        for stem, spec in MANY_TO_MANY_AGGS.items():
            agg_view = self._create_m2m_agg(con, stem, spec)
            key      = spec["key"]
            alias    = f"m_{spec['agg_prefix']}"
            p        = spec["agg_prefix"]
            agg_cols = [
                r[0] for r in con.execute(f"DESCRIBE {agg_view}").fetchall()
                if r[0] != key
            ]
            for col in agg_cols:
                if col in (f"{p}_count",):
                    select.append(f'COALESCE({alias}."{col}", 0) AS "{col}"')
                else:
                    select.append(f'{alias}."{col}"')
            joins.append(f'LEFT JOIN {agg_view} AS {alias} ON b."{key}" = {alias}."{key}"')

        sql = f"SELECT {', '.join(select)} FROM {base} AS b {' '.join(joins)}"
        con.execute(f"CREATE OR REPLACE VIEW {out} AS {sql}")
        return out

    def export(self, con: duckdb.DuckDBPyConnection, view: str, path: Path) -> Path:
        """Materialise the joined view to Parquet."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        con.execute(f"COPY (SELECT * FROM {view}) TO '{path}' (FORMAT PARQUET)")
        return path


# ══════════════════════════════════════════════════════════════════════════════
# s6 — Distribution normalisation (log1p)
# ══════════════════════════════════════════════════════════════════════════════
#
# Applied before MICE so that imputed values are on a more Gaussian-like scale.
# numVotes is extremely right-skewed (range 1k–2.5M), so log1p is standard.

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
                exprs.append(f'LN(1 + {q}) AS "{col}_log1p"')
            else:
                exprs.append(q)

        out = f"{table}_norm"
        con.execute(
            f"CREATE OR REPLACE VIEW {out} AS SELECT {', '.join(exprs)} FROM {table}"
        )
        return out


# ══════════════════════════════════════════════════════════════════════════════
# s7 — MICE imputation
# ══════════════════════════════════════════════════════════════════════════════
#
# Fit on the TRAIN joined table only, then transform val/test with the same
# fitted model — zero leakage of val/test distributions into imputed values.

# Numeric columns to impute. Only applied if the column exists in the table.
# numVotes_log1p: s6 renames numVotes → numVotes_log1p before imputation runs.
_IMPUTE_COLS: tuple[str, ...] = ("startYear", "runtimeMinutes", "numVotes_log1p")

# Columns that must be integers after imputation (MICE produces floats).
_ROUND_TO_INT: frozenset[str] = frozenset({"startYear", "runtimeMinutes"})

_KEY_COL = "tconst"


class MICEImputer:
    """Fit MICE on train, apply to any split — no leakage.

    Only numeric columns listed in _IMPUTE_COLS are imputed.
    All other columns pass through unchanged.
    """

    def __init__(self, max_iter: int = 10, random_state: int = 42) -> None:
        self._imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
        self._cols_to_impute: list[str] = []

    def _numeric_impute_cols(self, con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
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

        present = [c for c in self._cols_to_impute if c in {
            r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()
        }]
        if not present:
            return table

        sel = ", ".join(f'"{c}"' for c in [_KEY_COL] + present)
        df = con.execute(f"SELECT {sel} FROM {table}").fetchdf()

        imputed_arr = self._imputer.transform(df[present])
        df_imputed  = pd.DataFrame(imputed_arr, columns=present)
        df_imputed[_KEY_COL] = df[_KEY_COL].values

        for col in present:
            if col in _ROUND_TO_INT:
                df_imputed[col] = df_imputed[col].round().astype("Int64")

        tmp = f"_mice_imputed_{suffix or table.replace('-', '_')}"
        con.register(tmp, df_imputed)

        all_cols = [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]
        exprs = []
        for col in all_cols:
            if col in present:
                exprs.append(f'{tmp}."{col}" AS "{col}"')
            else:
                exprs.append(f'base."{col}"')

        out = f"{table}_imputed"
        con.execute(f"""
            CREATE OR REPLACE VIEW {out} AS
            SELECT {', '.join(exprs)}
            FROM {table} base
            JOIN {tmp} ON base."{_KEY_COL}" = {tmp}."{_KEY_COL}"
        """)
        print(f"[impute] MICE transformed {table} → {out}  cols={present}")
        return out


# ══════════════════════════════════════════════════════════════════════════════
# s8 — Quality gate + Parquet export
# ══════════════════════════════════════════════════════════════════════════════

# Columns that must be fully imputed before export.
_IMPUTED_COLS = ("startYear", "runtimeMinutes", "numVotes_log1p")


def assert_quality(con: duckdb.DuckDBPyConnection, table: str) -> None:
    """Raise ValueError if the cleaned table still has schema violations or remaining NULLs."""
    issues = validate(con, table)
    if issues:
        raise ValueError("Quality gate failed:\n" + "\n".join(f"  • {i}" for i in issues))

    existing = {r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()}
    for col in _IMPUTED_COLS:
        if col not in existing:
            continue
        n_null = con.execute(
            f'SELECT COUNT(*) FROM {table} WHERE "{col}" IS NULL'
        ).fetchone()[0]
        if n_null > 0:
            raise ValueError(
                f"Quality gate failed: {n_null} NULL values remain in {col} after imputation"
            )


def save_parquet(con: duckdb.DuckDBPyConnection, view: str, path: Path) -> Path:
    """Materialise a DuckDB view to Parquet, dropping internal __fp_* columns.

    __fp_* columns are fingerprint helpers added by s3 for deduplication.
    They are pipeline-internal and should not appear in output.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    _DROP = {"dir_ids", "wri_ids"}
    all_cols = [r[0] for r in con.execute(f"DESCRIBE {view}").fetchall()]
    keep = [f'"{c}"' for c in all_cols if not c.startswith("__fp_") and c not in _DROP]
    con.execute(f"COPY (SELECT {', '.join(keep)} FROM {view}) TO '{path}' (FORMAT PARQUET)")

    n = con.execute(f"SELECT COUNT(*) FROM '{path}'").fetchone()[0]
    dropped = len(all_cols) - len(keep)
    suffix = f"  (dropped {dropped} __fp_* cols)" if dropped else ""
    print(f"[save] {view:<45} → {path.name}  ({n:,} rows){suffix}")
    return path
