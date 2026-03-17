"""
Single source of truth for the relational schema used by the cleaning pipeline.

Declares tables, keys, ID patterns, and drop columns. Uses DuckDB SQL to
validate UUID formats and key uniqueness at runtime.

Every pipeline step reads from here instead of auto-detecting.
"""

import logging
import re

import duckdb
import pandas as pd

log = logging.getLogger("pipeline.schema")

# ── Schema version ────────────────────────────────────────────────────────────
SCHEMA_VERSION = "1.0.0"

# ── Data contracts ────────────────────────────────────────────────────────────
# Maps column names to type/null/range rules used by validate_contracts().
COLUMN_CONTRACTS: dict[str, dict] = {
    "tconst":         {"type": str,   "nullable": False, "pattern": r"^tt\d+$"},
    "startYear":      {"type": float, "nullable": True,  "min": 1880, "max": 2030},
    "runtimeMinutes": {"type": float, "nullable": True,  "min": 1,    "max": 600},
    "numVotes":       {"type": float, "nullable": True,  "min": 0},
    "averageRating":  {"type": float, "nullable": True,  "min": 0.0,  "max": 10.0},
    "label":          {"type": int,   "nullable": False, "values": [0, 1]},
    "isAdult":        {"type": int,   "nullable": True,  "values": [0, 1]},
}

# ── SCHEMA ────────────────────────────────────────────────────────────────────
#
#   key       : primary or composite key columns used for deduplication
#   id_cols   : identifier columns to preserve as-is (skip normalization)
#   id_regex  : DuckDB regex to validate the UUID format per id column
#   drop_cols : columns to drop early (too sparse or not useful)
#
SCHEMA: dict[str, dict] = {
    # ── Train / validation / test splits (data/raw/csv/) ─────────────────────
    "train": {
        "key":       ("tconst",),
        "id_cols":   ("tconst",),
        "id_regex":  {"tconst": r"^tt\d+$"},
        "drop_cols": ("endYear", "column0"),
    },
    "test_hidden": {
        "key":       ("tconst",),
        "id_cols":   ("tconst",),
        "id_regex":  {"tconst": r"^tt\d+$"},
        "drop_cols": ("endYear", "column0"),
    },
    "validation_hidden": {
        "key":       ("tconst",),
        "id_cols":   ("tconst",),
        "id_regex":  {"tconst": r"^tt\d+$"},
        "drop_cols": ("endYear", "column0"),
    },

    # ── IMDB reference tables (data/raw/IMDB_external_csv/) ──────────────────
    "title_basics": {
        "key":       ("tconst",),
        "id_cols":   ("tconst",),
        "id_regex":  {"tconst": r"^tt\d+$"},
    },
    "title_crew": {
        "key":       ("tconst",),
        "id_cols":   ("tconst",),
        "id_regex":  {"tconst": r"^tt\d+$"},
    },
    "title_principals": {
        "key":       ("tconst", "ordering", "nconst"),
        "id_cols":   ("tconst", "nconst"),
        "id_regex":  {"tconst": r"^tt\d+$", "nconst": r"^nm\d+$"},
    },
    "name_basics": {
        "key":       ("nconst",),
        "id_cols":   ("nconst",),
        "id_regex":  {"nconst": r"^nm\d+$"},
    },

    # ── Many-to-many edge tables (data/raw/csv/) ─────────────────────────────
    "movie_directors": {
        "key":       ("tconst", "director"),
        "id_cols":   ("tconst", "director"),
        "id_regex":  {"tconst": r"^tt\d+$", "director": r"^nm\d+$"},
    },
    "movie_writers": {
        "key":       ("tconst", "writer"),
        "id_cols":   ("tconst", "writer"),
        "id_regex":  {"tconst": r"^tt\d+$", "writer": r"^nm\d+$"},
    },
}


# ── Lookup helpers ───────────────────────────────────────────────────────────

def _match(table: str) -> dict | None:
    """Return the SCHEMA spec for *table*, or None if not found.

    Matching rule: exact match OR the table name starts with ``<prefix>_``
    (the pipeline appends suffixes like ``_no_missing_typed_std_dedup``).
    Requiring the ``_`` separator prevents ``"train_extended"`` from
    accidentally matching the ``"train"`` spec.
    """
    for prefix, spec in SCHEMA.items():
        if table == prefix or table.startswith(prefix + "_"):
            return spec
    return None


def get_key(table: str) -> tuple[str, ...] | None:
    spec = _match(table)
    return spec["key"] if spec else None


def get_id_cols(table: str) -> tuple[str, ...]:
    spec = _match(table)
    return spec.get("id_cols", ()) if spec else ()


def get_drop_cols(table: str) -> tuple[str, ...]:
    spec = _match(table)
    return spec.get("drop_cols", ()) if spec else ()


# ── SQL-based schema validation ──────────────────────────────────────────────

def validate(con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    """Run DuckDB SQL checks against a table. Returns a list of error messages (empty = OK).

    Checks performed:
      1. UUID format — each id_col matches its declared regex
      2. Key uniqueness — the declared key has no duplicates
    """
    spec = _match(table)
    if spec is None:
        return [f"Table '{table}' not found in SCHEMA"]

    errors = []
    existing = {r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()}

    # 1. UUID format validation
    for col, regex in spec.get("id_regex", {}).items():
        if col not in existing:
            continue
        bad = con.execute(f"""
            SELECT COUNT(*) FROM {table}
            WHERE "{col}" IS NOT NULL
              AND NOT regexp_matches(CAST("{col}" AS VARCHAR), '{regex}')
        """).fetchone()[0]
        if bad > 0:
            errors.append(f"{table}.{col}: {bad} rows fail regex {regex}")

    # 2. Key uniqueness
    key = spec.get("key", ())
    if key and all(k in existing for k in key):
        key_expr = ", ".join(f'"{k}"' for k in key)
        dupes = con.execute(f"""
            SELECT COUNT(*) FROM (
                SELECT {key_expr}, COUNT(*) AS n
                FROM {table}
                GROUP BY {key_expr}
                HAVING n > 1
            )
        """).fetchone()[0]
        if dupes > 0:
            errors.append(f"{table}: {dupes} duplicate key groups on ({', '.join(key)})")

    return errors


# ── DataFrame contract validation ─────────────────────────────────────────────

def validate_contracts(df: pd.DataFrame, split: str = "unknown") -> list[str]:
    """Validate a DataFrame against COLUMN_CONTRACTS.

    Checks each column declared in COLUMN_CONTRACTS that exists in *df*:
      - nullable: non-nullable columns must have zero nulls
      - type: numeric contracts require the column to hold numeric data
      - min/max: values must fall within declared range (NaN excluded)
      - values: values must be members of the allowed set (NaN excluded)
      - pattern: string columns must match the regex (non-null values only)

    Returns a list of warning strings (does NOT raise). Each warning is also
    emitted via the module logger at WARNING level.
    """
    warnings: list[str] = []

    for col, contract in COLUMN_CONTRACTS.items():
        if col not in df.columns:
            continue

        series = df[col]

        # ── nullable check ────────────────────────────────────────────────────
        null_count = int(series.isna().sum())
        if not contract.get("nullable", True) and null_count > 0:
            msg = (
                f"[{split}] contract violation — '{col}': "
                f"{null_count} null values but column is not nullable"
            )
            warnings.append(msg)
            log.warning(msg)

        # ── numeric type check ────────────────────────────────────────────────
        expected_type = contract.get("type")
        if expected_type in (int, float):
            try:
                pd.to_numeric(series.dropna(), errors="raise")
            except (ValueError, TypeError):
                msg = (
                    f"[{split}] contract violation — '{col}': "
                    f"expected numeric type {expected_type.__name__}, "
                    f"found dtype {series.dtype}"
                )
                warnings.append(msg)
                log.warning(msg)

        non_null = series.dropna()

        # ── allowed values check ──────────────────────────────────────────────
        if "values" in contract and len(non_null) > 0:
            allowed = set(contract["values"])
            try:
                bad_mask = ~non_null.isin(allowed)
                bad_count = int(bad_mask.sum())
            except TypeError:
                bad_count = 0
            if bad_count > 0:
                msg = (
                    f"[{split}] contract violation — '{col}': "
                    f"{bad_count} values not in allowed set {contract['values']}"
                )
                warnings.append(msg)
                log.warning(msg)

        # ── range checks ──────────────────────────────────────────────────────
        if ("min" in contract or "max" in contract) and len(non_null) > 0:
            try:
                numeric_vals = pd.to_numeric(non_null, errors="coerce").dropna()
                if "min" in contract:
                    below = int((numeric_vals < contract["min"]).sum())
                    if below > 0:
                        msg = (
                            f"[{split}] contract violation — '{col}': "
                            f"{below} values below minimum {contract['min']}"
                        )
                        warnings.append(msg)
                        log.warning(msg)
                if "max" in contract:
                    above = int((numeric_vals > contract["max"]).sum())
                    if above > 0:
                        msg = (
                            f"[{split}] contract violation — '{col}': "
                            f"{above} values above maximum {contract['max']}"
                        )
                        warnings.append(msg)
                        log.warning(msg)
            except (TypeError, ValueError):
                pass

        # ── regex pattern check ───────────────────────────────────────────────
        if "pattern" in contract and len(non_null) > 0:
            pattern = contract["pattern"]
            try:
                bad_count = int(
                    (~non_null.astype(str).str.match(pattern, na=False)).sum()
                )
                if bad_count > 0:
                    msg = (
                        f"[{split}] contract violation — '{col}': "
                        f"{bad_count} values do not match pattern '{pattern}'"
                    )
                    warnings.append(msg)
                    log.warning(msg)
            except (TypeError, re.error):
                pass

    return warnings
