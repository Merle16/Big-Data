"""
Single source of truth for the relational schema used by the cleaning pipeline.

Declares tables, keys, ID patterns, and drop columns. Uses DuckDB SQL to
validate UUID formats and key uniqueness at runtime.

Every pipeline step reads from here instead of auto-detecting.
"""

import duckdb

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
    for prefix, spec in SCHEMA.items():
        if table == prefix or table.startswith(prefix):
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


# ── SQL-based schema application ─────────────────────────────────────────────

def apply(con: duckdb.DuckDBPyConnection, table: str) -> str:
    """Create a DuckDB VIEW that enforces the declared schema for *table*.

    Uses SQL DESCRIBE to introspect current columns, then builds a SELECT that
    omits any ``drop_cols`` declared in SCHEMA.  Returns the new view name, or
    the original table name when nothing needs to be dropped.
    """
    spec = _match(table)
    if spec is None:
        return table

    drop = set(spec.get("drop_cols", ()))
    if not drop:
        return table

    keep = [
        f'"{r[0]}"'
        for r in con.execute(f"DESCRIBE {table}").fetchall()
        if r[0] not in drop
    ]
    if not keep:
        return table

    out = f"{table}_schema"
    con.execute(
        f"CREATE OR REPLACE VIEW {out} AS SELECT {', '.join(keep)} FROM {table}"
    )
    return out


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
