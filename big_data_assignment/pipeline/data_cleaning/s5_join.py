"""
Step 4½: Schema-specific joins.

Assembles a wide analytical view (1 row per tconst) from individually cleaned
tables so that downstream normalization and imputation have access to all features.

Relational schema (after steps 1–4 clean each table independently)
──────────────────────────────────────────────────────────────────
    train_all            tconst PK    ← base table (all 8 train splits UNIONed)
      │
      ├── 1:1  title_basics     tconst PK    → genres, titleType, isAdult
      ├── 1:1  title_crew       tconst PK    → directors (csv), writers (csv)
      │
      ├── M:1  movie_directors  (tconst, director)   → aggregated to 1 row per tconst
      │          └── → name_basics  nconst PK         → director person metadata
      │
      └── M:1  movie_writers    (tconst, writer)      → aggregated to 1 row per tconst
                 └── → name_basics  nconst PK         → writer person metadata

    SKIPPED:  title_principals (avg 20 rows/tconst → too wide, use in feature engineering)
"""

import warnings

import duckdb
from pathlib import Path


# ── Editable: columns to SELECT from each 1:1 join table ─────────────────────
# Only NEW columns not already in train/val/test.  Edit to add/remove.
#
# train already has: tconst, primaryTitle, originalTitle, startYear, endYear,
#                    runtimeMinutes, numVotes, label (train only)
# title_basics also has primaryTitle, originalTitle, startYear, endYear,
#   runtimeMinutes — do NOT include those (duplicates).
#   Only bring: genres, titleType, isAdult  ← genuinely new columns.
#
# title_crew.directors / title_crew.writers are raw comma-separated nconst
#   strings (e.g. "nm0001234,nm5678").  They are NOT included here because:
#   a) they aren't useful as ML features in raw form, and
#   b) richer aggregated person metadata comes from the M:M aggs below.
#
# No 'key' needed — auto-detected from shared ID columns via DuckDB DESCRIBE.
# Tables whose key doesn't exist in the base are skipped with a warning.
ONE_TO_ONE_JOINS = {
    "title_basics": {
        "cols": ["genres", "titleType", "isAdult"],
    },
    # title_crew intentionally omitted — raw director/writer ID strings
    # are superseded by the aggregated person metadata in MANY_TO_MANY_AGGS.
}

# ── Editable: aggregation specs for many-to-many edge tables ─────────────────
# Each entry produces 1 row per tconst via GROUP BY before joining.
MANY_TO_MANY_AGGS = {
    "movie_directors": {
        "key":          "tconst",
        "person_col":   "director",     # FK → name_basics.nconst
        "agg_prefix":   "dir",          # output columns: dir_count, dir_avg_birth_year, …
    },
    "movie_writers": {
        "key":          "tconst",
        "person_col":   "writer",       # FK → name_basics.nconst
        "agg_prefix":   "wri",          # output columns: wri_count, wri_avg_birth_year, …
    },
}

# ── Editable: person metadata table ──────────────────────────────────────────
PERSON_TABLE = "name_basics"
PERSON_KEY   = "nconst"

# ID columns that can serve as join keys, checked in priority order.
# Tables keyed only on nconst (e.g. name_basics) cannot join directly to a
# tconst-only base and will be skipped automatically.
_JOIN_KEY_CANDIDATES = ("tconst", "nconst")
# ─────────────────────────────────────────────────────────────────────────────


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
        """Return the first candidate key column present in both base and tbl.

        Uses DuckDB DESCRIBE on each table so that key detection is driven by
        the actual runtime schema rather than hard-coded config.  Returns None
        when no shared candidate exists (e.g. a nconst-only table joined to a
        tconst-only base).
        """
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
                    f"({', '.join(_JOIN_KEY_CANDIDATES)}) found between "
                    f"'{base}' and '{tbl}'."
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
                # COALESCE counts to 0 — a movie with no directors/writers
                # gets NULL from the LEFT JOIN, but 0 is the correct value.
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
