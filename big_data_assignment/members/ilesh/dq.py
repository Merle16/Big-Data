"""
members/ilesh/dq.py
===================
Step 1 — DuckDB: Data Quality Detection & Fixing.

WHY DuckDB HERE?
  - In-process SQL, zero cluster overhead for 8K rows.
  - TRY_CAST, QUANTILE_CONT, COPY TO PARQUET, and UDF registration are
    single SQL calls — each DQ check = one SQL string, trivial to update
    if the schema changes.
  - all_varchar=True preserves raw \\N tokens so every check runs on the
    original strings before any coercion.

Can be imported and called independently:
    from members.ilesh.dq import step1_duckdb
    step1_duckdb()
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

# ── Title normalisation (registered as a DuckDB UDF in step1_duckdb) ─────────
_STRIP_PUNCT = re.compile(r"[^\w\s]")
_COLLAPSE_WS = re.compile(r"\s+")


def normalize_title(s: str) -> str:
    """NFKD + ASCII: strips synthetically-injected diacritics (Déstiny→Destiny).
    Safe for legitimate foreign titles (Le Samouraï → Le Samourai)."""
    if not s or s != s:   # empty or NaN
        return s
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = _STRIP_PUNCT.sub(" ", s)
    s = _COLLAPSE_WS.sub(" ", s).strip()
    return s


# ── Report helpers ────────────────────────────────────────────────────────────
_log: list[str] = []


def rpt(msg: str = "") -> None:
    print(msg)
    _log.append(str(msg))


def _save_report(path: Path) -> None:
    path.write_text("\n".join(_log))
    print(f"\n[DQ report → {path}]")


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def step1_duckdb(
    raw_csv: Path,
    pqdir: Path,
    db_path: Path,
    rpt_path: Path,
) -> None:
    """
    Run all 13 DQ checks (D1–D13) and apply all 9 fixes (F1–F9).
    Exports 7 Parquet files to *pqdir* for PySpark Step 2.

    Parameters
    ----------
    raw_csv  : directory containing train-*.csv, validation_hidden.csv,
               test_hidden.csv, movie_directors.csv, movie_writers.csv
    pqdir    : output directory for Parquet files
    db_path  : path for the persistent DuckDB database file
    rpt_path : path where the text DQ report is written
    """
    import duckdb
    import pandas as pd

    pqdir.mkdir(parents=True, exist_ok=True)
    _log.clear()

    DIRECTORS_CSV = str(raw_csv / "movie_directors.csv")
    WRITERS_CSV   = str(raw_csv / "movie_writers.csv")
    TRAIN_GLOB    = str(raw_csv / "train-*.csv")
    VAL_CSV       = str(raw_csv / "validation_hidden.csv")
    TEST_CSV      = str(raw_csv / "test_hidden.csv")

    rpt("=" * 68)
    rpt("STEP 1 — DuckDB: Data Quality Detection & Fixing")
    rpt("=" * 68)

    con = duckdb.connect(str(db_path))
    con.create_function("normalize_title", normalize_title, [str], str)

    # ── A) INGESTION ──────────────────────────────────────────────────────────
    # all_varchar=True: preserves \N tokens as literal strings (not silently NaN)
    # F1: column0 excluded — spurious pandas index column
    rpt("\n>> A) Ingesting raw data...")
    con.execute("""
        CREATE OR REPLACE TABLE movies_train_raw AS
        SELECT * EXCLUDE (column0)
        FROM read_csv_auto(?, header=True, all_varchar=True, ignore_errors=True)
    """, [TRAIN_GLOB])

    for tbl, path in [("movies_val_raw", VAL_CSV), ("movies_test_raw", TEST_CSV)]:
        con.execute(f"""
            CREATE OR REPLACE TABLE {tbl} AS
            SELECT * EXCLUDE (column0)
            FROM read_csv_auto(?, header=True, all_varchar=True, ignore_errors=True)
        """, [path])

    # F6: drop invalid person_id edges at ingestion
    # \N  (single-escaped) — 297 writer rows, standard IMDb unknown token
    # \\N (double-escaped) —   2 director rows, corrupted variant
    con.execute(f"""
        CREATE OR REPLACE TABLE directing AS
        SELECT tconst, director AS director_id
        FROM read_csv_auto('{DIRECTORS_CSV}', header=True)
        WHERE director IS NOT NULL
          AND TRIM(director) NOT IN ('\\N', '\\\\N')
          AND TRIM(director) != ''
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE writing AS
        SELECT tconst, writer AS writer_id
        FROM read_csv_auto('{WRITERS_CSV}', header=True)
        WHERE writer IS NOT NULL
          AND TRIM(writer) NOT IN ('\\N', '\\\\N')
          AND TRIM(writer) != ''
    """)

    n_train = con.execute("SELECT COUNT(*) FROM movies_train_raw").fetchone()[0]
    rpt(f"  movies_train_raw : {n_train:,} rows")
    rpt(f"  directing        : {con.execute('SELECT COUNT(*) FROM directing').fetchone()[0]:,} edges")
    rpt(f"  writing          : {con.execute('SELECT COUNT(*) FROM writing').fetchone()[0]:,} edges")

    # ── B) DQ DETECTION (D1–D13) ─────────────────────────────────────────────
    rpt("\n>> B) Automatic Data Quality Detection ...")

    # D1: Schema validation
    rpt("\n[D1] Schema validation:")
    actual_cols   = set(con.execute("DESCRIBE movies_train_raw").fetchdf()["column_name"].tolist())
    expected_cols = {"tconst", "primaryTitle", "originalTitle", "startYear",
                     "endYear", "runtimeMinutes", "numVotes", "label"}
    missing_cols  = expected_cols - actual_cols
    extra_cols    = actual_cols - expected_cols
    rpt(f"  Expected columns present : {'✓' if not missing_cols else '✗ MISSING: ' + str(missing_cols)}")
    rpt(f"  Unexpected extra columns : {'none' if not extra_cols else str(extra_cols)}")

    # D2: \N token counts (incl. numVotes — silently cast to NULL by TRY_CAST)
    rpt("\n[D2] Disguised \\N tokens per column:")
    for col in ["startYear", "endYear", "runtimeMinutes", "numVotes", "originalTitle", "primaryTitle"]:
        n = con.execute(
            f"SELECT COUNT(*) FROM movies_train_raw WHERE TRIM({col}) = '\\N'"
        ).fetchone()[0]
        rpt(f"  {col:<20}: {n:>5} ({100*n/n_train:.1f}%)")

    # D3: True NULLs
    rpt("\n[D3] True NULL counts:")
    null_df = con.execute("""
        SELECT SUM(primaryTitle IS NULL) AS null_primaryTitle,
               SUM(originalTitle IS NULL) AS null_originalTitle,
               SUM(startYear IS NULL) AS null_startYear,
               SUM(endYear IS NULL) AS null_endYear,
               SUM(runtimeMinutes IS NULL) AS null_runtimeMinutes,
               SUM(numVotes IS NULL) AS null_numVotes
        FROM movies_train_raw
    """).fetchdf()
    for col in null_df.columns:
        n = int(null_df[col].values[0])
        if n > 0:
            rpt(f"  {col:<30}: {n}")

    # D4: tconst format
    n_bad_tconst = con.execute(
        "SELECT COUNT(*) FROM movies_train_raw WHERE NOT regexp_matches(tconst, '^tt[0-9]+$')"
    ).fetchone()[0]
    rpt(f"\n[D4] Malformed tconst (not tt[0-9]+): {n_bad_tconst} ({'OK' if n_bad_tconst == 0 else '← FOUND'})")

    # D5: Cross-split leakage
    rpt("\n[D5] Cross-split leakage (shared tconst across splits):")
    for pair, q in [
        ("train ∩ val ", "SELECT COUNT(*) FROM movies_train_raw t JOIN movies_val_raw  v ON t.tconst=v.tconst"),
        ("train ∩ test", "SELECT COUNT(*) FROM movies_train_raw t JOIN movies_test_raw x ON t.tconst=x.tconst"),
        ("val ∩ test  ", "SELECT COUNT(*) FROM movies_val_raw   v JOIN movies_test_raw x ON v.tconst=x.tconst"),
    ]:
        n = con.execute(q).fetchone()[0]
        rpt(f"  {pair}: {n} {'(OK)' if n == 0 else '← LEAKAGE'}")

    # D6: Duplicate tconst
    n_dup = con.execute("SELECT COUNT(*)-COUNT(DISTINCT tconst) FROM movies_train_raw").fetchone()[0]
    rpt(f"\n[D6] Duplicate tconst rows: {n_dup} ({'none' if n_dup==0 else 'FOUND'})")

    # D7: numVotes IQR outlier fence
    rpt("\n[D7] numVotes IQR outlier analysis:")
    qr = con.execute("""
        SELECT QUANTILE_CONT(TRY_CAST(numVotes AS DOUBLE), 0.25) AS q1,
               QUANTILE_CONT(TRY_CAST(numVotes AS DOUBLE), 0.50) AS med,
               QUANTILE_CONT(TRY_CAST(numVotes AS DOUBLE), 0.75) AS q3
        FROM movies_train_raw WHERE numVotes IS NOT NULL
    """).fetchdf()
    q1 = float(qr["q1"].values[0])
    q3 = float(qr["q3"].values[0])
    VOTES_FENCE = round(q3 + 1.5 * (q3 - q1), 2)
    n_above = con.execute(
        f"SELECT COUNT(*) FROM movies_train_raw "
        f"WHERE TRY_CAST(numVotes AS DOUBLE) > {VOTES_FENCE}"
    ).fetchone()[0]
    rpt(f"  Q1={q1:.0f}  Q3={q3:.0f}  fence={VOTES_FENCE:.0f}  rows above={n_above} ({100*n_above/n_train:.1f}%)")

    # D8: Referential integrity
    rpt("\n[D8] Referential integrity (edges vs train+val+test combined):")
    for etbl in ["directing", "writing"]:
        n = con.execute(f"""
            SELECT COUNT(DISTINCT e.tconst) FROM {etbl} e
            LEFT JOIN movies_train_raw t ON e.tconst=t.tconst
            LEFT JOIN movies_val_raw   v ON e.tconst=v.tconst
            LEFT JOIN movies_test_raw  x ON e.tconst=x.tconst
            WHERE t.tconst IS NULL AND v.tconst IS NULL AND x.tconst IS NULL
        """).fetchone()[0]
        rpt(f"  {etbl}: {n} edge tconst not in any split {'(OK)' if n==0 else '← ISSUE'}")

    # D9: Person ID format
    rpt("\n[D9] Person ID format check (pre-filter, raw CSVs):")
    con.execute(f"CREATE OR REPLACE TEMP TABLE dir_raw AS SELECT * FROM read_csv_auto('{DIRECTORS_CSV}', header=True, all_varchar=True)")
    con.execute(f"CREATE OR REPLACE TEMP TABLE wri_raw AS SELECT * FROM read_csv_auto('{WRITERS_CSV}',   header=True, all_varchar=True)")
    for tbl, col in [("dir_raw", "director"), ("wri_raw", "writer")]:
        bad = con.execute(f"""
            SELECT TRIM({col}) AS bad_id, COUNT(*) AS n
            FROM {tbl}
            WHERE {col} IS NOT NULL AND NOT regexp_matches(TRIM({col}), '^nm[0-9]+$')
            GROUP BY bad_id ORDER BY n DESC
        """).fetchdf()
        if bad.empty:
            rpt(f"  {tbl}: all IDs match nm[0-9]+ (OK)")
        else:
            rpt(f"  {tbl}: {len(bad)} non-conforming ID pattern(s):")
            rpt("  " + bad.to_string(index=False))
    rpt("  → F6 removes all non-conforming IDs at ingestion")

    # D10: Label balance
    rpt("\n[D10] Label balance:")
    lb = con.execute("""
        SELECT label, COUNT(*) AS n, ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER(),2) AS pct
        FROM movies_train_raw GROUP BY label ORDER BY label
    """).fetchdf()
    rpt(lb.to_string(index=False))

    # D11: Missingness mechanism
    rpt("\n[D11] Missingness mechanism — label rate: missing vs present:")
    for col, token in [("startYear", "\\N"), ("runtimeMinutes", "\\N"), ("numVotes", "\\N")]:
        df = con.execute(f"""
            SELECT (TRIM({col})='{token}') AS is_missing, COUNT(*) AS n,
                   ROUND(AVG(CASE WHEN label='True' THEN 1.0 ELSE 0.0 END),4) AS avg_label
            FROM movies_train_raw GROUP BY is_missing ORDER BY is_missing
        """).fetchdf()
        rpt(f"  {col}:")
        rpt("  " + df.to_string(index=False))
    rpt("  (small difference → keep missingness flags as features)")

    # D12: Range validity
    rpt("\n[D12] Out-of-range value checks (train):")
    bad_year = con.execute("""
        SELECT COUNT(*) FROM movies_train_raw
        WHERE TRY_CAST(NULLIF(TRIM(startYear), '\\N') AS INTEGER) NOT BETWEEN 1880 AND 2030
          AND TRIM(startYear) != '\\N' AND startYear IS NOT NULL
    """).fetchone()[0]
    bad_rt = con.execute("""
        SELECT COUNT(*) FROM movies_train_raw
        WHERE TRY_CAST(NULLIF(TRIM(runtimeMinutes), '\\N') AS INTEGER) <= 0
          AND TRIM(runtimeMinutes) != '\\N' AND runtimeMinutes IS NOT NULL
    """).fetchone()[0]
    bad_votes = con.execute("""
        SELECT COUNT(*) FROM movies_train_raw
        WHERE TRY_CAST(numVotes AS DOUBLE) <= 0
          AND numVotes IS NOT NULL AND TRIM(numVotes) != '\\N'
    """).fetchone()[0]
    rpt(f"  startYear outside [1880, 2030] : {bad_year} {'(OK)' if bad_year == 0 else '← FOUND'}")
    rpt(f"  runtimeMinutes <= 0            : {bad_rt}   {'(OK)' if bad_rt == 0 else '← FOUND'}")
    rpt(f"  numVotes <= 0                  : {bad_votes} {'(OK)' if bad_votes == 0 else '← FOUND'}")

    # D13: Text corruption (accent injection) — both title columns
    rpt("\n[D13] Text corruption — non-ASCII titles:")
    primary_df  = con.execute("SELECT primaryTitle FROM movies_train_raw").fetchdf()
    original_df = con.execute(
        "SELECT originalTitle FROM movies_train_raw WHERE originalTitle IS NOT NULL"
    ).fetchdf()
    n_primary  = primary_df["primaryTitle"].apply(
        lambda x: normalize_title(str(x)) != str(x)
    ).sum()
    n_original = original_df["originalTitle"].apply(
        lambda x: normalize_title(str(x)) != str(x)
    ).sum()
    rpt(f"  primaryTitle  rows with synthetic accents: {n_primary}  ({100*n_primary/n_train:.1f}%)")
    rpt(f"  originalTitle rows with synthetic accents: {n_original} ({100*n_original/len(original_df):.1f}% of non-null rows)")

    # ── C) FIXES ──────────────────────────────────────────────────────────────
    rpt("\n>> C) Applying fixes ...")
    rpt("[F1] column0 excluded at ingestion ✓")
    rpt("[F6] \\N + \\\\N person_ids dropped at ingestion (\\N=297 writers, \\\\N=2 directors) ✓")
    rpt("[F8] numVotes \\N token → NULL via TRY_CAST (implicit; consistent with F3) ✓")

    def _build_clean(out_tbl: str, src: str, has_label: bool) -> None:
        """One-pass clean applying F2, F3, F4, F8, F9."""
        label_col = ", CAST(label AS BOOLEAN) AS label" if has_label else ""
        con.execute(f"""
            CREATE OR REPLACE TABLE {out_tbl} AS
            SELECT
                tconst,
                normalize_title(primaryTitle) AS primaryTitle,
                COALESCE(
                    NULLIF(normalize_title(TRIM(COALESCE(originalTitle,''))), ''),
                    normalize_title(primaryTitle)
                ) AS originalTitle,
                TRY_CAST(
                    NULLIF(NULLIF(TRIM(startYear),''), '\\N')
                AS INTEGER) AS startYear,
                -- F2: endYear DROPPED — 90.1% missing, near-zero signal
                TRY_CAST(
                    NULLIF(NULLIF(TRIM(runtimeMinutes),''), '\\N')
                AS INTEGER) AS runtimeMinutes,
                TRY_CAST(numVotes AS DOUBLE) AS numVotes
                {label_col}
            FROM ({src})
        """)

    _build_clean("movies_train_clean", "SELECT * FROM movies_train_raw", has_label=True)
    _build_clean("movies_val_clean",   "SELECT * FROM movies_val_raw",   has_label=False)
    _build_clean("movies_test_clean",  "SELECT * FROM movies_test_raw",  has_label=False)

    # F5: numVotes IQR cap (train fence applied to all splits — no leakage)
    for tbl in ["movies_train_clean", "movies_val_clean", "movies_test_clean"]:
        con.execute(f"""
            CREATE OR REPLACE TABLE {tbl} AS
            SELECT * REPLACE (
                CASE WHEN numVotes > {VOTES_FENCE} THEN {VOTES_FENCE}
                     ELSE numVotes END AS numVotes
            ) FROM {tbl}
        """)
    rpt(f"[F5] numVotes capped at IQR fence={VOTES_FENCE:.0f} ({n_above} train rows capped)")

    # F7: Director/writer success-rate encoding (INNER JOIN train only → no leakage)
    rpt("[F7] Director/writer success-rate encoding (train labels only)...")
    con.execute("""
        CREATE OR REPLACE TABLE director_success AS
        SELECT d.director_id,
               COUNT(DISTINCT d.tconst)     AS director_movie_count,
               AVG(CAST(m.label AS DOUBLE)) AS director_success_rate
        FROM directing d
        INNER JOIN movies_train_clean m ON d.tconst = m.tconst
        GROUP BY d.director_id
    """)
    con.execute("""
        CREATE OR REPLACE TABLE writer_success AS
        SELECT w.writer_id,
               COUNT(DISTINCT w.tconst)     AS writer_movie_count,
               AVG(CAST(m.label AS DOUBLE)) AS writer_success_rate
        FROM writing w
        INNER JOIN movies_train_clean m ON w.tconst = m.tconst
        GROUP BY w.writer_id
    """)
    rpt(f"  {con.execute('SELECT COUNT(*) FROM director_success').fetchone()[0]:,} directors encoded")
    rpt(f"  {con.execute('SELECT COUNT(*) FROM writer_success').fetchone()[0]:,} writers encoded")

    # ── D) POST-FIX VERIFICATION ──────────────────────────────────────────────
    rpt("\n>> D) Post-fix verification:")
    v = con.execute("""
        SELECT COUNT(*) AS total,
               SUM(startYear IS NULL) AS null_startYear,
               SUM(runtimeMinutes IS NULL) AS null_runtimeMinutes,
               SUM(numVotes IS NULL) AS null_numVotes,
               SUM(primaryTitle IS NULL) AS null_primaryTitle,
               MAX(numVotes) AS max_numVotes_capped
        FROM movies_train_clean
    """).fetchdf()
    rpt(v.to_string(index=False))

    # ── E) EXPORT TO PARQUET ──────────────────────────────────────────────────
    rpt("\n>> E) Exporting Parquet for PySpark...")
    for tbl, fname in [
        ("movies_train_clean",  "movies_train.parquet"),
        ("movies_val_clean",    "movies_val.parquet"),
        ("movies_test_clean",   "movies_test.parquet"),
        ("directing",           "directing.parquet"),
        ("writing",             "writing.parquet"),
        ("director_success",    "director_success.parquet"),
        ("writer_success",      "writer_success.parquet"),
    ]:
        path = pqdir / fname
        con.execute(f"COPY {tbl} TO '{path}' (FORMAT PARQUET)")
        n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        rpt(f"  [OK] {tbl:<25}: {n:,} rows → {fname}")

    con.close()
    _save_report(rpt_path)
    print(f"\n[STEP 1 DONE] → {pqdir}")
