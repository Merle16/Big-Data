#!/usr/bin/env python3
"""
members/ilesh/pipeline.py
=========================
End-to-end IMDB pipeline (Docker-compatible standalone script).

Mirrors data_pipeline.ipynb exactly so the full pipeline can run without Jupyter.

Run locally (from big_data_assignment/):
    python members/ilesh/pipeline.py

Run in Docker:
    docker build -t big-data-imdb .
    docker run -it --rm \\
      -v $(pwd)/data:/app/data \\
      -v $(pwd)/members:/app/members \\
      -v $(pwd)/submissions:/app/submissions \\
      big-data-imdb \\
      python members/ilesh/pipeline.py

Tool split rationale
────────────────────
DuckDB  — ingestion, 13 DQ checks, 9 fixes, IQR/success-rate computation,
           Parquet export.  In-process SQL; no cluster needed for 8 K rows.
           each check = one SQL string → easy to audit/update if schema changes.
PySpark — Parquet load, MLlib Imputer (fit train / transform val+test),
           VectorAssembler + StandardScaler pipeline, Parquet write.
           Serialisable pipeline object = leakage-free distributed pattern.
sklearn — LogisticRegression on the scaled feature matrix.
"""

from __future__ import annotations

import datetime
import re
import sys
import unicodedata
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. PATHS  (works locally, in Docker at /app, and from any cwd)
# ─────────────────────────────────────────────────────────────────────────────
_here = Path(__file__).resolve()
# step up: ilesh/ → members/ → big_data_assignment/
PROJECT_ROOT = _here.parent.parent.parent
while (PROJECT_ROOT.name != "big_data_assignment"
       and not (PROJECT_ROOT / "config" / "config.yaml").exists()
       and PROJECT_ROOT.parent != PROJECT_ROOT):
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAW_CSV    = PROJECT_ROOT / "data" / "raw" / "csv"
ILESH_DIR  = PROJECT_ROOT / "members" / "ilesh"
OUT_DIR    = PROJECT_ROOT / "data" / "processed" / "ilesh"
PQDIR      = OUT_DIR / "parquet_base"
FEAT_DIR   = OUT_DIR / "features"
SUB_DIR    = PROJECT_ROOT / "submissions"
DB_PATH    = OUT_DIR / "step1.duckdb"
RPT_PATH   = OUT_DIR / "dq_report.txt"

for d in [PQDIR, FEAT_DIR, SUB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"[pipeline] Project root  : {PROJECT_ROOT}")
print(f"[pipeline] Raw CSV dir   : {RAW_CSV}")
print(f"[pipeline] Parquet dir   : {PQDIR}")
print(f"[pipeline] Features dir  : {FEAT_DIR}")
print(f"[pipeline] DuckDB        : {DB_PATH}")

DIRECTORS_CSV = str(RAW_CSV / "movie_directors.csv")
WRITERS_CSV   = str(RAW_CSV / "movie_writers.csv")
TRAIN_GLOB    = str(RAW_CSV / "train-*.csv")
VAL_CSV       = str(RAW_CSV / "validation_hidden.csv")
TEST_CSV      = str(RAW_CSV / "test_hidden.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 2. TITLE NORMALISATION  (used as DuckDB UDF + in Pandas steps)
# ─────────────────────────────────────────────────────────────────────────────
_STRIP_PUNCT = re.compile(r"[^\w\s]")
_COLLAPSE_WS = re.compile(r"\s+")


def _normalize_title(s: str) -> str:
    """NFKD + ASCII: strips synthetically-injected diacritics (Déstiny→Destiny).
    Safe for legitimate foreign titles (Le Samouraï → Le Samourai)."""
    if not s or s != s:   # empty or NaN
        return s
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = _STRIP_PUNCT.sub(" ", s)
    s = _COLLAPSE_WS.sub(" ", s).strip()
    return s


# ─────────────────────────────────────────────────────────────────────────────
# 3. REPORT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
_log: list[str] = []


def rpt(msg: str = "") -> None:
    print(msg)
    _log.append(str(msg))


def save_report() -> None:
    RPT_PATH.write_text("\n".join(_log))
    print(f"\n[Report → {RPT_PATH}]")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — DuckDB: Data Quality Detection & Fixing
# ═════════════════════════════════════════════════════════════════════════════

def step1_duckdb() -> None:
    import duckdb

    rpt("=" * 68)
    rpt("STEP 1 — DuckDB: Data Quality Detection & Fixing")
    rpt("=" * 68)

    con = duckdb.connect(str(DB_PATH))

    # Register Python title-normalisation as a DuckDB UDF so we can call it in SQL
    con.create_function("normalize_title", _normalize_title, [str], str)

    # ── A) INGESTION ─────────────────────────────────────────────────────────
    # all_varchar=True: preserves \N tokens as literal strings (not silently NaN)
    # F1: Unnamed:0 excluded — it is the spurious pandas index column (range 2–9999)
    rpt("\n>> A) Ingesting raw data...")
    # The first CSV column has no header → DuckDB names it "column0". Drop it.
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

    # F6: drop \N person_id edges at ingestion (2 director, 297 writer rows)
    con.execute(f"""
        CREATE OR REPLACE TABLE directing AS
        SELECT tconst, director AS director_id
        FROM read_csv_auto('{DIRECTORS_CSV}', header=True)
        WHERE director IS NOT NULL AND TRIM(director) != '\\N'
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE writing AS
        SELECT tconst, writer AS writer_id
        FROM read_csv_auto('{WRITERS_CSV}', header=True)
        WHERE writer IS NOT NULL AND TRIM(writer) != '\\N'
    """)

    n_train = con.execute("SELECT COUNT(*) FROM movies_train_raw").fetchone()[0]
    rpt(f"  movies_train_raw : {n_train:,} rows")
    rpt(f"  directing        : {con.execute('SELECT COUNT(*) FROM directing').fetchone()[0]:,} edges")
    rpt(f"  writing          : {con.execute('SELECT COUNT(*) FROM writing').fetchone()[0]:,} edges")

    # ── B) AUTOMATIC DETECTION REPORT (D1–D13) ───────────────────────────────
    rpt("\n>> B) Automatic Data Quality Detection ...")

    # D2: \N token counts per VARCHAR column
    rpt("\n[D2] Disguised \\N tokens per column:")
    for col in ["startYear", "endYear", "runtimeMinutes", "originalTitle", "primaryTitle"]:
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

    # D6: Duplicate tconst
    n_dup = con.execute(
        "SELECT COUNT(*)-COUNT(DISTINCT tconst) FROM movies_train_raw"
    ).fetchone()[0]
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

    # D8: Referential integrity — edges vs ALL movie splits combined
    # Edges cover train+val+test; checking per-split would always show "orphans"
    # from the other splits. Instead check that no edge movie is outside all splits.
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

    # D10: Label balance
    rpt("\n[D10] Label balance:")
    lb = con.execute("""
        SELECT label, COUNT(*) AS n, ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER(),2) AS pct
        FROM movies_train_raw GROUP BY label ORDER BY label
    """).fetchdf()
    rpt(lb.to_string(index=False))

    # D11: Missingness mechanism
    # label is 'True'/'False' string (all_varchar=True) — use CASE to convert
    rpt("\n[D11] Missingness mechanism — label rate: missing vs present:")
    for col, token in [("startYear", "\\N"), ("runtimeMinutes", "\\N")]:
        df = con.execute(f"""
            SELECT (TRIM({col})='{token}') AS is_missing, COUNT(*) AS n,
                   ROUND(AVG(CASE WHEN label='True' THEN 1.0 ELSE 0.0 END),4) AS avg_label
            FROM movies_train_raw GROUP BY is_missing ORDER BY is_missing
        """).fetchdf()
        rpt(f"  {col}:")
        rpt("  " + df.to_string(index=False))
    rpt("  (small difference → keep missingness flags as features)")

    # D13: Text corruption (accent injection)
    rpt("\n[D13] Text corruption — non-ASCII titles:")
    import pandas as pd
    text_df = con.execute("SELECT primaryTitle FROM movies_train_raw").fetchdf()
    n_corrupt = text_df["primaryTitle"].apply(
        lambda x: _normalize_title(str(x)) != str(x)
    ).sum()
    rpt(f"  primaryTitle rows with synthetic accents: {n_corrupt} ({100*n_corrupt/n_train:.1f}%)")

    # ── C) AUTOMATIC FIXES ───────────────────────────────────────────────────
    rpt("\n>> C) Applying fixes ...")
    rpt("[F1] Unnamed:0 excluded at ingestion ✓")
    rpt("[F6] \\N person_ids dropped at ingestion ✓")

    def _build_clean(out_tbl: str, src: str, has_label: bool) -> None:
        """One-pass clean:
        F2: endYear DROPPED (90.1% missing, no signal)
        F3: startYear/runtimeMinutes: TRY_CAST(NULLIF(NULLIF(TRIM),''),'\\N') → INTEGER
        F4: originalTitle NULLs → fallback to primaryTitle
        F9: normalize_title() UDF strips synthetic accents on both title columns
        """
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

    # F7: Director/writer success-rate target encoding
    # INNER JOIN on train labels only → no leakage into val/test
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

    # ── D) POST-FIX VERIFICATION ─────────────────────────────────────────────
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

    # ── E) EXPORT TO PARQUET ─────────────────────────────────────────────────
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
        path = PQDIR / fname
        con.execute(f"COPY {tbl} TO '{path}' (FORMAT PARQUET)")
        n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        rpt(f"  [OK] {tbl:<25}: {n:,} rows → {fname}")

    con.close()
    save_report()
    print(f"\n[STEP 1 DONE] → {PQDIR}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — PySpark: Imputation, Feature Engineering & Scaling
# ═════════════════════════════════════════════════════════════════════════════

def step2_spark() -> tuple:
    """Returns (train_pdf, val_pdf, test_pdf) as Pandas DataFrames with FEATURE_COLS."""
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler
    import numpy as np, pandas as pd

    print("\n>> STEP 2 — PySpark: feature engineering...")

    spark = (
        SparkSession.builder
        .appName("IMDB-Ilesh-Features")
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # A) Load Parquet from Step 1
    movies_train = spark.read.parquet(str(PQDIR / "movies_train.parquet"))
    movies_val   = spark.read.parquet(str(PQDIR / "movies_val.parquet"))
    movies_test  = spark.read.parquet(str(PQDIR / "movies_test.parquet"))
    directing    = spark.read.parquet(str(PQDIR / "directing.parquet"))
    writing      = spark.read.parquet(str(PQDIR / "writing.parquet"))
    dir_success  = spark.read.parquet(str(PQDIR / "director_success.parquet"))
    wri_success  = spark.read.parquet(str(PQDIR / "writer_success.parquet"))

    # B) Missingness flags — BEFORE imputation so they capture real NULL pattern
    def _add_flags(sdf):
        return (sdf
                .withColumn("is_missing_numVotes",
                            F.when(F.col("numVotes").isNull(), 1.0).otherwise(0.0))
                .withColumn("is_missing_startYear",
                            F.when(F.col("startYear").isNull(), 1.0).otherwise(0.0))
                .withColumn("is_missing_runtimeMinutes",
                            F.when(F.col("runtimeMinutes").isNull(), 1.0).otherwise(0.0)))

    movies_train = _add_flags(movies_train)
    movies_val   = _add_flags(movies_val)
    movies_test  = _add_flags(movies_test)

    # C) M:M join — director/writer aggregate features
    # LEFT JOIN: keeps every movie even with no person data
    # Unknown persons in val/test (not in train) produce NULL → imputed to 0.5 below
    def _add_person_features(sdf):
        dir_agg = (
            directing
            .join(dir_success, on="director_id", how="left")
            .groupBy("tconst")
            .agg(
                F.countDistinct("director_id").cast(DoubleType()).alias("num_directors"),
                F.avg("director_success_rate").alias("avg_director_success_rate"),
                F.max("director_success_rate").alias("max_director_success_rate"),
            )
        )
        wri_agg = (
            writing
            .join(wri_success, on="writer_id", how="left")
            .groupBy("tconst")
            .agg(
                F.countDistinct("writer_id").cast(DoubleType()).alias("num_writers"),
                F.avg("writer_success_rate").alias("avg_writer_success_rate"),
                F.max("writer_success_rate").alias("max_writer_success_rate"),
            )
        )
        return sdf.join(dir_agg, on="tconst", how="left").join(wri_agg, on="tconst", how="left")

    movies_train = _add_person_features(movies_train)
    movies_val   = _add_person_features(movies_val)
    movies_test  = _add_person_features(movies_test)

    # D) Cast all numeric cols to Double (MLlib requirement)
    NUMERIC_COLS = [
        "startYear", "runtimeMinutes", "numVotes",
        "num_directors", "num_writers",
        "avg_director_success_rate", "avg_writer_success_rate",
        "max_director_success_rate", "max_writer_success_rate",
    ]

    def _cast_double(sdf):
        for col in NUMERIC_COLS:
            sdf = sdf.withColumn(col, F.col(col).cast(DoubleType()))
        return sdf

    movies_train = _cast_double(movies_train)
    movies_val   = _cast_double(movies_val)
    movies_test  = _cast_double(movies_test)

    # E) Imputation
    # Zero-fill: num_directors/writers — 0 is factually correct (no edges)
    # 0.5-fill: success rates for unknown persons — neutral prior (no info)
    # Median imputation (MLlib Imputer, fit on train only): startYear, runtimeMinutes, numVotes
    ZERO_FILL    = ["num_directors", "num_writers"]
    NEUTRAL_FILL = ["avg_director_success_rate", "avg_writer_success_rate",
                    "max_director_success_rate", "max_writer_success_rate"]

    def _fixed_fills(sdf):
        for col in ZERO_FILL:
            sdf = sdf.withColumn(col, F.coalesce(F.col(col), F.lit(0.0)))
        for col in NEUTRAL_FILL:
            sdf = sdf.withColumn(col, F.coalesce(F.col(col), F.lit(0.5)))
        return sdf

    movies_train = _fixed_fills(movies_train)
    movies_val   = _fixed_fills(movies_val)
    movies_test  = _fixed_fills(movies_test)

    MEDIAN_COLS   = ["startYear", "runtimeMinutes", "numVotes"]
    imputer       = Imputer(strategy="median",
                            inputCols=MEDIAN_COLS,
                            outputCols=[c + "_imp" for c in MEDIAN_COLS])
    imputer_model = imputer.fit(movies_train)   # ← FIT ON TRAIN ONLY

    def _apply_imputer(sdf):
        sdf = imputer_model.transform(sdf)
        for col in MEDIAN_COLS:
            sdf = sdf.drop(col).withColumnRenamed(col + "_imp", col)
        return sdf

    movies_train = _apply_imputer(movies_train)
    movies_val   = _apply_imputer(movies_val)
    movies_test  = _apply_imputer(movies_test)

    # F) Feature engineering
    # log1p(numVotes): addresses extreme right skew (skewness 9.81 → 1.22)
    # True movies average ~5× more votes than False → log linearises the relationship for LR
    def _engineer(sdf):
        sdf = sdf.withColumn("log_numVotes", F.log1p(F.col("numVotes")))
        return sdf.drop("numVotes")

    movies_train = _engineer(movies_train)
    movies_val   = _engineer(movies_val)
    movies_test  = _engineer(movies_test)

    FEATURE_COLS = [
        "startYear",                    # r = −0.264
        "runtimeMinutes",               # r = +0.302  (strongest)
        "log_numVotes",                 # r = +0.246  (after log)
        "num_directors",
        "num_writers",
        "avg_director_success_rate",
        "max_director_success_rate",
        "avg_writer_success_rate",
        "max_writer_success_rate",
        "is_missing_numVotes",
        "is_missing_startYear",
        "is_missing_runtimeMinutes",
    ]

    # G) PySpark MLlib Pipeline: VectorAssembler + StandardScaler
    # withMean=True/withStd=True: critical for LR/SVM (gradient descent scale-sensitive)
    # Fit on train only, transform all splits → serialisable, leakage-free
    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features_raw",
                                handleInvalid="keep")
    scaler    = StandardScaler(inputCol="features_raw", outputCol="features",
                               withMean=True, withStd=True)
    prep_pipeline = Pipeline(stages=[assembler, scaler])
    prep_model    = prep_pipeline.fit(movies_train)   # ← FIT ON TRAIN ONLY

    movies_train = prep_model.transform(movies_train)
    movies_val   = prep_model.transform(movies_val)
    movies_test  = prep_model.transform(movies_test)

    # H) Export flat CSVs for sklearn Step 3
    def _export(sdf, path, has_label):
        import pandas as pd, numpy as np
        keep = ["tconst"] + (["label"] if has_label else [])
        pdf  = sdf.select(keep + ["features"]).toPandas()
        mat  = np.array([v.toArray() for v in pdf["features"]])
        out  = pd.concat([pdf[keep].reset_index(drop=True),
                          pd.DataFrame(mat, columns=FEATURE_COLS)], axis=1)
        out.to_csv(path, index=False)
        print(f"  [OK] {Path(path).name}: {out.shape}  NULLs={out.isnull().sum().sum()}")
        return out

    train_pdf = _export(movies_train, FEAT_DIR / "train_features.csv",  has_label=True)
    val_pdf   = _export(movies_val,   FEAT_DIR / "val_features.csv",    has_label=False)
    test_pdf  = _export(movies_test,  FEAT_DIR / "test_features.csv",   has_label=False)

    spark.stop()
    print(f"[STEP 2 DONE] → {FEAT_DIR}")
    return train_pdf, val_pdf, test_pdf


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — sklearn: Model Training & Predictions
# ═════════════════════════════════════════════════════════════════════════════

FEATURE_COLS_FLAT = [
    "startYear", "runtimeMinutes", "log_numVotes",
    "num_directors", "num_writers",
    "avg_director_success_rate", "max_director_success_rate",
    "avg_writer_success_rate",   "max_writer_success_rate",
    "is_missing_numVotes", "is_missing_startYear", "is_missing_runtimeMinutes",
]


def step3_model(train_pdf=None, val_pdf=None, test_pdf=None) -> None:
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    print("\n>> STEP 3 — sklearn: model training...")

    # Load CSVs if not passed in from Step 2
    if train_pdf is None:
        train_pdf = pd.read_csv(FEAT_DIR / "train_features.csv")
        val_pdf   = pd.read_csv(FEAT_DIR / "val_features.csv")
        test_pdf  = pd.read_csv(FEAT_DIR / "test_features.csv")

    avail = [c for c in FEATURE_COLS_FLAT if c in train_pdf.columns]
    X_all = train_pdf[avail].fillna(0).astype(float)
    y_all = (train_pdf["label"].astype(str).str.strip() == "True").astype(int)

    X_tr, X_hv, y_tr, y_hv = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    print(f"  Train={len(X_tr):,}  internal-val={len(X_hv):,}  features={len(avail)}")

    # Features are already StandardScaled by PySpark pipeline — no extra scaler needed
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_tr, y_tr)

    acc = accuracy_score(y_hv, model.predict(X_hv))
    auc = roc_auc_score(y_hv, model.predict_proba(X_hv)[:, 1])
    print(f"  Internal val — Accuracy={acc:.4f}  ROC-AUC={auc:.4f}")

    # Write submissions
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def _write_sub(X, path):
        preds = model.predict(X[avail].fillna(0).astype(float))
        lines = ["True" if p == 1 else "False" for p in preds]
        path.write_text("\n".join(lines))
        print(f"  [OK] {path.name}: {len(lines)} predictions")

    _write_sub(val_pdf,  SUB_DIR / f"ilesh_val_{ts}.txt")
    _write_sub(test_pdf, SUB_DIR / f"ilesh_test_{ts}.txt")
    print(f"[STEP 3 DONE] → {SUB_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    step1_duckdb()
    train_pdf, val_pdf, test_pdf = step2_spark()
    step3_model(train_pdf, val_pdf, test_pdf)
    print("\n[pipeline] ALL STEPS COMPLETE.")
