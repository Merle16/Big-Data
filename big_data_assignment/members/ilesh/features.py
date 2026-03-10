"""
members/ilesh/features.py
=========================
Step 2 — PySpark: Imputation, Feature Engineering & Scaling.

WHY PySpark HERE?
  - MLlib Imputer is a Pipeline stage: fit on train, transform val/test
    with the same medians → standard leakage-free distributed pattern
    from the course.
  - VectorAssembler + StandardScaler pipeline is serialisable: re-running
    on new data only requires .transform(), not rewriting logic.
  - Scales to a real cluster by changing .master() — zero code changes.

Can be imported and called independently:
    from members.ilesh.features import step2_spark, FEATURE_COLS
    train_pdf, val_pdf, test_pdf = step2_spark(pqdir, feat_dir)
"""

from __future__ import annotations

from pathlib import Path

# Feature columns produced by this step (used by model.py and pipeline.py)
FEATURE_COLS = [
    "startYear",                    # r = −0.264
    "runtimeMinutes",               # r = +0.302  (strongest single feature)
    "log_numVotes",                 # r = +0.246  (log1p reduces skew 9.81→1.22)
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


def step2_spark(pqdir: Path, feat_dir: Path) -> tuple:
    """
    Load DuckDB Parquet output, engineer features, scale, and export CSVs.

    Parameters
    ----------
    pqdir    : directory containing Parquet files from step1_duckdb()
    feat_dir : output directory for train/val/test feature CSVs

    Returns
    -------
    (train_pdf, val_pdf, test_pdf) — Pandas DataFrames with FEATURE_COLS
    """
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler
    import numpy as np
    import pandas as pd

    feat_dir.mkdir(parents=True, exist_ok=True)
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
    movies_train = spark.read.parquet(str(pqdir / "movies_train.parquet"))
    movies_val   = spark.read.parquet(str(pqdir / "movies_val.parquet"))
    movies_test  = spark.read.parquet(str(pqdir / "movies_test.parquet"))
    directing    = spark.read.parquet(str(pqdir / "directing.parquet"))
    writing      = spark.read.parquet(str(pqdir / "writing.parquet"))
    dir_success  = spark.read.parquet(str(pqdir / "director_success.parquet"))
    wri_success  = spark.read.parquet(str(pqdir / "writer_success.parquet"))

    # B) Missingness flags — BEFORE imputation so they capture the real NULL pattern
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
    # LEFT JOIN: keeps every movie even with no person data.
    # Unknown persons in val/test (not in train) produce NULL → imputed to 0.5.
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
    # 0.5-fill:  success rates for unknown persons — neutral prior
    # Median (MLlib Imputer, FIT ON TRAIN ONLY): startYear, runtimeMinutes, numVotes
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
    # log1p(numVotes): skewness 9.81 → 1.22; True movies average ~5× more votes
    def _engineer(sdf):
        return sdf.withColumn("log_numVotes", F.log1p(F.col("numVotes"))).drop("numVotes")

    movies_train = _engineer(movies_train)
    movies_val   = _engineer(movies_val)
    movies_test  = _engineer(movies_test)

    # G) PySpark MLlib Pipeline: VectorAssembler + StandardScaler
    # withMean=True/withStd=True: critical for LR (gradient descent is scale-sensitive)
    # FIT ON TRAIN ONLY → serialisable, leakage-free
    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features_raw",
                                handleInvalid="keep")
    scaler    = StandardScaler(inputCol="features_raw", outputCol="features",
                               withMean=True, withStd=True)
    prep_model = Pipeline(stages=[assembler, scaler]).fit(movies_train)

    movies_train = prep_model.transform(movies_train)
    movies_val   = prep_model.transform(movies_val)
    movies_test  = prep_model.transform(movies_test)

    # H) Export flat CSVs for sklearn Step 3
    def _export(sdf, path, has_label):
        keep = ["tconst"] + (["label"] if has_label else [])
        pdf  = sdf.select(keep + ["features"]).toPandas()
        mat  = np.array([v.toArray() for v in pdf["features"]])
        out  = pd.concat([pdf[keep].reset_index(drop=True),
                          pd.DataFrame(mat, columns=FEATURE_COLS)], axis=1)
        out.to_csv(path, index=False)
        print(f"  [OK] {Path(path).name}: {out.shape}  NULLs={out.isnull().sum().sum()}")
        return out

    train_pdf = _export(movies_train, feat_dir / "train_features.csv",  has_label=True)
    val_pdf   = _export(movies_val,   feat_dir / "val_features.csv",    has_label=False)
    test_pdf  = _export(movies_test,  feat_dir / "test_features.csv",   has_label=False)

    spark.stop()
    print(f"[STEP 2 DONE] → {feat_dir}")
    return train_pdf, val_pdf, test_pdf
