#!/usr/bin/env python3
"""
members/ilesh/pipeline.py
=========================
Orchestrator — calls each step in sequence.

Steps
-----
  dq.py       Step 1 — DuckDB:  13 DQ checks, 9 fixes, Parquet export
  features.py Step 2 — PySpark: imputation, feature engineering, scaling
  model.py    Step 3 — sklearn: LogisticRegression, leakage-free evaluation,
                                submission files

Each step can also be imported and run independently:
    from members.ilesh.dq       import step1_duckdb
    from members.ilesh.features import step2_spark
    from members.ilesh.model    import step3_model

Run locally (from big_data_assignment/):
    python members/ilesh/pipeline.py

Run in Docker:
    docker build -t big-data-imdb .
    docker run --rm \\
      -v $(pwd)/data:/app/data \\
      -v $(pwd)/members:/app/members \\
      -v $(pwd)/submissions:/app/submissions \\
      big-data-imdb \\
      python members/ilesh/pipeline.py

Tool split rationale
────────────────────
DuckDB  (dq.py)       — ingestion, 13 DQ checks, 9 fixes, IQR/success-rate
                        computation, Parquet export. In-process SQL; no
                        cluster needed for 8K rows. Each check = one SQL
                        string → trivial to update if the schema changes.
PySpark (features.py) — Parquet load, MLlib Imputer (fit train / transform
                        val+test), VectorAssembler + StandardScaler pipeline.
                        Serialisable pipeline object = leakage-free distributed
                        pattern from the course; scales to a cluster by
                        changing .master() with zero code changes.
sklearn (model.py)    — LogisticRegression on the scaled feature matrix.
                        Fast, familiar API. Features already scaled by Spark
                        — no extra preprocessing. No JVM overhead justified
                        at 8K rows.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Resolve project root (works locally, in Docker at /app, from any cwd) ────
_here = Path(__file__).resolve()
PROJECT_ROOT = _here.parent.parent.parent
while (PROJECT_ROOT.name != "big_data_assignment"
       and not (PROJECT_ROOT / "config" / "config.yaml").exists()
       and PROJECT_ROOT.parent != PROJECT_ROOT):
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_CSV  = PROJECT_ROOT / "data" / "raw" / "csv"
OUT_DIR  = PROJECT_ROOT / "data" / "processed" / "ilesh"
PQDIR    = OUT_DIR / "parquet_base"
FEAT_DIR = OUT_DIR / "features"
SUB_DIR  = PROJECT_ROOT / "submissions"
DB_PATH  = OUT_DIR / "step1.duckdb"
RPT_PATH = OUT_DIR / "dq_report.txt"

print(f"[pipeline] Project root  : {PROJECT_ROOT}")
print(f"[pipeline] Raw CSV dir   : {RAW_CSV}")
print(f"[pipeline] Parquet dir   : {PQDIR}")
print(f"[pipeline] Features dir  : {FEAT_DIR}")
print(f"[pipeline] DuckDB        : {DB_PATH}")

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from members.ilesh.dq       import step1_duckdb
    from members.ilesh.features import step2_spark
    from members.ilesh.model    import step3_model

    step1_duckdb(RAW_CSV, PQDIR, DB_PATH, RPT_PATH)
    train_pdf, val_pdf, test_pdf = step2_spark(PQDIR, FEAT_DIR)
    step3_model(train_pdf, val_pdf, test_pdf, RAW_CSV, FEAT_DIR, SUB_DIR)

    print("\n[pipeline] ALL STEPS COMPLETE.")
