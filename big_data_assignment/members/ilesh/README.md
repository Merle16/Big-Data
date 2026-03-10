# Ilesh — IMDB Data Pipeline

End-to-end pipeline: raw CSV shards → DuckDB quality fixes → PySpark feature engineering → LogisticRegression → predictions.

---

## Files

```
members/ilesh/
├── pipeline.py                  ← standalone script (mirrors the notebook, Docker-runnable)
├── notebooks/
│   ├── data_pipeline.ipynb      ← full pipeline with rationale, diagram, reusability discussion
│   ├── eda.ipynb                ← EDA justifying every pipeline decision with plots
│   └── baseline.ipynb           ← shared baseline (untouched)
└── artifacts/                   ← auto-created; stores pipeline diagram + EDA plots
```

Outputs written to `data/processed/ilesh/` and `submissions/`.

---

## Running with Docker (recommended — handles Java + all dependencies)

Run all commands from the **`big_data_assignment/`** root directory.

### 1. Build the image (once)

```bash
docker build -t big-data-imdb .
```

This installs Java 21, DuckDB ≥1.0, PySpark 3.5, sklearn, numpy, pandas, matplotlib, and all other dependencies from `requirements.txt`.

### 2. Run the full pipeline

```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/members:/app/members \
  -v $(pwd)/submissions:/app/submissions \
  big-data-imdb \
  python members/ilesh/pipeline.py
```

**What this does:**
| Step | Tool | Output |
|---|---|---|
| Step 1 — DQ detection & fixing | DuckDB | `data/processed/ilesh/parquet_base/` (7 Parquet files) + `dq_report.txt` |
| Step 2 — Imputation & feature engineering | PySpark MLlib | `data/processed/ilesh/features/` (3 CSVs) |
| Step 3 — Model + predictions | sklearn | `submissions/ilesh_val_<ts>.txt` + `submissions/ilesh_test_<ts>.txt` |

### 3. Open notebooks in Docker (optional)

```bash
docker run -it --rm -p 8888:8888 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/members:/app/members \
  -v $(pwd)/submissions:/app/submissions \
  big-data-imdb \
  jupyter lab --ip=0.0.0.0 --no-browser --allow-root --notebook-dir=/app
```

Then open `http://localhost:8888` in your browser and navigate to `members/ilesh/notebooks/`.

---

## Running locally (Step 1 only — no Java required)

Step 1 (DuckDB) runs on plain Python with no Java. Steps 2–3 require Java for PySpark.

```bash
pip install duckdb pandas numpy scikit-learn matplotlib
python3 members/ilesh/pipeline.py   # Step 2 will fail gracefully without Java
```

To run only Step 1:

```python
from members.ilesh.pipeline import step1_duckdb
step1_duckdb()
```

---

## Assignment questions answered

| Question | Where |
|---|---|
| Pipeline diagram | `notebooks/data_pipeline.ipynb` — Section 2 (colour-coded by tool) |
| Reusability | `notebooks/data_pipeline.ipynb` — Section 6 (table of design decisions + schema-change effort) |
| Schema evolution effort | `notebooks/data_pipeline.ipynb` — Section 6 ("How much effort if the schema changes?" table) |
| DuckDB vs PySpark rationale | `notebooks/data_pipeline.ipynb` — intro table + Section 3 + Section 4 markdown cells |
| EDA / pipeline justification | `notebooks/eda.ipynb` — 11 sections with plots for every decision |
| End-to-end predictions | `submissions/ilesh_val_*.txt` + `submissions/ilesh_test_*.txt` |

---

## Tool-choice rationale (summary)

**DuckDB** — Steps 1 (ingestion, 13 DQ checks, 9 fixes, IQR/success-rate computation, Parquet export)
- In-process SQL; no cluster overhead for 8K rows
- `TRY_CAST`, `QUANTILE_CONT`, `COPY TO PARQUET`, and UDF registration are single SQL calls
- Each DQ check = one SQL string → trivial to update if schema changes

**PySpark** — Step 2 (Parquet load, MLlib Imputer, VectorAssembler + StandardScaler pipeline)
- MLlib `Imputer` is a Pipeline stage: fit on train, transform val/test with the same medians → leakage-free distributed pattern
- Serialisable pipeline object: re-running on new data only requires `.transform()`, not rewriting logic

**sklearn** — Step 3 (LogisticRegression, predictions)
- Features already scaled by Spark pipeline — no extra preprocessing
- Fast, familiar API; no JVM overhead justified at 8K rows

---

## DQ fixes summary

| Fix | What | Decision |
|---|---|---|
| F1 | Drop `column0` (spurious pandas index) | Excluded at read via `EXCLUDE (column0)` |
| F2 | Drop `endYear` | 90.1% `\N`; no row has both startYear+endYear → imputation impossible |
| F3 | `\N`→NULL for `startYear`, `runtimeMinutes` | `TRY_CAST(NULLIF(NULLIF(TRIM),''),'\N')` handles whitespace variants |
| F4 | `originalTitle` NULLs | `COALESCE` → `primaryTitle` (50.1% missing) |
| F5 | `numVotes` IQR cap | Train fence=26,074 applied to all splits (no leakage) |
| F6 | Drop `\N` person IDs | 297 writer rows removed at ingestion |
| F7 | Director/writer success-rate encoding | `INNER JOIN` train labels only → leakage-free |
| F9 | NFKD title normalisation | DuckDB UDF; 30% of titles have synthetic accent injection |
