# Djamel IMDb Pipeline

This folder contains the standalone pipeline and analysis notebooks for the IMDb assignment.

## Files Included

1. `imdb_pipeline_audit_and_fix.py`
2. `notebooks/feature_phase_workflow.ipynb`
3. `notebooks/feature_diagnostics_figures.ipynb`
4. `README.md`

## Run Order

1. Run the pipeline script first:

```bash
/home/djameldino/anaconda3/envs/uva/bin/python members/djamel/imdb_pipeline_audit_and_fix.py
```

2. Open `notebooks/feature_phase_workflow.ipynb` for staged baseline-vs-enhanced feature workflow.
3. Open `notebooks/feature_diagnostics_figures.ipynb` for diagnostics figures and feature-level analysis.

## What the Pipeline Does

- Converts `directing.json` and `writing.json` into many-to-many edges using **DuckDB**.
- Cleans disguised missing values (`\N`, `\\N`, null-like tokens), normalizes datatypes, and checks duplicates.
- Engineers relational and text-based features (including leakage-safe OOF encodings like `director_hit_rate` and `writer_hit_rate`).
- Applies train-only capping/imputation, trains models, runs feature diagnostics, and exports artifacts.

## Engine Usage

- **DuckDB**: JSON-to-many-to-many conversion step.
- **Spark**: not used in this implementation.
