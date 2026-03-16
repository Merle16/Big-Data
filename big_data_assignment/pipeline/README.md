# Pipeline

End-to-end machine learning pipeline for predicting IMDb movie hits. A single entry point (`run.py`) orchestrates five phases: data cleaning, external enrichment, feature engineering, model training, and HTML report generation. The pipeline is fully modular — each phase lives in its own subfolder and can be run independently or in any combination using stage flags.


## Running the pipeline

All commands are run from the project root (`big_data_assignment/`).

```bash
# Full pipeline with no enrichment
python pipeline/run.py

# Full pipeline with both enrichments
python pipeline/run.py --genre --rt-oscar

# Genre enrichment only (default data/Movies_by_Genre/ folder)
python pipeline/run.py --genre

# Genre enrichment from a custom path
python pipeline/run.py --genre /path/to/Movies_by_Genre

# RT and Oscar enrichment only
python pipeline/run.py --rt-oscar

# Skip cleaning, use existing parquets (useful when iterating on features or models)
python pipeline/run.py --from features

# Skip cleaning and features, just retrain models
python pipeline/run.py --from models

# Run one stage only
python pipeline/run.py --only models
python pipeline/run.py --only report

# Run an explicit subset of stages in canonical order
python pipeline/run.py --stages features models report
```

The final output is written to `pipeline/outputs/full_pipeline_report.html` — a single self-contained HTML file covering every phase with figures, audit tables, and model results.


## Stage and enrichment toggles

### Stage flags

The three stage flags `--from`, `--only`, and `--stages` are mutually exclusive. When none of them is provided, all four stages run in order.

| Flag | Behaviour |
|---|---|
| (none) | Run all four stages: cleaning → features → models → report |
| `--from STAGE` | Skip everything before STAGE and run from it to the end |
| `--only STAGE` | Run exactly one named stage |
| `--stages A B ...` | Run the listed stages in canonical order |

Valid stage names are `cleaning`, `features`, `models`, and `report`.

### Enrichment flags

Enrichment steps are opt-in and run automatically after cleaning, before features. They patch the output parquets in place. Both flags can be combined freely and are independent of the stage flags.

| Flag | What it does |
|---|---|
| `--genre [PATH]` | Joins Movies_by_Genre genre labels onto all three splits. Omit PATH to use the default `data/Movies_by_Genre/` folder. |
| `--rt-oscar` | Joins Rotten Tomatoes scores and Academy Award nomination history onto all three splits. Includes MNAR missingness analysis. |

The legacy flag `--external-dataset` is kept as a hidden alias for `--genre` for backwards compatibility.


## Project structure

```
pipeline/
    run.py

    data_cleaning/
        __init__.py
        schema.py
        steps.py
        report.py
        utils/
            figures.py
            audits.py
            quality_report.py

    enrichment/
        __init__.py
        genre.py
        rt_oscar.py

    feature_engineering/
        __init__.py
        f1_candidate_features.py
        f2_feature_selection.py
        f3_feature_quality.py

    models/
        __init__.py
        m1_train.py
        m2_diagnostics.py
        m3_reduced_model.py
        m4_export.py

    reporting/
        __init__.py
        make_full_report.py
```


## Phase 1a — Data cleaning (`data_cleaning/`)

The cleaning phase reads raw IMDb CSVs, applies nine sequential transformation steps inside a DuckDB in-memory database, and exports three cleaned parquet files: `train_clean.parquet`, `validation_hidden_clean.parquet`, and `test_hidden_clean.parquet`.

### `schema.py`

Single source of truth for the relational schema. Declares the primary key, ID columns, UUID regex patterns, and columns to drop for every table in the pipeline. Every downstream step reads from here instead of auto-detecting structure at runtime. The `validate()` function runs DuckDB SQL checks to verify UUID format compliance and key uniqueness.

### `steps.py`

Contains all eight transformer classes. Each class exposes a `.transform(con, table)` method that creates a new DuckDB view and returns its name, making the steps composable in a chain.

**`MissingTokenReplacer` (step 1)** replaces disguised-null tokens — values like `\N` and `\\N` that IMDb uses to represent missing data — with SQL NULL across all columns. The token list is declared at the top of the file and is intended to be edited after running `quality_report.py` on a new dataset.

**`DTypeEnforcer` (step 2)** probes each VARCHAR column to determine whether it can be safely cast to BOOLEAN, INTEGER, or DOUBLE. It uses ratio thresholds (e.g. more than 95% true/false values → BOOLEAN, more than 50% parseable as numeric → INTEGER or DOUBLE) rather than trying to cast every value, making it robust to mixed-type columns.

**`StringStandardizer` (step 3)** applies NFKD Unicode normalisation to all text columns, strips punctuation, and collapses whitespace. ID columns (from `schema.py`) are skipped so join keys are never modified. A `__fp_*` fingerprint column is added alongside each normalised column for near-duplicate detection during deduplication.

**`Deduplicator` (step 4)** removes duplicate rows using the key declared in `schema.py`. For tables not listed in the schema it falls back to auto-detecting the first column that looks like an ID (alphanumeric pattern, fully unique). Uses `ROW_NUMBER() OVER (PARTITION BY key ORDER BY key)` so the selection is deterministic.

**`JoinBuilder` (step 5)** assembles a wide analytical view (one row per `tconst`) by joining cleaned reference tables onto the base split. One-to-one joins (e.g. `title_basics` → genres, titleType, isAdult) and many-to-many aggregations (movie directors and writers → aggregated person metadata) are declared in two editable config dicts at the top of the file. Join keys are auto-detected from the runtime schema rather than hardcoded.

**`Normalizer` (step 6)** applies `log1p` (i.e. `LN(1 + x)`) to heavy-tailed numeric columns. Currently only `numVotes` is transformed, producing `numVotes_log1p`. This is done before MICE imputation so that imputed values are on a more Gaussian-like scale.

**`MICEImputer` (step 7)** fits scikit-learn `IterativeImputer` (MICE) on the training split only and applies the fitted model to all three splits. This is the correct approach: fitting on validation or test data would leak their missingness patterns into the imputed values. After imputation, integer-valued columns (`startYear`, `runtimeMinutes`) are rounded back to integers because MICE produces floats.

**`assert_quality` and `save_parquet` (step 8)** are functions rather than a class. `assert_quality` raises `ValueError` if any schema violations or remaining NULLs in the imputed columns are detected. `save_parquet` materialises the final DuckDB view to Parquet, dropping internal `__fp_*` fingerprint columns and raw ID string columns that were only needed during the pipeline.

### `report.py`

Runs after all parquets are written. Loads both raw CSVs and cleaned parquets, performs nine validity check categories (row count preservation, remaining NULLs, domain bounds, MICE invariant check, missingness stability, outliers, join coverage, label balance), prints a pass/warn/fail summary to stdout, and saves figures to `pipeline/outputs/cleaning/`. Also delegates to `utils/audits.py` for deeper imputation quality tests and join fanout checks.

### `utils/figures.py`

Pure matplotlib helper functions. No pipeline logic, no file I/O. Each function accepts plain DataFrames and returns a `matplotlib.Figure`. Saving is handled by the caller (`report.py`). Figures include missingness before/after, numeric distributions, label balance, join coverage, domain bounds, MICE invariant, imputation summary, distributions by label, class separation, and outlier summaries.

### `utils/audits.py`

Two audit modules combined into one file.

`run_imputation()` performs deep imputation quality validation: it masks 20% of observed values, runs MICE, and measures MAE/RMSE/within-tolerance against median and mean baselines. It also computes KS statistics, Wasserstein distance, PSI, correlation preservation between complete-case and imputed matrices, and conditional plausibility per titleType.

`run_join()` validates join correctness: it checks `tconst` uniqueness after every join stage to detect 1-to-many fanout that silently duplicates rows, reconciles row counts through the raw→clean funnel, and computes PSI/KS/mean/std drift for each numeric column across train, validation, and test splits.

### `utils/quality_report.py`

Standalone CLI audit tool, not called by the pipeline. Run it directly on any CSV or Parquet file to get a schema-agnostic quality audit covering column types, missingness rates, disguised-null tokens, outliers (MAD, IQR, k-trimmed-mean), fingerprint-keyed near-duplicate candidates, and cross-table linkage. Useful for initial exploration of a new dataset before configuring the pipeline.

```bash
python -m pipeline.data_cleaning.utils.quality_report data/raw/csv/train-1.csv
```


## Phase 1b — Genre enrichment (`enrichment/genre.py`)

Joins genre labels from a folder of per-genre CSV files (`Movies_by_Genre/`) onto all three splits using normalised title matching. Evaluates multiple join methods and selects the one with the best composite score. Adds binary genre indicator columns and a `genre_hit_rate` feature. Writes outputs to `pipeline/outputs/enrichment/`.

Activated with `--genre` or `--genre /custom/path`.


## Phase 1c — RT and Oscar enrichment (`enrichment/rt_oscar.py`)

Joins two external datasets onto all three splits. Neither file contains an IMDb `tconst` key, so both joins use normalised title matching with year and runtime guards.

**Rotten Tomatoes join:** Nine methods are evaluated, ranging from exact title + exact year to fuzzy SequenceMatcher (≥0.90 ratio) within a year±1 block. Each method is scored on a composite of coverage (fraction of IMDb movies matched), unique ratio (fraction of movies with exactly one RT candidate), and label alignment (whether tomatometer ≥ 75 agrees with the IMDb hit label). The best method is selected automatically. Each split is joined independently to prevent tconst leakage from training matches.

**Oscar join:** Title + year±1 matching. The raw dataset has one row per nomination, so it is aggregated to one row per film before joining. Films with no Oscar record receive 0 for all Oscar features — absence of a nomination is itself an informative signal, not a missing value.

**MNAR missingness analysis:** Before any imputation decision, the module applies the same MCAR/MAR/MNAR testing framework used in the IMDb cleaning phase. Point-biserial correlations test whether each missingness indicator correlates with observed IMDb numeric fields (startYear, runtimeMinutes, numVotes_log1p). Decade-level missingness rates test for systematic temporal variation. Both RT join missingness and within-RT score missingness are classified as MNAR — absence correlates with popularity, which correlates with the hit label. Oscar absence is structural MNAR — absence means not nominated. The correct decision for all three is no imputation: XGBoost handles NaN natively, and `rt_match_flag` explicitly encodes the join-level missingness as a feature.

Activated with `--rt-oscar`.


## Phase 2 — Feature engineering (`feature_engineering/`)

Reads the cleaned (and optionally enriched) parquets and builds a supervised feature matrix.

### `f1_candidate_features.py`

Generates all candidate features. Covers IMDb base features (title length, decade, runtime buckets, vote count tiers), genre-based features (multi-hot genre encoding, genre hit rates from training data), person-level features (director and writer hit rates estimated from training data using a leave-one-out approach to avoid self-leakage), title similarity features (token overlap between primary and original title), and all 14 RT and Oscar features when present. Outputs `features_train.parquet` and corresponding CSV files.

### `f2_feature_selection.py`

Reduces the candidate feature set using three independent filters applied in sequence. OOF AUC filter: runs 5-fold cross-validation with a lightweight XGBoost model and drops features whose out-of-fold AUC contribution falls below a threshold. PSI stability filter: drops features where the Population Stability Index between train and validation exceeds a threshold, indicating distribution shift that would hurt generalisation. Variance filter: drops near-zero-variance features. Outputs `features_train_prepped.parquet`.

### `f3_feature_quality.py`

Computes three goodness metrics for every feature in the prepped matrix: mutual information with the label (MI), individual AUC from a single-feature logistic regression, and PSI between train and validation. Classifies each feature as keep, drop candidate, or review. Outputs `feature_goodness.csv` and figures showing the distribution of each metric.


## Phase 3 — Model training (`models/`)

### `m1_train.py`

Trains two models on the feature matrix: Logistic Regression (scikit-learn, L2 regularisation) and XGBoost. Both are evaluated using 5-fold stratified cross-validation with AUC as the metric. The best model is selected automatically. Outputs figures and a pickled model file.

### `m2_diagnostics.py`

Runs permutation importance on the winning model to identify which features contribute most and which can be safely dropped without hurting AUC. Each feature is permuted 10 times and the mean AUC drop is recorded. Features with a drop below a threshold are classified as drop candidates. Outputs `feature_diagnostics.csv`.

### `m3_reduced_model.py`

Runs an ablation study: starts with the top-5 features by permutation importance and adds one feature at a time until AUC plateaus. Retrains a final model on the resulting keep-set. This produces a reduced model that is smaller, faster, and typically as good as or slightly better than the full model due to reduced noise from irrelevant features.

### `m4_export.py`

Generates validation set predictions (`predictions_val.csv`), runs threshold analysis to show the precision/recall tradeoff at different classification cutoffs, and saves final export figures. This is the artefact used for the competition submission.


## Phase 4 — Report generation (`reporting/make_full_report.py`)

Reads all pipeline outputs — parquets, figures, CSVs — and assembles a single self-contained HTML file with embedded base64 images. The report covers every phase with analysis cards following a structured format: objective, how to read the figure, result, threshold/decision logic, implication, and action taken. Navigation is provided by a fixed sidebar with phase colour coding.

Activated with `python pipeline/run.py --only report` after any pipeline run.


## Output files

All pipeline outputs land under `pipeline/outputs/`.

```
pipeline/outputs/
    cleaning/
        train_clean.parquet
        validation_hidden_clean.parquet
        test_hidden_clean.parquet
        01_missingness.png  through  18_*.png

    enrichment/
        (genre enrichment outputs)

    enrichment_rt_oscar/
        rt_missingness_mechanism.csv
        rt_decade_missingness.csv
        rt_missingness_corr.csv
        01_rt_join_method_comparison.png  through  06_missingness_mechanism.png
        (plus audit CSVs for RT and Oscar joins)

    features/
        features_train.parquet
        features_train_prepped.parquet
        feature_goodness.csv
        (figures: AUC bars, MI, PSI, OOF distributions)

    models/
        (figures: ROC, confusion matrix, permutation importance, ablation curve)

    full_pipeline_report.html
```

Model artefacts that are not tracked in git (`models.pkl`, `reduced_model.pkl`, `predictions_val.csv`, `threshold_analysis.csv`) are written to `data/processed/`.


## Missingness policy

This table summarises every missingness decision made across all phases.

| Column or dataset | Mechanism | Decision and rationale |
|---|---|---|
| IMDb `runtimeMinutes` | MAR — correlates with `titleType` (shorts have very different runtimes than features) | MICE imputation, fit on train only |
| IMDb `startYear` | MAR — correlates with `numVotes` (older films less likely to have complete metadata) | MICE imputation, fit on train only |
| RT join miss (`rt_match_flag = 0`) | MNAR — unmatched movies are systematically low-popularity, which correlates with the hit label | No imputation. `rt_match_flag` is a feature that encodes this signal directly |
| RT score NaNs (within matched rows) | MNAR — niche films that did match still have missing audience scores, correlating with popularity | No imputation. XGBoost learns a split direction for NaN natively |
| Oscar absence | Structural MNAR — absence means the film was not nominated, not that the data is missing | Encoded as 0. Absence is informative and correct |

Evidence for MNAR classification is in `pipeline/outputs/enrichment_rt_oscar/rt_missingness_mechanism.csv` and the three-panel figure `06_missingness_mechanism.png`.


## Current model results

| Model | Validation AUC |
|---|---|
| Logistic Regression | 0.9460 |
| XGBoost (full, 61 features) | 0.9596 |
| XGBoost (reduced, 22 features) | 0.9610 |

AUC progression by phase: IMDb baseline 0.8981 → after genre enrichment 0.9406 → after RT and Oscar enrichment 0.9620.
