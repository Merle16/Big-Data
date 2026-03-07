#!/usr/bin/env python3
"""
Standalone IMDB pipeline that fixes the main repository issues without editing teammate code.

What this script does end-to-end:
1. Audits the current repository setup and many-to-many JSON handling.
2. Rebuilds directing/writing many-to-many edges from raw JSON with DuckDB.
3. Cleans movie + edge data (including robust missing token handling).
4. Engineers model features (counts, caps, missing flags, OOF target encodings).
5. Trains a reference logistic model on an internal split and exports artifacts.
6. Writes a comparison report: needed fix -> why -> how fixed.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


SEED = 42
MISSING_TOKENS = {"\\N", "\\\\N", "", "NA", "N/A", "null", "None", "nan", "NaN"}

STEP_ENGINE_USAGE: Dict[str, Tuple[str, str]] = {
    "load_data": ("Pandas", "Load shard CSV files into memory."),
    "json_to_many_to_many_duckdb": ("DuckDB", "Parse JSON and rebuild many-to-many edge tables safely."),
    "clean_movies": ("Pandas", "Normalize missing tokens/types and de-duplicate movie keys."),
    "clean_edges": ("Pandas", "Normalize and clean relation edges before aggregation."),
    "feature_aggregates_many_to_many": ("Pandas", "Aggregate cleaned edges into movie-level relation counts."),
    "feature_engineering_base": ("Pandas/NumPy", "Create base title/year/missingness features."),
    "attach_people_columns": ("Pandas", "Attach director_ids/writer_ids for enriched exports."),
    "split_internal_validation": ("scikit-learn", "Create stratified internal validation split."),
    "outlier_capping_fit_apply": ("Pandas/NumPy", "Fit train-only quantile caps and apply."),
    "target_encoding_oof": ("Pandas/NumPy/scikit-learn", "OOF leakage-safe target encoding."),
    "title_group_conflict_features": ("Pandas", "Build canonical-title conflict features."),
    "title_similarity_features": ("scikit-learn", "TF-IDF + cosine similarity title signals."),
    "impute_and_scale": ("Pandas/scikit-learn", "Median imputation + standardization."),
    "feature_goodness_analysis": ("scikit-learn", "Univariate AUC, MI, and drift diagnostics."),
    "train_and_validate_models": ("scikit-learn/XGBoost", "Fit baseline models and evaluate."),
    "feature_diagnostics_ablation_shap": ("XGBoost/scikit-learn", "Permutation AUC drop + SHAP-like diagnostics."),
    "reduced_feature_retrain": ("scikit-learn/XGBoost", "Retrain with reduced keep-set features."),
    "export_artifacts": ("Pandas", "Write outputs, predictions, and metrics."),
}

# The explicit "what needed to be fixed / why / how fixed" map requested by the user.
ISSUE_FIX_LOG: List[Dict[str, str]] = [
    {
        "needed_fix": "Missing src.models package referenced by shared baseline/train code.",
        "why": "python -m src.pipeline.baseline crashes immediately with ModuleNotFoundError.",
        "how_fixed": "Implemented an independent, runnable model pipeline in this script without touching shared files.",
        "evidence": "Runtime error confirmed during audit.",
    },
    {
        "needed_fix": "config/config.yaml points to imdb_train.csv/imdb_validation.csv that are not present.",
        "why": "train.py fails with FileNotFoundError before any model step.",
        "how_fixed": "Loads shard files directly from data/raw/csv/train-*.csv and hidden splits from data/raw/csv/*.csv.",
        "evidence": "Runtime error confirmed during audit.",
    },
    {
        "needed_fix": "Shared train/predict scaffolds are not implemented.",
        "why": "predict.py raises NotImplementedError by design; shared path is non-functional.",
        "how_fixed": "This script provides a complete train/validate/predict flow and writes prediction files.",
        "evidence": "Runtime error confirmed during audit.",
    },
    {
        "needed_fix": "JSON conversion utility expects review/label JSON schema.",
        "why": "IMDB JSON has many-to-many movie-person structure, not review text rows.",
        "how_fixed": "Directing and writing JSON are normalized into edge tables (tconst, director_id/writer_id).",
        "evidence": "Schema mismatch between src/data/json_to_tabular.py and data/raw/json/*.json.",
    },
    {
        "needed_fix": "movie_directors.csv contains double-escaped unknown token (\\\\N) for some rows.",
        "why": "Simple '\\N' replacement misses these values and leaves dirty unknown IDs.",
        "how_fixed": "Cleaning step normalizes both '\\N' and '\\\\N' as missing.",
        "evidence": "Observed director token lengths 3 ('\\\\\\\\N') in data/raw/csv/movie_directors.csv.",
    },
    {
        "needed_fix": "Notebook many_to_many.ipynb is empty.",
        "why": "No reproducible implementation exists in that file.",
        "how_fixed": "Added reproducible script-based many-to-many conversion and checks.",
        "evidence": "File inspection shows 0-byte notebook.",
    },
    {
        "needed_fix": "Some notebook path assumptions are invalid.",
        "why": "Notebook outputs show IO errors for train-*.csv path patterns.",
        "how_fixed": "Path resolution is anchored to project root derived from this script location.",
        "evidence": "Stored notebook errors in members/lisa/notebooks/imdb_duckdb_pipeline.ipynb.",
    },
]


@dataclass
class StepTimer:
    timings: List[Dict[str, float]] = field(default_factory=list)

    @contextmanager
    def track(self, step: str) -> Iterable[None]:
        start = time.perf_counter()
        yield
        seconds = time.perf_counter() - start
        self.timings.append({"step": step, "seconds": round(seconds, 4)})


def get_paths() -> Dict[str, Path]:
    project_root = Path(__file__).resolve().parents[2]
    member_root = Path(__file__).resolve().parent
    output_dir = member_root / "outputs_restart"
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "project_root": project_root,
        "member_root": member_root,
        "output_dir": output_dir,
        "raw_csv_dir": project_root / "data" / "raw" / "csv",
        "raw_json_dir": project_root / "data" / "raw" / "json",
    }


def normalize_missing_tokens(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        series = out[col].astype("string").str.strip()
        series = series.replace(list(MISSING_TOKENS), pd.NA)
        out[col] = series
    return out


def safe_to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def canonicalize_title(title: str) -> str:
    if title is None:
        return ""
    text = str(title).lower().strip()
    text = re.sub(r"\s+", " ", text)
    # Keep alnum + spaces only so "Hunger!" and "Hunger" map together.
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return text.strip()


def build_name_maps(directors: pd.DataFrame, writers: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dir_map = (
        directors.groupby("tconst")["director_id"]
        .apply(lambda s: "|".join(sorted(set(s.astype(str)))))
        .rename("director_ids")
        .reset_index()
    )
    wr_map = (
        writers.groupby("tconst")["writer_id"]
        .apply(lambda s: "|".join(sorted(set(s.astype(str)))))
        .rename("writer_ids")
        .reset_index()
    )
    return dir_map, wr_map


def compute_oof_group_rate(
    keys: pd.Series,
    labels: pd.Series,
    n_splits: int = 5,
    smoothing: float = 20.0,
) -> Tuple[np.ndarray, Dict[str, float], float]:
    y = labels.astype(int).to_numpy()
    key_values = keys.fillna("").astype(str).to_numpy()
    global_mean = float(np.mean(y))
    oof = np.full(len(y), global_mean, dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    def fit_lookup(fit_keys: np.ndarray, fit_y: np.ndarray) -> Dict[str, float]:
        sums: Dict[str, float] = defaultdict(float)
        counts: Dict[str, int] = defaultdict(int)
        for k, lbl in zip(fit_keys, fit_y):
            sums[k] += float(lbl)
            counts[k] += 1
        lookup_local: Dict[str, float] = {}
        for k, cnt in counts.items():
            lookup_local[k] = (sums[k] + smoothing * global_mean) / (cnt + smoothing)
        return lookup_local

    for fit_idx, holdout_idx in skf.split(key_values, y):
        lookup = fit_lookup(key_values[fit_idx], y[fit_idx])
        for i in holdout_idx:
            oof[i] = lookup.get(key_values[i], global_mean)

    full_lookup = fit_lookup(key_values, y)
    return oof, full_lookup, global_mean


def apply_group_rate(keys: pd.Series, lookup: Dict[str, float], global_mean: float) -> np.ndarray:
    return keys.fillna("").astype(str).map(lambda k: lookup.get(k, global_mean)).astype(float).to_numpy()


def build_title_similarity_features_oof(
    train_titles: pd.Series,
    train_labels: pd.Series,
    n_splits: int = 5,
    max_features: int = 800,
) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer, np.ndarray, np.ndarray]:
    titles = train_titles.fillna("").astype(str)
    y = train_labels.astype(int).to_numpy()

    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=max_features)
    X = vectorizer.fit_transform(titles)

    oof_hit = np.zeros(len(titles), dtype=float)
    oof_non_hit = np.zeros(len(titles), dtype=float)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for fit_idx, holdout_idx in skf.split(np.zeros(len(y)), y):
        X_fit = X[fit_idx]
        y_fit = y[fit_idx]
        hit_centroid = np.asarray(X_fit[y_fit == 1].mean(axis=0)).reshape(1, -1)
        non_hit_centroid = np.asarray(X_fit[y_fit == 0].mean(axis=0)).reshape(1, -1)

        X_hold = X[holdout_idx]
        oof_hit[holdout_idx] = cosine_similarity(X_hold, hit_centroid).ravel()
        oof_non_hit[holdout_idx] = cosine_similarity(X_hold, non_hit_centroid).ravel()

    full_hit_centroid = np.asarray(X[y == 1].mean(axis=0)).reshape(1, -1)
    full_non_hit_centroid = np.asarray(X[y == 0].mean(axis=0)).reshape(1, -1)
    return oof_hit, oof_non_hit, vectorizer, full_hit_centroid, full_non_hit_centroid


def apply_title_similarity_features(
    titles: pd.Series,
    vectorizer: TfidfVectorizer,
    hit_centroid: np.ndarray,
    non_hit_centroid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    X = vectorizer.transform(titles.fillna("").astype(str))
    hit_sim = cosine_similarity(X, hit_centroid).ravel()
    non_hit_sim = cosine_similarity(X, non_hit_centroid).ravel()
    return hit_sim, non_hit_sim


def load_train_and_hidden(raw_csv_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_paths = sorted(raw_csv_dir.glob("train-*.csv"))
    if not train_paths:
        raise FileNotFoundError(f"No train-*.csv files found in {raw_csv_dir}")
    train_df = pd.concat([pd.read_csv(p) for p in train_paths], ignore_index=True)
    validation_hidden = pd.read_csv(raw_csv_dir / "validation_hidden.csv")
    test_hidden = pd.read_csv(raw_csv_dir / "test_hidden.csv")
    return train_df, validation_hidden, test_hidden


def build_edges_from_json(raw_json_dir: Path, timer: StepTimer) -> Tuple[pd.DataFrame, pd.DataFrame]:
    directing_json = raw_json_dir / "directing.json"
    writing_json = raw_json_dir / "writing.json"
    if not directing_json.exists() or not writing_json.exists():
        raise FileNotFoundError("Missing directing.json or writing.json in data/raw/json.")

    con = duckdb.connect()

    with timer.track("json_to_many_to_many_duckdb"):
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE directing_edges_raw AS
            WITH src AS (
              SELECT json(movie) AS movie_obj, json(director) AS director_obj
              FROM read_json_auto(?)
            ),
            movies AS (
              SELECT je.key AS k, trim(both '"' from CAST(je.value AS VARCHAR)) AS tconst
              FROM src, json_each(movie_obj) je
            ),
            directors AS (
              SELECT je.key AS k, trim(both '"' from CAST(je.value AS VARCHAR)) AS director_id
              FROM src, json_each(director_obj) je
            )
            SELECT m.tconst, d.director_id
            FROM movies m
            JOIN directors d USING (k)
            """,
            [str(directing_json)],
        )

        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE writing_edges_raw AS
            SELECT CAST(movie AS VARCHAR) AS tconst, CAST(writer AS VARCHAR) AS writer_id
            FROM read_json_auto(?)
            """,
            [str(writing_json)],
        )

        directors = con.execute("SELECT tconst, director_id FROM directing_edges_raw").fetchdf()
        writers = con.execute("SELECT tconst, writer_id FROM writing_edges_raw").fetchdf()

    con.close()
    return directors, writers


def clean_movie_frames(
    train_df: pd.DataFrame,
    validation_hidden: pd.DataFrame,
    test_hidden: pd.DataFrame,
    timer: StepTimer,
) -> pd.DataFrame:
    with timer.track("clean_movies"):
        train = train_df.copy()
        val_h = validation_hidden.copy()
        test_h = test_hidden.copy()

        train["split"] = "train"
        val_h["split"] = "validation_hidden"
        test_h["split"] = "test_hidden"
        val_h["label"] = pd.NA
        test_h["label"] = pd.NA

        merged = pd.concat([train, val_h, test_h], ignore_index=True)
        merged = normalize_missing_tokens(
            merged,
            cols=["primaryTitle", "originalTitle", "startYear", "endYear", "runtimeMinutes", "numVotes"],
        )
        merged = safe_to_numeric(merged, cols=["startYear", "endYear", "runtimeMinutes", "numVotes"])

        if "Unnamed: 0" in merged.columns:
            merged = merged.drop(columns=["Unnamed: 0"])

        merged = merged.drop_duplicates(subset=["split", "tconst"], keep="first")

        if "label" in merged.columns:
            merged["label"] = merged["label"].replace({True: 1, False: 0, "True": 1, "False": 0, "1": 1, "0": 0})
            merged["label"] = pd.to_numeric(merged["label"], errors="coerce")

    return merged


def clean_edges(directors: pd.DataFrame, writers: pd.DataFrame, timer: StepTimer) -> Tuple[pd.DataFrame, pd.DataFrame]:
    with timer.track("clean_edges"):
        d = normalize_missing_tokens(directors, cols=["tconst", "director_id"])
        w = normalize_missing_tokens(writers, cols=["tconst", "writer_id"])

        d = d.dropna(subset=["tconst"]).drop_duplicates()
        w = w.dropna(subset=["tconst"]).drop_duplicates()

        # Unknown IDs are not informative for target encoding/count features.
        d = d.dropna(subset=["director_id"])
        w = w.dropna(subset=["writer_id"])

    return d, w


def add_aggregate_features(movies: pd.DataFrame, directors: pd.DataFrame, writers: pd.DataFrame, timer: StepTimer) -> pd.DataFrame:
    with timer.track("feature_aggregates_many_to_many"):
        d_agg = (
            directors.groupby("tconst")
            .agg(num_directors=("director_id", "size"), num_unique_directors=("director_id", "nunique"))
            .reset_index()
        )
        w_agg = (
            writers.groupby("tconst")
            .agg(num_writers=("writer_id", "size"), num_unique_writers=("writer_id", "nunique"))
            .reset_index()
        )

        feat = movies.merge(d_agg, on="tconst", how="left").merge(w_agg, on="tconst", how="left")

        for col in ["num_directors", "num_unique_directors", "num_writers", "num_unique_writers"]:
            feat[col] = feat[col].fillna(0).astype(float)
        feat["is_auteur"] = ((feat["num_unique_directors"] == 1) & (feat["num_unique_writers"] == 1)).astype(float)

    return feat


def add_base_features(df: pd.DataFrame, timer: StepTimer) -> pd.DataFrame:
    with timer.track("feature_engineering_base"):
        out = df.copy()
        out["primaryTitle"] = out["primaryTitle"].fillna("")
        out["originalTitle"] = out["originalTitle"].fillna("")
        out["canonical_title"] = out["primaryTitle"].map(canonicalize_title)
        out["title_len"] = out["primaryTitle"].astype(str).str.len().astype(float)
        out["title_word_count"] = out["primaryTitle"].astype(str).str.split().str.len().fillna(0).astype(float)
        out["title_has_digit"] = out["primaryTitle"].astype(str).str.contains(r"\d", regex=True).astype(float)
        out["title_has_colon"] = out["primaryTitle"].astype(str).str.contains(":", regex=False).astype(float)
        out["title_has_question"] = out["primaryTitle"].astype(str).str.contains(r"\?", regex=True).astype(float)
        out["title_upper_ratio"] = (
            out["primaryTitle"].astype(str).str.count(r"[A-Z]") / out["title_len"].replace(0, np.nan)
        ).fillna(0.0)
        out["has_original_title"] = out["originalTitle"].astype(str).str.strip().ne("").astype(float)
        out["runtime_missing"] = out["runtimeMinutes"].isna().astype(float)
        out["votes_missing"] = out["numVotes"].isna().astype(float)
        out["start_missing"] = out["startYear"].isna().astype(float)
        out["end_missing"] = out["endYear"].isna().astype(float)

        out["year_span"] = (out["endYear"] - out["startYear"]).where(
            out["startYear"].notna() & out["endYear"].notna(), 0.0
        )
        out["year_span"] = out["year_span"].clip(lower=0)
        out["numVotes_log1p"] = np.log1p(out["numVotes"].clip(lower=0))

    return out


def fit_cap_bounds(train_df: pd.DataFrame, cols: List[str], q_low: float = 0.01, q_high: float = 0.99) -> Dict[str, Tuple[float, float]]:
    bounds: Dict[str, Tuple[float, float]] = {}
    for col in cols:
        values = train_df[col].dropna()
        if values.empty:
            bounds[col] = (0.0, 0.0)
        else:
            bounds[col] = (float(values.quantile(q_low)), float(values.quantile(q_high)))
    return bounds


def apply_caps(df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    out = df.copy()
    for col, (lo, hi) in bounds.items():
        if col in out.columns:
            numeric_col = pd.to_numeric(out[col], errors="coerce").astype(float)
            out[f"{col}_capped"] = numeric_col.clip(lower=lo, upper=hi)
    return out


def build_entity_index(edges: pd.DataFrame, entity_col: str) -> Dict[str, List[str]]:
    grouped = edges.groupby("tconst")[entity_col].apply(lambda s: sorted(set(s.dropna().astype(str))))
    return grouped.to_dict()


def fit_entity_rate_lookup(
    tconsts: Iterable[str],
    labels: Iterable[int],
    entity_index: Dict[str, List[str]],
    global_mean: float,
    smoothing: float,
) -> Dict[str, float]:
    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)

    for tconst, label in zip(tconsts, labels):
        entities = entity_index.get(str(tconst), [])
        for entity in entities:
            sums[entity] += float(label)
            counts[entity] += 1

    lookup: Dict[str, float] = {}
    for entity, cnt in counts.items():
        lookup[entity] = (sums[entity] + smoothing * global_mean) / (cnt + smoothing)
    return lookup


def movie_entity_score(
    tconst: str,
    entity_index: Dict[str, List[str]],
    lookup: Dict[str, float],
    global_mean: float,
) -> float:
    entities = entity_index.get(str(tconst), [])
    if not entities:
        return float(global_mean)
    rates = [lookup.get(entity, global_mean) for entity in entities]
    return float(np.mean(rates)) if rates else float(global_mean)


def compute_oof_target_encoding(
    train_df: pd.DataFrame,
    entity_index: Dict[str, List[str]],
    n_splits: int = 5,
    smoothing: float = 20.0,
) -> Tuple[np.ndarray, Dict[str, float], float]:
    y = train_df["label"].astype(int).to_numpy()
    tconsts = train_df["tconst"].astype(str).to_numpy()
    global_mean = float(np.mean(y))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.full(len(train_df), global_mean, dtype=float)

    for fit_idx, holdout_idx in skf.split(tconsts, y):
        fold_lookup = fit_entity_rate_lookup(
            tconsts[fit_idx],
            y[fit_idx],
            entity_index=entity_index,
            global_mean=global_mean,
            smoothing=smoothing,
        )
        for i in holdout_idx:
            oof[i] = movie_entity_score(tconsts[i], entity_index, fold_lookup, global_mean)

    full_lookup = fit_entity_rate_lookup(
        tconsts,
        y,
        entity_index=entity_index,
        global_mean=global_mean,
        smoothing=smoothing,
    )
    return oof, full_lookup, global_mean


def apply_target_encoding(
    df: pd.DataFrame,
    entity_index: Dict[str, List[str]],
    lookup: Dict[str, float],
    global_mean: float,
) -> np.ndarray:
    tconsts = df["tconst"].astype(str).to_numpy()
    vals = [movie_entity_score(tconst, entity_index, lookup, global_mean) for tconst in tconsts]
    return np.array(vals, dtype=float)


def compute_psi(train_values: pd.Series, val_values: pd.Series, n_bins: int = 10) -> float:
    train = pd.to_numeric(train_values, errors="coerce").dropna()
    val = pd.to_numeric(val_values, errors="coerce").dropna()
    if train.empty or val.empty:
        return float("nan")

    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.unique(np.quantile(train, quantiles))
    if len(bin_edges) < 3:
        return float("nan")

    train_bins = pd.cut(train, bins=bin_edges, include_lowest=True)
    val_bins = pd.cut(val, bins=bin_edges, include_lowest=True)
    train_dist = train_bins.value_counts(normalize=True).sort_index()
    val_dist = val_bins.value_counts(normalize=True).sort_index()

    aligned = pd.concat([train_dist, val_dist], axis=1).fillna(1e-6)
    aligned.columns = ["train_pct", "val_pct"]
    aligned["train_pct"] = aligned["train_pct"].clip(lower=1e-6)
    aligned["val_pct"] = aligned["val_pct"].clip(lower=1e-6)
    psi = ((aligned["val_pct"] - aligned["train_pct"]) * np.log(aligned["val_pct"] / aligned["train_pct"])).sum()
    return float(psi)


def compute_feature_goodness(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    val_X: pd.DataFrame,
    val_y: pd.Series,
    feature_cols: List[str],
) -> pd.DataFrame:
    y_train = train_y.astype(int).to_numpy()
    y_val = val_y.astype(int).to_numpy()
    rows: List[Dict[str, float]] = []

    for feat in feature_cols:
        x_train = pd.to_numeric(train_X[feat], errors="coerce")
        x_val = pd.to_numeric(val_X[feat], errors="coerce")
        median_val = float(x_train.median()) if not x_train.dropna().empty else 0.0
        x_train_fill = x_train.fillna(median_val)
        x_val_fill = x_val.fillna(median_val)

        auc_train = float("nan")
        auc_val = float("nan")
        if x_train_fill.nunique() > 1 and len(np.unique(y_train)) == 2:
            auc_train = float(roc_auc_score(y_train, x_train_fill))
            auc_train = max(auc_train, 1.0 - auc_train)
        if x_val_fill.nunique() > 1 and len(np.unique(y_val)) == 2:
            auc_val = float(roc_auc_score(y_val, x_val_fill))
            auc_val = max(auc_val, 1.0 - auc_val)

        mi = float(
            mutual_info_classif(
                x_train_fill.to_numpy().reshape(-1, 1),
                y_train,
                discrete_features=False,
                random_state=SEED,
            )[0]
        )

        if x_train_fill.nunique() > 1:
            spear_train = float(pd.Series(x_train_fill).corr(pd.Series(y_train), method="spearman"))
        else:
            spear_train = float("nan")
        if x_val_fill.nunique() > 1:
            spear_val = float(pd.Series(x_val_fill).corr(pd.Series(y_val), method="spearman"))
        else:
            spear_val = float("nan")
        psi = compute_psi(x_train, x_val)

        rows.append(
            {
                "feature": feat,
                "missing_rate_train": float(x_train.isna().mean()),
                "missing_rate_val": float(x_val.isna().mean()),
                "std_train": float(x_train_fill.std(ddof=0)),
                "univariate_auc_train": auc_train,
                "univariate_auc_val": auc_val,
                "mutual_info_train": mi,
                "spearman_train": spear_train,
                "spearman_val": spear_val,
                "abs_spearman_val": abs(spear_val) if not np.isnan(spear_val) else float("nan"),
                "psi_train_vs_val": psi,
            }
        )

    df = pd.DataFrame(rows)
    auc_rank = df["univariate_auc_val"].fillna(0.5).rank(pct=True)
    mi_rank = df["mutual_info_train"].fillna(0).rank(pct=True)
    spearman_rank = df["abs_spearman_val"].fillna(0).rank(pct=True)
    psi_base = df["psi_train_vs_val"].max(skipna=True)
    if pd.isna(psi_base):
        psi_base = 1.0
    psi_rank = (-df["psi_train_vs_val"].fillna(psi_base)).rank(pct=True)
    missing_rank = (-df["missing_rate_train"].fillna(1.0)).rank(pct=True)

    df["goodness_score"] = (
        0.35 * auc_rank
        + 0.25 * mi_rank
        + 0.20 * spearman_rank
        + 0.10 * psi_rank
        + 0.10 * missing_rank
    )
    df = df.sort_values("goodness_score", ascending=False).reset_index(drop=True)
    return df


def predict_probs_for_model(
    model_name: str,
    model_obj,
    X_df: pd.DataFrame,
    scaler: StandardScaler | None = None,
) -> np.ndarray:
    if model_name == "xgboost":
        return model_obj.predict_proba(X_df)[:, 1]
    if scaler is None:
        raise ValueError("Scaler is required for logistic predictions.")
    X_scaled = scaler.transform(X_df)
    return model_obj.predict_proba(X_scaled)[:, 1]


def compute_permutation_auc_drop(
    model_name: str,
    model_obj,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_cols: List[str],
    scaler: StandardScaler | None = None,
) -> pd.DataFrame:
    y = y_val.astype(int).to_numpy()
    base_probs = predict_probs_for_model(model_name, model_obj, X_val[feature_cols], scaler=scaler)
    baseline_auc = float(roc_auc_score(y, base_probs))

    rng = np.random.default_rng(SEED)
    rows: List[Dict[str, float]] = []
    for feat in feature_cols:
        shuffled = X_val[feature_cols].copy()
        shuffled[feat] = rng.permutation(shuffled[feat].to_numpy())
        perm_probs = predict_probs_for_model(model_name, model_obj, shuffled, scaler=scaler)
        perm_auc = float(roc_auc_score(y, perm_probs))
        rows.append(
            {
                "feature": feat,
                "baseline_auc": baseline_auc,
                "permuted_auc": perm_auc,
                "perm_auc_drop": baseline_auc - perm_auc,
            }
        )
    return pd.DataFrame(rows).sort_values("perm_auc_drop", ascending=False).reset_index(drop=True)


def compute_xgb_shap_summary(
    xgb_model: XGBClassifier | None,
    X_val: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    if xgb_model is None:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])
    import xgboost as xgb  # local import to keep optional dependency behavior

    booster = xgb_model.get_booster()
    dval = xgb.DMatrix(X_val[feature_cols], feature_names=feature_cols)
    contrib = booster.predict(dval, pred_contribs=True)
    # Last column is bias term.
    mean_abs = np.abs(contrib[:, :-1]).mean(axis=0)
    out = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
    return out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def classify_feature_status(feature_diag_df: pd.DataFrame) -> pd.DataFrame:
    df = feature_diag_df.copy()
    df["xgb_gain"] = df["xgb_gain"].fillna(0.0)
    df["perm_auc_drop"] = df["perm_auc_drop"].fillna(0.0)

    good_rank = df["goodness_score"].rank(pct=True)
    perm_rank = df["perm_auc_drop"].clip(lower=0).rank(pct=True)
    gain_rank = df["xgb_gain"].rank(pct=True)
    df["diagnostic_score"] = 0.50 * good_rank + 0.35 * perm_rank + 0.15 * gain_rank

    median_good = float(df["goodness_score"].median())
    df["status"] = "review"
    drop_mask = (df["perm_auc_drop"] <= 0) & (df["goodness_score"] < median_good)
    keep_mask = (df["diagnostic_score"] >= 0.60) | (df["perm_auc_drop"] >= 0.002)
    df.loc[drop_mask, "status"] = "drop_candidate"
    df.loc[keep_mask, "status"] = "keep"
    return df.sort_values(["status", "diagnostic_score"], ascending=[True, False]).reset_index(drop=True)


def build_comparison_metrics(project_root: Path) -> Dict[str, int]:
    raw_csv_dir = project_root / "data" / "raw" / "csv"
    raw_json_dir = project_root / "data" / "raw" / "json"
    directors_csv = pd.read_csv(raw_csv_dir / "movie_directors.csv")
    writers_csv = pd.read_csv(raw_csv_dir / "movie_writers.csv")

    with open(raw_json_dir / "directing.json", "r", encoding="utf-8") as f:
        directing_raw = json.load(f)
    directing_pairs = list(zip(directing_raw["movie"].values(), directing_raw["director"].values()))
    directing_json_df = pd.DataFrame(directing_pairs, columns=["tconst", "director_id"])

    with open(raw_json_dir / "writing.json", "r", encoding="utf-8") as f:
        writing_raw = json.load(f)
    writing_json_df = pd.DataFrame(writing_raw).rename(columns={"movie": "tconst", "writer": "writer_id"})

    dir_json_set = set(map(tuple, directing_json_df[["tconst", "director_id"]].astype(str).to_records(index=False)))
    dir_csv_set = set(map(tuple, directors_csv.rename(columns={"director": "director_id"})[["tconst", "director_id"]].astype(str).to_records(index=False)))

    wr_json_set = set(map(tuple, writing_json_df[["tconst", "writer_id"]].astype(str).to_records(index=False)))
    wr_csv_set = set(map(tuple, writers_csv.rename(columns={"writer": "writer_id"})[["tconst", "writer_id"]].astype(str).to_records(index=False)))

    many_to_many_nb = project_root / "members" / "vanshita" / "json_to_csv" / "many_to_many.ipynb"
    many_to_many_size = many_to_many_nb.stat().st_size if many_to_many_nb.exists() else -1

    dir_json_unknown_lens = sorted(
        {len(str(x)) for x in directing_json_df["director_id"].dropna().astype(str) if "N" in str(x)}
    )
    dir_csv_unknown_lens = sorted(
        {len(str(x)) for x in directors_csv["director"].dropna().astype(str) if "N" in str(x)}
    )

    return {
        "directing_pairs_json": len(dir_json_set),
        "directing_pairs_csv": len(dir_csv_set),
        "directing_missing_in_csv": len(dir_json_set - dir_csv_set),
        "directing_extra_in_csv": len(dir_csv_set - dir_json_set),
        "directing_json_unknown_token_char_len": dir_json_unknown_lens[0] if dir_json_unknown_lens else -1,
        "directing_csv_unknown_token_char_len": dir_csv_unknown_lens[0] if dir_csv_unknown_lens else -1,
        "writing_pairs_json": len(wr_json_set),
        "writing_pairs_csv": len(wr_csv_set),
        "writing_missing_in_csv": len(wr_json_set - wr_csv_set),
        "writing_extra_in_csv": len(wr_csv_set - wr_json_set),
        "many_to_many_notebook_bytes": many_to_many_size,
    }


def write_comparison_report(
    report_path: Path,
    issue_fix_log: List[Dict[str, str]],
    metrics: Dict[str, float],
    model_metrics: Dict[str, float],
    timing_path: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Djamel Fix Comparison Report")
    lines.append("")
    lines.append("This report compares what needed fixing, why, and how it was fixed in `imdb_pipeline_audit_and_fix.py`.")
    lines.append("")
    lines.append("## Needed Fix -> Why -> How Fixed")
    lines.append("")
    lines.append("| Needed Fix | Why It Needed Fixing | How I Fixed It | Evidence |")
    lines.append("|---|---|---|---|")
    for row in issue_fix_log:
        lines.append(
            f"| {row['needed_fix']} | {row['why']} | {row['how_fixed']} | {row['evidence']} |"
        )

    lines.append("")
    lines.append("## JSON-to-Many-to-Many Comparison Metrics")
    lines.append("")
    for key, value in metrics.items():
        lines.append(f"- `{key}`: {value}")

    lines.append("")
    lines.append("## Validation (Internal Split)")
    lines.append("")
    for key, value in model_metrics.items():
        lines.append(f"- `{key}`: {value:.6f}" if isinstance(value, (float, np.floating)) else f"- `{key}`: {value}")

    lines.append("")
    lines.append(f"Step timings were written to: `{timing_path}`")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def count_missing_tokens(series: pd.Series) -> int:
    normalized = series.astype("string").str.strip()
    return int(normalized.isin(list(MISSING_TOKENS)).sum())


def build_cleaning_summary(
    train_df: pd.DataFrame,
    validation_hidden: pd.DataFrame,
    test_hidden: pd.DataFrame,
    movies_clean: pd.DataFrame,
    directors_raw: pd.DataFrame,
    writers_raw: pd.DataFrame,
    directors_clean: pd.DataFrame,
    writers_clean: pd.DataFrame,
    features: pd.DataFrame,
    train_part: pd.DataFrame,
    title_stats: pd.DataFrame,
    cap_bounds: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:
    raw_movies = pd.concat(
        [
            train_df.assign(split="train"),
            validation_hidden.assign(split="validation_hidden"),
            test_hidden.assign(split="test_hidden"),
        ],
        ignore_index=True,
    )
    raw_movies_missing_hits = 0
    for col in ["primaryTitle", "originalTitle", "startYear", "endYear", "runtimeMinutes", "numVotes"]:
        if col in raw_movies.columns:
            raw_movies_missing_hits += count_missing_tokens(raw_movies[col])

    raw_movie_dups = int(raw_movies.duplicated(subset=["split", "tconst"]).sum())

    dir_unknown_raw = count_missing_tokens(directors_raw["director_id"])
    wr_unknown_raw = count_missing_tokens(writers_raw["writer_id"])
    dir_unknown_removed = int(len(directors_raw) - len(directors_clean))
    wr_unknown_removed = int(len(writers_raw) - len(writers_clean))

    train_features = features[features["split"] == "train"].copy()
    canonical_variants = (
        train_features.groupby("canonical_title")["primaryTitle"].nunique()
        if not train_features.empty
        else pd.Series(dtype=float)
    )
    canonical_variant_groups = int((canonical_variants > 1).sum()) if not canonical_variants.empty else 0

    summary: Dict[str, float] = {
        "raw_movie_rows_total": int(len(raw_movies)),
        "clean_movie_rows_total": int(len(movies_clean)),
        "movie_duplicate_keys_removed": raw_movie_dups,
        "raw_missing_token_hits_in_movie_fields": int(raw_movies_missing_hits),
        "raw_director_edges": int(len(directors_raw)),
        "clean_director_edges": int(len(directors_clean)),
        "raw_writer_edges": int(len(writers_raw)),
        "clean_writer_edges": int(len(writers_clean)),
        "raw_director_unknown_token_hits": int(dir_unknown_raw),
        "raw_writer_unknown_token_hits": int(wr_unknown_raw),
        "director_rows_removed_in_cleaning": int(dir_unknown_removed),
        "writer_rows_removed_in_cleaning": int(wr_unknown_removed),
        "canonical_title_variant_groups_train": int(canonical_variant_groups),
        "title_conflicting_year_groups_train": int(title_stats["title_conflicting_years"].sum()),
        "title_rows_with_conflicting_year_flag_train": int(train_part["title_conflicting_years"].sum()),
        "cap_runtimeMinutes_low": float(cap_bounds["runtimeMinutes"][0]),
        "cap_runtimeMinutes_high": float(cap_bounds["runtimeMinutes"][1]),
        "cap_numVotes_log1p_low": float(cap_bounds["numVotes_log1p"][0]),
        "cap_numVotes_log1p_high": float(cap_bounds["numVotes_log1p"][1]),
    }
    return summary


def write_cleaning_pipeline_report(
    report_path: Path,
    summary: Dict[str, float],
    conversion_metrics: Dict[str, int],
    timings_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    timings_map = (
        timings_df.set_index("step")["seconds"].to_dict()
        if not timings_df.empty and "step" in timings_df and "seconds" in timings_df
        else {}
    )

    def sec(step: str) -> str:
        val = timings_map.get(step)
        if val is None:
            return "n/a"
        return f"{float(val):.4f}s"

    lines: List[str] = []
    lines.append("# Data Cleaning Pipeline (Exact Steps, Why, Evidence)")
    lines.append("")
    lines.append("This is the explicit cleaning pipeline used in `imdb_pipeline_audit_and_fix.py`.")
    lines.append("")
    lines.append("## Cleaning Pipeline Table")
    lines.append("")
    lines.append("| Step | What We Did | Why It Was Necessary | Evidence From This Run |")
    lines.append("|---|---|---|---|")
    lines.append(
        "| 1) Normalize movie rows (`clean_movies`) | Unified train/validation/test schema, normalized missing tokens (`\\N`, `\\\\N`, empty/null variants), coerced numeric columns, normalized labels, dropped duplicate (`split`,`tconst`) keys. | Prevents inconsistent null handling and duplicate leakage across splits. | "
        f"raw rows={summary['raw_movie_rows_total']}, cleaned rows={summary['clean_movie_rows_total']}, duplicates removed={summary['movie_duplicate_keys_removed']}, raw missing-token hits={summary['raw_missing_token_hits_in_movie_fields']}, time={sec('clean_movies')} |"
    )
    lines.append(
        "| 2) JSON -> many-to-many with DuckDB (`json_to_many_to_many_duckdb`) | Rebuilt `directing.json` and `writing.json` as edge tables using DuckDB JSON parsing (`json_each`) instead of brittle CSV assumptions. | Preserves relational correctness and prevents fan-out bugs from malformed joins. | "
        f"director edges raw={summary['raw_director_edges']}, writer edges raw={summary['raw_writer_edges']}, time={sec('json_to_many_to_many_duckdb')} |"
    )
    lines.append(
        "| 3) Clean relation edges (`clean_edges`) | Deduplicated edges, removed null `tconst`, removed unknown person IDs before aggregation/encoding. | Unknown IDs add noise and can corrupt entity-based rates. | "
        f"director rows removed={summary['director_rows_removed_in_cleaning']} (unknown hits raw={summary['raw_director_unknown_token_hits']}), "
        f"writer rows removed={summary['writer_rows_removed_in_cleaning']} (unknown hits raw={summary['raw_writer_unknown_token_hits']}), time={sec('clean_edges')} |"
    )
    lines.append(
        "| 4) Canonicalize title + conflict signals (`feature_engineering_base`, `title_group_conflict_features`) | Built canonical title text, then flagged title groups with conflicting years based only on train. | Novel cleaning feature: keeps ambiguous duplicates explicit rather than silently dropping them. | "
        f"canonical variant groups (train)={summary['canonical_title_variant_groups_train']}, conflicting-year groups={summary['title_conflicting_year_groups_train']}, rows flagged={summary['title_rows_with_conflicting_year_flag_train']}, time={sec('title_group_conflict_features')} |"
    )
    lines.append(
        "| 5) Train-only outlier capping (`outlier_capping_fit_apply`) | Fit caps on train quantiles and applied to all splits. | Prevents extreme values from dominating while avoiding leakage from validation/test distributions. | "
        f"runtime cap=[{summary['cap_runtimeMinutes_low']:.3f}, {summary['cap_runtimeMinutes_high']:.3f}], "
        f"log(numVotes) cap=[{summary['cap_numVotes_log1p_low']:.3f}, {summary['cap_numVotes_log1p_high']:.3f}], time={sec('outlier_capping_fit_apply')} |"
    )
    lines.append(
        "| 6) Leakage-safe entity encodings (`target_encoding_oof`) | Built OOF target encodings for directors/writers and canonical titles on train, then applied mappings to holdout/test. | Keeps high-signal relational features while preventing target leakage. | "
        f"time={sec('target_encoding_oof')} |"
    )

    lines.append("")
    lines.append("## Novel / Defensible Cleaning Decisions")
    lines.append("")
    lines.append("- Explicitly normalized both `\\N` and `\\\\N` unknown markers (double-escaped tokens were present).")
    lines.append("- Used DuckDB JSON parsing for many-to-many reconstruction instead of trusting preconverted CSV artifacts.")
    lines.append("- Kept ambiguous duplicate-like titles as modeled uncertainty (`title_conflicting_years`) instead of hard dropping.")
    lines.append("- Used train-only statistics for capping and OOF encodings to avoid leakage.")
    lines.append("")
    lines.append("## JSON Conversion Consistency Checks")
    lines.append("")
    for key, value in conversion_metrics.items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("## Engine Usage (DuckDB vs Spark)")
    lines.append("")
    lines.append("- `DuckDB`: used for JSON -> many-to-many conversion (`json_to_many_to_many_duckdb`).")
    lines.append("- `Spark`: not used in this current pipeline implementation.")
    lines.append("")
    lines.append("| Pipeline Step | Engine | Why This Engine | Runtime |")
    lines.append("|---|---|---|---|")
    for _, row in timings_df.iterrows():
        step = str(row["step"])
        runtime = f"{float(row['seconds']):.4f}s"
        engine, why_engine = STEP_ENGINE_USAGE.get(step, ("Unspecified", "No explicit mapping set."))
        lines.append(f"| `{step}` | {engine} | {why_engine} | {runtime} |")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Cleaning report: `{report_path}`")
    lines.append(f"- Cleaning metrics CSV: `{output_dir / 'cleaning_audit_metrics.csv'}`")
    lines.append(f"- Step timings: `{output_dir / 'step_timings.csv'}`")
    lines.append("")
    lines.append("## Note")
    lines.append("")
    lines.append("This cleaning pipeline is executed before model fitting; feature diagnostics and model comparison are downstream.")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def generate_styled_visuals_and_tables(
    output_dir: Path,
    timings_df: pd.DataFrame,
    model_metrics: Dict[str, float],
    comparison_metrics: Dict[str, float],
    feature_goodness_df: pd.DataFrame | None = None,
    feature_diagnostics_df: pd.DataFrame | None = None,
    cleaning_missing_df: pd.DataFrame | None = None,
    edge_clean_df: pd.DataFrame | None = None,
    cap_impact_df: pd.DataFrame | None = None,
    impute_impact_df: pd.DataFrame | None = None,
    datatype_audit_df: pd.DataFrame | None = None,
    duplicates_df: pd.DataFrame | None = None,
    disguised_tokens_df: pd.DataFrame | None = None,
    distribution_stats_df: pd.DataFrame | None = None,
) -> None:
    imdb_yellow = "#F5C518"
    accent_blue = "#1848f5"
    bg_dark = "#000000"
    card_dark = "#111111"
    text_light = "#FFFFFF"
    muted = "#CFCFCF"

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: step timings
    plot_df = timings_df.copy().sort_values("seconds", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg_dark)
    ax.set_facecolor(card_dark)
    ax.barh(plot_df["step"], plot_df["seconds"], color=imdb_yellow)
    ax.set_title("Pipeline Step Timings", color=text_light, fontsize=14, fontweight="bold")
    ax.set_xlabel("Seconds", color=text_light)
    ax.tick_params(axis="x", colors=text_light)
    ax.tick_params(axis="y", colors=text_light, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(muted)
    ax.grid(axis="x", alpha=0.2, color=accent_blue)
    fig.tight_layout()
    fig.savefig(fig_dir / "styled_step_timings.png", dpi=150, facecolor=bg_dark)
    plt.close(fig)

    # Figure 1b: missing token normalization evidence.
    if cleaning_missing_df is not None and not cleaning_missing_df.empty:
        miss_plot = cleaning_missing_df.copy()
        miss_plot = miss_plot.sort_values("raw_missing_tokens", ascending=True)
        y = np.arange(len(miss_plot))
        h = 0.38
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg_dark)
        ax.set_facecolor(card_dark)
        ax.barh(y - h / 2, miss_plot["raw_missing_tokens"], height=h, color=imdb_yellow, label="raw missing tokens")
        ax.barh(y + h / 2, miss_plot["clean_missing_na"], height=h, color=accent_blue, label="clean missing NA")
        ax.set_yticks(y)
        ax.set_yticklabels(miss_plot["field"])
        ax.set_title("Missing Tokens Before vs After Cleaning", color=text_light, fontsize=14, fontweight="bold")
        ax.set_xlabel("Count", color=text_light)
        ax.tick_params(axis="x", colors=text_light)
        ax.tick_params(axis="y", colors=text_light, labelsize=10)
        for spine in ax.spines.values():
            spine.set_color(muted)
        ax.grid(axis="x", alpha=0.2, color=accent_blue)
        leg = ax.legend(facecolor=card_dark, edgecolor=muted)
        for t in leg.get_texts():
            t.set_color(text_light)
        fig.tight_layout()
        fig.savefig(fig_dir / "styled_cleaning_missing_before_after.png", dpi=150, facecolor=bg_dark)
        plt.close(fig)

    # Figure 1c: edge cleaning impact (directors/writers).
    if edge_clean_df is not None and not edge_clean_df.empty:
        plot_df = edge_clean_df.copy()
        x = np.arange(len(plot_df))
        w = 0.36
        fig, ax = plt.subplots(figsize=(7, 4.6), facecolor=bg_dark)
        ax.set_facecolor(card_dark)
        ax.bar(x - w / 2, plot_df["raw_edges"], width=w, color=imdb_yellow, label="raw edges")
        ax.bar(x + w / 2, plot_df["clean_edges"], width=w, color=accent_blue, label="clean edges")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["entity"])
        ax.set_title("Edge Cleaning Impact (JSON Many-to-Many)", color=text_light, fontsize=14, fontweight="bold")
        ax.set_ylabel("Rows", color=text_light)
        ax.tick_params(axis="x", colors=text_light)
        ax.tick_params(axis="y", colors=text_light)
        for spine in ax.spines.values():
            spine.set_color(muted)
        ax.grid(axis="y", alpha=0.2, color=accent_blue)
        leg = ax.legend(facecolor=card_dark, edgecolor=muted)
        for t in leg.get_texts():
            t.set_color(text_light)
        fig.tight_layout()
        fig.savefig(fig_dir / "styled_edge_cleaning_impact.png", dpi=150, facecolor=bg_dark)
        plt.close(fig)

    # Figure 1d: train-only outlier capping impact.
    if cap_impact_df is not None and not cap_impact_df.empty:
        plot_df = cap_impact_df.copy().sort_values("rows_clipped", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=bg_dark)
        ax.set_facecolor(card_dark)
        ax.barh(plot_df["feature"], plot_df["rows_clipped"], color=imdb_yellow)
        ax.set_title("Outlier Capping Impact (Train Fit Only)", color=text_light, fontsize=14, fontweight="bold")
        ax.set_xlabel("Rows Clipped", color=text_light)
        ax.tick_params(axis="x", colors=text_light)
        ax.tick_params(axis="y", colors=text_light)
        for spine in ax.spines.values():
            spine.set_color(muted)
        ax.grid(axis="x", alpha=0.2, color=accent_blue)
        for i, row in plot_df.reset_index(drop=True).iterrows():
            ax.text(
                float(row["rows_clipped"]) + 0.5,
                i,
                f"{float(row['clip_rate']) * 100:.1f}%",
                color=text_light,
                va="center",
                fontsize=9,
            )
        fig.tight_layout()
        fig.savefig(fig_dir / "styled_capping_impact.png", dpi=150, facecolor=bg_dark)
        plt.close(fig)

    # Figure 1e: imputation effect on training matrix.
    if impute_impact_df is not None and not impute_impact_df.empty:
        plot_df = impute_impact_df.copy()
        plot_df = plot_df[plot_df["missing_before"] > 0].sort_values("missing_before", ascending=False).head(15)
        if not plot_df.empty:
            plot_df = plot_df.sort_values("missing_before", ascending=True)
            y = np.arange(len(plot_df))
            h = 0.38
            fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg_dark)
            ax.set_facecolor(card_dark)
            ax.barh(y - h / 2, plot_df["missing_before"], height=h, color=imdb_yellow, label="before impute")
            ax.barh(y + h / 2, plot_df["missing_after"], height=h, color=accent_blue, label="after impute")
            ax.set_yticks(y)
            ax.set_yticklabels(plot_df["feature"])
            ax.set_title("Imputation Effect (Train Features)", color=text_light, fontsize=14, fontweight="bold")
            ax.set_xlabel("Missing Values Count", color=text_light)
            ax.tick_params(axis="x", colors=text_light)
            ax.tick_params(axis="y", colors=text_light, labelsize=9)
            for spine in ax.spines.values():
                spine.set_color(muted)
            ax.grid(axis="x", alpha=0.2, color=accent_blue)
            leg = ax.legend(facecolor=card_dark, edgecolor=muted)
            for t in leg.get_texts():
                t.set_color(text_light)
            fig.tight_layout()
            fig.savefig(fig_dir / "styled_imputation_impact.png", dpi=150, facecolor=bg_dark)
            plt.close(fig)

    # Figure 1f: JSON consistency evidence for DuckDB conversion.
    json_check = pd.DataFrame(
        [
            ("directing_missing_in_csv", float(comparison_metrics.get("directing_missing_in_csv", 0))),
            ("directing_extra_in_csv", float(comparison_metrics.get("directing_extra_in_csv", 0))),
            ("writing_missing_in_csv", float(comparison_metrics.get("writing_missing_in_csv", 0))),
            ("writing_extra_in_csv", float(comparison_metrics.get("writing_extra_in_csv", 0))),
        ],
        columns=["metric", "count"],
    )
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=bg_dark)
    ax.set_facecolor(card_dark)
    ax.bar(json_check["metric"], json_check["count"], color=[imdb_yellow, accent_blue, imdb_yellow, accent_blue])
    ax.set_title("JSON vs CSV Consistency Checks", color=text_light, fontsize=14, fontweight="bold")
    ax.set_ylabel("Pair Count Difference", color=text_light)
    ax.tick_params(axis="x", colors=text_light, rotation=20)
    ax.tick_params(axis="y", colors=text_light)
    for spine in ax.spines.values():
        spine.set_color(muted)
    ax.grid(axis="y", alpha=0.2, color=accent_blue)
    fig.tight_layout()
    fig.savefig(fig_dir / "styled_json_consistency.png", dpi=150, facecolor=bg_dark)
    plt.close(fig)

    # Figure 1g: datatype parse/coercion errors.
    if datatype_audit_df is not None and not datatype_audit_df.empty:
        dt_plot = datatype_audit_df.copy()
        dt_plot["parse_error_count"] = pd.to_numeric(dt_plot["parse_error_count"], errors="coerce").fillna(0)
        dt_plot = dt_plot[dt_plot["parse_error_count"] > 0].sort_values("parse_error_count", ascending=False)
        if not dt_plot.empty:
            fig, ax = plt.subplots(figsize=(9, 5), facecolor=bg_dark)
            ax.set_facecolor(card_dark)
            ax.bar(dt_plot["column"], dt_plot["parse_error_count"], color=accent_blue)
            ax.set_title("Datatype Parse Errors Before Cleaning", color=text_light, fontsize=14, fontweight="bold")
            ax.set_ylabel("Rows Failing Numeric Parse", color=text_light)
            ax.tick_params(axis="x", colors=text_light, rotation=20)
            ax.tick_params(axis="y", colors=text_light)
            for spine in ax.spines.values():
                spine.set_color(muted)
            ax.grid(axis="y", alpha=0.2, color=imdb_yellow)
            fig.tight_layout()
            fig.savefig(fig_dir / "styled_datatype_parse_errors.png", dpi=150, facecolor=bg_dark)
            plt.close(fig)

    # Figure 1h: duplicate keys by split (before vs after cleaning).
    if duplicates_df is not None and not duplicates_df.empty:
        dup_plot = duplicates_df.copy().sort_values("split")
        x = np.arange(len(dup_plot))
        w = 0.36
        fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=bg_dark)
        ax.set_facecolor(card_dark)
        ax.bar(x - w / 2, dup_plot["duplicate_tconst_before"], width=w, color=imdb_yellow, label="before")
        ax.bar(x + w / 2, dup_plot["duplicate_tconst_after"], width=w, color=accent_blue, label="after")
        ax.set_xticks(x)
        ax.set_xticklabels(dup_plot["split"])
        ax.set_title("Duplicate tconst Keys by Split", color=text_light, fontsize=14, fontweight="bold")
        ax.set_ylabel("Duplicate Count", color=text_light)
        ax.tick_params(axis="x", colors=text_light)
        ax.tick_params(axis="y", colors=text_light)
        for spine in ax.spines.values():
            spine.set_color(muted)
        ax.grid(axis="y", alpha=0.2, color=accent_blue)
        leg = ax.legend(facecolor=card_dark, edgecolor=muted)
        for t in leg.get_texts():
            t.set_color(text_light)
        fig.tight_layout()
        fig.savefig(fig_dir / "styled_duplicates_before_after.png", dpi=150, facecolor=bg_dark)
        plt.close(fig)

    # Figure 1i: disguised token counts per column.
    if disguised_tokens_df is not None and not disguised_tokens_df.empty:
        dis_plot = disguised_tokens_df.copy().sort_values("disguised_missing_tokens", ascending=False)
        fig, ax = plt.subplots(figsize=(9, 5), facecolor=bg_dark)
        ax.set_facecolor(card_dark)
        ax.bar(dis_plot["column"], dis_plot["disguised_missing_tokens"], color=imdb_yellow)
        ax.set_title("Disguised Missing Tokens by Column", color=text_light, fontsize=14, fontweight="bold")
        ax.set_ylabel("Token Count", color=text_light)
        ax.tick_params(axis="x", colors=text_light, rotation=20)
        ax.tick_params(axis="y", colors=text_light)
        for spine in ax.spines.values():
            spine.set_color(muted)
        ax.grid(axis="y", alpha=0.2, color=accent_blue)
        fig.tight_layout()
        fig.savefig(fig_dir / "styled_disguised_tokens.png", dpi=150, facecolor=bg_dark)
        plt.close(fig)

    # Figure 2: model AUC comparison (available models only)
    auc_rows = []
    if "logistic_validation_roc_auc" in model_metrics and model_metrics["logistic_validation_roc_auc"] >= 0:
        auc_rows.append(("Logistic", float(model_metrics["logistic_validation_roc_auc"])))
    if "xgb_validation_roc_auc" in model_metrics and model_metrics["xgb_validation_roc_auc"] >= 0:
        auc_rows.append(("XGBoost", float(model_metrics["xgb_validation_roc_auc"])))

    if auc_rows:
        auc_df = pd.DataFrame(auc_rows, columns=["model", "roc_auc"])
        fig, ax = plt.subplots(figsize=(6, 4), facecolor=bg_dark)
        ax.set_facecolor(card_dark)
        bar_colors = [imdb_yellow if m == "Logistic" else accent_blue for m in auc_df["model"].tolist()]
        ax.bar(auc_df["model"], auc_df["roc_auc"], color=bar_colors)
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Validation ROC-AUC", color=text_light, fontsize=13, fontweight="bold")
        ax.set_ylabel("ROC-AUC", color=text_light)
        ax.tick_params(axis="x", colors=text_light)
        ax.tick_params(axis="y", colors=text_light)
        for spine in ax.spines.values():
            spine.set_color(muted)
        ax.grid(axis="y", alpha=0.2, color=accent_blue)
        fig.tight_layout()
        fig.savefig(fig_dir / "styled_model_auc.png", dpi=150, facecolor=bg_dark)
        plt.close(fig)

    # Figure 3: XGBoost feature importance if present
    imp_path = output_dir / "xgb_feature_importance.csv"
    if imp_path.exists():
        imp_df = pd.read_csv(imp_path).sort_values("gain", ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg_dark)
        ax.set_facecolor(card_dark)
        ax.barh(imp_df["feature"][::-1], imp_df["gain"][::-1], color=imdb_yellow)
        ax.set_title("XGBoost Feature Importance (Gain)", color=text_light, fontsize=14, fontweight="bold")
        ax.set_xlabel("Gain", color=text_light)
        ax.tick_params(axis="x", colors=text_light)
        ax.tick_params(axis="y", colors=text_light, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(muted)
        ax.grid(axis="x", alpha=0.2, color=accent_blue)
        fig.tight_layout()
        fig.savefig(fig_dir / "styled_xgb_feature_importance.png", dpi=150, facecolor=bg_dark)
        plt.close(fig)

    # Figure 4: feature goodness top ranks
    if feature_goodness_df is not None and not feature_goodness_df.empty:
        good_df = feature_goodness_df.head(15).copy().sort_values("goodness_score", ascending=True)
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg_dark)
        ax.set_facecolor(card_dark)
        ax.barh(good_df["feature"], good_df["goodness_score"], color=accent_blue)
        ax.set_title("Feature Goodness (Top 15)", color=text_light, fontsize=14, fontweight="bold")
        ax.set_xlabel("Goodness Score", color=text_light)
        ax.tick_params(axis="x", colors=text_light)
        ax.tick_params(axis="y", colors=text_light, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(muted)
        ax.grid(axis="x", alpha=0.2, color=imdb_yellow)
        fig.tight_layout()
        fig.savefig(fig_dir / "styled_feature_goodness.png", dpi=150, facecolor=bg_dark)
        plt.close(fig)

    if feature_diagnostics_df is not None and not feature_diagnostics_df.empty:
        diag_plot = feature_diagnostics_df.sort_values("perm_auc_drop", ascending=False).head(15).copy()
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg_dark)
        ax.set_facecolor(card_dark)
        colors = [imdb_yellow if x >= 0 else accent_blue for x in diag_plot["perm_auc_drop"]]
        ax.barh(diag_plot["feature"][::-1], diag_plot["perm_auc_drop"][::-1], color=colors[::-1])
        ax.set_title("Permutation AUC Drop (Top 15)", color=text_light, fontsize=14, fontweight="bold")
        ax.set_xlabel("AUC Drop After Permutation", color=text_light)
        ax.tick_params(axis="x", colors=text_light)
        ax.tick_params(axis="y", colors=text_light, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(muted)
        ax.grid(axis="x", alpha=0.2, color=imdb_yellow)
        fig.tight_layout()
        fig.savefig(fig_dir / "styled_permutation_auc_drop.png", dpi=150, facecolor=bg_dark)
        plt.close(fig)

    # Styled HTML table report
    metrics_table = pd.DataFrame(
        [{"metric": k, "value": v} for k, v in model_metrics.items()]
    ).to_html(index=False, classes="tbl")
    compare_table = pd.DataFrame(
        [{"metric": k, "value": v} for k, v in comparison_metrics.items()]
    ).to_html(index=False, classes="tbl")
    datatype_table = ""
    duplicates_table = ""
    disguised_table = ""
    dist_table = ""
    if datatype_audit_df is not None and not datatype_audit_df.empty:
        datatype_table = datatype_audit_df.to_html(index=False, classes="tbl")
    if duplicates_df is not None and not duplicates_df.empty:
        duplicates_table = duplicates_df.to_html(index=False, classes="tbl")
    if disguised_tokens_df is not None and not disguised_tokens_df.empty:
        disguised_table = disguised_tokens_df.to_html(index=False, classes="tbl")
    if distribution_stats_df is not None and not distribution_stats_df.empty:
        dist_table = distribution_stats_df.to_html(index=False, classes="tbl")
    goodness_table = ""
    if feature_goodness_df is not None and not feature_goodness_df.empty:
        keep_cols = ["feature", "goodness_score", "univariate_auc_val", "mutual_info_train", "psi_train_vs_val"]
        goodness_table = feature_goodness_df[keep_cols].head(20).to_html(index=False, classes="tbl")

    good_features_table = ""
    bad_features_table = ""
    if feature_diagnostics_df is not None and not feature_diagnostics_df.empty:
        good_cols = ["feature", "diagnostic_score", "perm_auc_drop", "goodness_score", "xgb_gain"]
        good_df = feature_diagnostics_df[feature_diagnostics_df["status"] == "keep"][good_cols].head(20)
        bad_df = feature_diagnostics_df[feature_diagnostics_df["status"] == "drop_candidate"][good_cols].head(20)
        good_features_table = good_df.to_html(index=False, classes="tbl")
        bad_features_table = bad_df.to_html(index=False, classes="tbl")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Styled Pipeline Report</title>
  <style>
    :root {{
      --bg: {bg_dark};
      --card: {card_dark};
      --accent: {imdb_yellow};
      --accent2: {accent_blue};
      --text: {text_light};
      --muted: {muted};
    }}
    body {{
      margin: 0;
      padding: 24px;
      background: radial-gradient(circle at top right, #1a1a1a 0%, var(--bg) 50%);
      color: var(--text);
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
    }}
    h1, h2 {{
      color: var(--accent);
      margin: 0 0 10px 0;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--accent2);
      border-radius: 12px;
      padding: 16px;
      margin: 0 0 18px 0;
    }}
    .tbl {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    .tbl th {{
      text-align: left;
      background: var(--accent);
      color: #111;
      padding: 8px 10px;
    }}
    .tbl td {{
      padding: 8px 10px;
      border-bottom: 1px solid #2a2a2a;
      color: var(--text);
    }}
    .note {{
      color: var(--muted);
      font-size: 13px;
    }}
    a {{
      color: var(--accent2);
    }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Pipeline Tables & Figures</h1>
    <p class="note">IMDb-style color scheme applied to all generated visuals and tables.</p>
  </div>
  <div class="card">
    <h2>Model Metrics</h2>
    {metrics_table}
  </div>
  <div class="card">
    <h2>JSON Many-to-Many Comparison</h2>
    {compare_table}
  </div>
  <div class="card">
    <h2>Cleaning & Data Engineering Figures</h2>
    <p><a href="./figures/styled_cleaning_missing_before_after.png">styled_cleaning_missing_before_after.png</a></p>
    <p><a href="./figures/styled_edge_cleaning_impact.png">styled_edge_cleaning_impact.png</a></p>
    <p><a href="./figures/styled_json_consistency.png">styled_json_consistency.png</a></p>
    <p><a href="./figures/styled_capping_impact.png">styled_capping_impact.png</a></p>
    <p><a href="./figures/styled_imputation_impact.png">styled_imputation_impact.png</a></p>
    <p><a href="./figures/styled_datatype_parse_errors.png">styled_datatype_parse_errors.png</a></p>
    <p><a href="./figures/styled_duplicates_before_after.png">styled_duplicates_before_after.png</a></p>
    <p><a href="./figures/styled_disguised_tokens.png">styled_disguised_tokens.png</a></p>
  </div>
  <div class="card">
    <h2>Data Quality Tables</h2>
    {datatype_table}
    <br/>
    {duplicates_table}
    <br/>
    {disguised_table}
    <br/>
    {dist_table}
  </div>
  <div class="card">
    <h2>Figures</h2>
    <p><a href="./figures/styled_step_timings.png">styled_step_timings.png</a></p>
    <p><a href="./figures/styled_model_auc.png">styled_model_auc.png</a></p>
    <p><a href="./figures/styled_xgb_feature_importance.png">styled_xgb_feature_importance.png</a> (if XGBoost run)</p>
    <p><a href="./figures/styled_feature_goodness.png">styled_feature_goodness.png</a></p>
    <p><a href="./figures/styled_permutation_auc_drop.png">styled_permutation_auc_drop.png</a></p>
  </div>
  <div class="card">
    <h2>Feature Goodness (Top 20)</h2>
    {goodness_table}
  </div>
  <div class="card">
    <h2>Recommended Keep Features</h2>
    {good_features_table}
  </div>
  <div class="card">
    <h2>Drop Candidates (Review Before Removing)</h2>
    {bad_features_table}
  </div>
</body>
</html>
"""
    (output_dir / "styled_tables_report.html").write_text(html, encoding="utf-8")


def main() -> None:
    timer = StepTimer()
    paths = get_paths()

    with timer.track("load_data"):
        train_df, validation_hidden, test_hidden = load_train_and_hidden(paths["raw_csv_dir"])
        raw_movies_all = pd.concat(
            [
                train_df.assign(split="train"),
                validation_hidden.assign(split="validation_hidden"),
                test_hidden.assign(split="test_hidden"),
            ],
            ignore_index=True,
        )

    print("[DuckDB] Building many-to-many edge tables from directing.json and writing.json.")
    directors_raw, writers_raw = build_edges_from_json(paths["raw_json_dir"], timer)
    print(f"[DuckDB] Edge counts -> directors: {len(directors_raw)}, writers: {len(writers_raw)}")
    movies_clean = clean_movie_frames(train_df, validation_hidden, test_hidden, timer)
    directors_clean, writers_clean = clean_edges(directors_raw, writers_raw, timer)

    print("[Feature Engineering] Using DuckDB-derived edges to compute director/writer engineered features.")
    features = add_aggregate_features(movies_clean, directors_clean, writers_clean, timer)
    features = add_base_features(features, timer)

    # Data quality audit artifacts (datatype errors, disguised tokens, duplicates).
    audit_cols = ["primaryTitle", "originalTitle", "startYear", "endYear", "runtimeMinutes", "numVotes", "label"]
    numeric_audit_cols = {"startYear", "endYear", "runtimeMinutes", "numVotes", "label"}
    datatype_rows: List[Dict[str, object]] = []
    disguised_rows: List[Dict[str, object]] = []
    for col in audit_cols:
        if col not in raw_movies_all.columns:
            continue
        raw_col = raw_movies_all[col]
        disguised_hits = count_missing_tokens(raw_col)
        disguised_rows.append({"column": col, "disguised_missing_tokens": int(disguised_hits)})

        normalized = raw_col.astype("string").str.strip().replace(list(MISSING_TOKENS), pd.NA)
        parse_errors = 0
        if col in numeric_audit_cols:
            if col == "label":
                normalized = normalized.replace({"True": "1", "False": "0", "true": "1", "false": "0"})
            parsed = pd.to_numeric(normalized, errors="coerce")
            parse_errors = int((normalized.notna() & parsed.isna()).sum())

        datatype_rows.append(
            {
                "column": col,
                "raw_dtype": str(raw_col.dtype),
                "clean_dtype": str(movies_clean[col].dtype) if col in movies_clean.columns else "n/a",
                "parse_error_count": int(parse_errors),
                "disguised_missing_tokens": int(disguised_hits),
                "raw_non_null": int(raw_col.notna().sum()),
                "clean_non_null": int(movies_clean[col].notna().sum()) if col in movies_clean.columns else 0,
            }
        )

    datatype_audit_df = pd.DataFrame(datatype_rows)
    disguised_tokens_df = pd.DataFrame(disguised_rows).sort_values("disguised_missing_tokens", ascending=False)

    dup_before = (
        raw_movies_all.groupby("split")["tconst"]
        .apply(lambda s: int(s.duplicated().sum()))
        .reset_index(name="duplicate_tconst_before")
    )
    dup_after = (
        movies_clean.groupby("split")["tconst"]
        .apply(lambda s: int(s.duplicated().sum()))
        .reset_index(name="duplicate_tconst_after")
    )
    duplicates_df = dup_before.merge(dup_after, on="split", how="outer").fillna(0)

    edge_clean_df = pd.DataFrame(
        [
            {"entity": "directors", "raw_edges": int(len(directors_raw)), "clean_edges": int(len(directors_clean))},
            {"entity": "writers", "raw_edges": int(len(writers_raw)), "clean_edges": int(len(writers_clean))},
        ]
    )
    edge_clean_df["removed_edges"] = edge_clean_df["raw_edges"] - edge_clean_df["clean_edges"]

    cleaning_missing_df = pd.DataFrame(
        [
            {
                "field": col,
                "raw_missing_tokens": int(count_missing_tokens(raw_movies_all[col])) if col in raw_movies_all.columns else 0,
                "clean_missing_na": int(movies_clean[col].isna().sum()) if col in movies_clean.columns else 0,
            }
            for col in ["primaryTitle", "originalTitle", "startYear", "endYear", "runtimeMinutes", "numVotes"]
        ]
    )

    with timer.track("attach_people_columns"):
        dir_map, wr_map = build_name_maps(directors_clean, writers_clean)
        features = features.merge(dir_map, on="tconst", how="left").merge(wr_map, on="tconst", how="left")
        features["director_ids"] = features["director_ids"].fillna("")
        features["writer_ids"] = features["writer_ids"].fillna("")

    with timer.track("split_internal_validation"):
        labeled = features[features["split"] == "train"].copy()
        labeled = labeled.dropna(subset=["label"])
        labeled["label"] = labeled["label"].astype(int)
        train_part, validation_part = train_test_split(
            labeled,
            test_size=0.2,
            random_state=SEED,
            stratify=labeled["label"],
        )
        validation_hidden_part = features[features["split"] == "validation_hidden"].copy()
        test_hidden_part = features[features["split"] == "test_hidden"].copy()

    distribution_rows: List[Dict[str, float | str]] = []
    cap_impact_df = pd.DataFrame()
    impute_impact_df = pd.DataFrame()

    with timer.track("outlier_capping_fit_apply"):
        runtime_pre = pd.to_numeric(train_part["runtimeMinutes"], errors="coerce")
        votes_log_pre = pd.to_numeric(train_part["numVotes_log1p"], errors="coerce")
        cap_bounds = fit_cap_bounds(train_part, cols=["runtimeMinutes", "numVotes_log1p"])
        train_part = apply_caps(train_part, cap_bounds)
        validation_part = apply_caps(validation_part, cap_bounds)
        validation_hidden_part = apply_caps(validation_hidden_part, cap_bounds)
        test_hidden_part = apply_caps(test_hidden_part, cap_bounds)

        runtime_lo, runtime_hi = cap_bounds["runtimeMinutes"]
        votes_lo, votes_hi = cap_bounds["numVotes_log1p"]
        runtime_clipped = int((runtime_pre.notna() & ((runtime_pre < runtime_lo) | (runtime_pre > runtime_hi))).sum())
        votes_clipped = int((votes_log_pre.notna() & ((votes_log_pre < votes_lo) | (votes_log_pre > votes_hi))).sum())
        n_train = max(int(len(train_part)), 1)
        cap_impact_df = pd.DataFrame(
            [
                {
                    "feature": "runtimeMinutes",
                    "lower_bound": float(runtime_lo),
                    "upper_bound": float(runtime_hi),
                    "rows_clipped": runtime_clipped,
                    "clip_rate": runtime_clipped / n_train,
                },
                {
                    "feature": "numVotes_log1p",
                    "lower_bound": float(votes_lo),
                    "upper_bound": float(votes_hi),
                    "rows_clipped": votes_clipped,
                    "clip_rate": votes_clipped / n_train,
                },
            ]
        )

        for feature_name, series in [
            ("runtimeMinutes_pre_cap", runtime_pre),
            ("runtimeMinutes_capped", pd.to_numeric(train_part["runtimeMinutes_capped"], errors="coerce")),
            ("numVotes_log1p_pre_cap", votes_log_pre),
            ("numVotes_log1p_capped", pd.to_numeric(train_part["numVotes_log1p_capped"], errors="coerce")),
        ]:
            s = series.dropna()
            if s.empty:
                continue
            distribution_rows.append(
                {
                    "feature": feature_name,
                    "stage": "train",
                    "count": int(s.count()),
                    "mean": float(s.mean()),
                    "std": float(s.std(ddof=0)),
                    "min": float(s.min()),
                    "p01": float(s.quantile(0.01)),
                    "p50": float(s.quantile(0.50)),
                    "p99": float(s.quantile(0.99)),
                    "max": float(s.max()),
                }
            )

    with timer.track("target_encoding_oof"):
        director_index = build_entity_index(directors_clean, "director_id")
        writer_index = build_entity_index(writers_clean, "writer_id")

        train_part = train_part.reset_index(drop=True)
        validation_part = validation_part.reset_index(drop=True)
        validation_hidden_part = validation_hidden_part.reset_index(drop=True)
        test_hidden_part = test_hidden_part.reset_index(drop=True)

        dir_oof, dir_lookup, dir_prior = compute_oof_target_encoding(train_part, director_index)
        wr_oof, wr_lookup, wr_prior = compute_oof_target_encoding(train_part, writer_index)
        train_part["director_hit_rate"] = dir_oof
        train_part["writer_hit_rate"] = wr_oof

        validation_part["director_hit_rate"] = apply_target_encoding(validation_part, director_index, dir_lookup, dir_prior)
        validation_part["writer_hit_rate"] = apply_target_encoding(validation_part, writer_index, wr_lookup, wr_prior)
        validation_hidden_part["director_hit_rate"] = apply_target_encoding(
            validation_hidden_part, director_index, dir_lookup, dir_prior
        )
        validation_hidden_part["writer_hit_rate"] = apply_target_encoding(
            validation_hidden_part, writer_index, wr_lookup, wr_prior
        )
        test_hidden_part["director_hit_rate"] = apply_target_encoding(
            test_hidden_part, director_index, dir_lookup, dir_prior
        )
        test_hidden_part["writer_hit_rate"] = apply_target_encoding(test_hidden_part, writer_index, wr_lookup, wr_prior)

        title_oof, title_lookup, title_prior = compute_oof_group_rate(train_part["canonical_title"], train_part["label"])
        train_part["canonical_title_hit_rate"] = title_oof
        validation_part["canonical_title_hit_rate"] = apply_group_rate(
            validation_part["canonical_title"], title_lookup, title_prior
        )
        validation_hidden_part["canonical_title_hit_rate"] = apply_group_rate(
            validation_hidden_part["canonical_title"], title_lookup, title_prior
        )
        test_hidden_part["canonical_title_hit_rate"] = apply_group_rate(
            test_hidden_part["canonical_title"], title_lookup, title_prior
        )

    with timer.track("title_group_conflict_features"):
        title_stats = (
            train_part.groupby("canonical_title")
            .agg(
                title_group_size_train=("tconst", "size"),
                title_unique_years_train=("startYear", lambda s: pd.to_numeric(s, errors="coerce").dropna().nunique()),
            )
            .reset_index()
        )
        title_stats["title_conflicting_years"] = (title_stats["title_unique_years_train"] > 1).astype(float)

        for frame in [train_part, validation_part, validation_hidden_part, test_hidden_part]:
            frame["title_group_size_train"] = (
                frame["canonical_title"].map(title_stats.set_index("canonical_title")["title_group_size_train"]).fillna(0).astype(float)
            )
            frame["title_unique_years_train"] = (
                frame["canonical_title"].map(title_stats.set_index("canonical_title")["title_unique_years_train"]).fillna(0).astype(float)
            )
            frame["title_conflicting_years"] = (
                frame["canonical_title"].map(title_stats.set_index("canonical_title")["title_conflicting_years"]).fillna(0).astype(float)
            )

    with timer.track("title_similarity_features"):
        oof_hit_sim, oof_non_hit_sim, title_vectorizer, hit_centroid, non_hit_centroid = build_title_similarity_features_oof(
            train_part["primaryTitle"], train_part["label"]
        )
        train_part["title_sim_to_hit"] = oof_hit_sim
        train_part["title_sim_to_non_hit"] = oof_non_hit_sim
        train_part["title_sim_margin"] = train_part["title_sim_to_hit"] - train_part["title_sim_to_non_hit"]

        val_hit, val_non_hit = apply_title_similarity_features(
            validation_part["primaryTitle"], title_vectorizer, hit_centroid, non_hit_centroid
        )
        vh_hit, vh_non_hit = apply_title_similarity_features(
            validation_hidden_part["primaryTitle"], title_vectorizer, hit_centroid, non_hit_centroid
        )
        test_hit, test_non_hit = apply_title_similarity_features(
            test_hidden_part["primaryTitle"], title_vectorizer, hit_centroid, non_hit_centroid
        )

        validation_part["title_sim_to_hit"] = val_hit
        validation_part["title_sim_to_non_hit"] = val_non_hit
        validation_part["title_sim_margin"] = validation_part["title_sim_to_hit"] - validation_part["title_sim_to_non_hit"]

        validation_hidden_part["title_sim_to_hit"] = vh_hit
        validation_hidden_part["title_sim_to_non_hit"] = vh_non_hit
        validation_hidden_part["title_sim_margin"] = (
            validation_hidden_part["title_sim_to_hit"] - validation_hidden_part["title_sim_to_non_hit"]
        )

        test_hidden_part["title_sim_to_hit"] = test_hit
        test_hidden_part["title_sim_to_non_hit"] = test_non_hit
        test_hidden_part["title_sim_margin"] = test_hidden_part["title_sim_to_hit"] - test_hidden_part["title_sim_to_non_hit"]

    cleaning_summary = build_cleaning_summary(
        train_df=train_df,
        validation_hidden=validation_hidden,
        test_hidden=test_hidden,
        movies_clean=movies_clean,
        directors_raw=directors_raw,
        writers_raw=writers_raw,
        directors_clean=directors_clean,
        writers_clean=writers_clean,
        features=features,
        train_part=train_part,
        title_stats=title_stats,
        cap_bounds=cap_bounds,
    )

    feature_cols = [
        "startYear",
        "endYear",
        "runtimeMinutes_capped",
        "numVotes_log1p_capped",
        "title_len",
        "title_word_count",
        "title_has_digit",
        "title_has_colon",
        "title_has_question",
        "title_upper_ratio",
        "has_original_title",
        "runtime_missing",
        "votes_missing",
        "start_missing",
        "end_missing",
        "year_span",
        "num_directors",
        "num_unique_directors",
        "num_writers",
        "num_unique_writers",
        "is_auteur",
        "director_hit_rate",
        "writer_hit_rate",
        "canonical_title_hit_rate",
        "title_group_size_train",
        "title_unique_years_train",
        "title_conflicting_years",
        "title_sim_to_hit",
        "title_sim_to_non_hit",
        "title_sim_margin",
    ]

    with timer.track("impute_and_scale"):
        missing_before = train_part[feature_cols].isna().sum()
        train_impute = train_part[feature_cols].median(numeric_only=True)
        train_X = train_part[feature_cols].fillna(train_impute)
        val_X = validation_part[feature_cols].fillna(train_impute)
        val_hidden_X = validation_hidden_part[feature_cols].fillna(train_impute)
        test_hidden_X = test_hidden_part[feature_cols].fillna(train_impute)
        missing_after = train_X.isna().sum()
        impute_impact_df = pd.DataFrame(
            {
                "feature": feature_cols,
                "missing_before": [int(missing_before.get(f, 0)) for f in feature_cols],
                "missing_after": [int(missing_after.get(f, 0)) for f in feature_cols],
            }
        )
        impute_impact_df["imputed_values"] = impute_impact_df["missing_before"] - impute_impact_df["missing_after"]

        scaler = StandardScaler()
        train_X_scaled = scaler.fit_transform(train_X)
        val_X_scaled = scaler.transform(val_X)
        val_hidden_X_scaled = scaler.transform(val_hidden_X)
        test_hidden_X_scaled = scaler.transform(test_hidden_X)

    distribution_stats_df = pd.DataFrame(distribution_rows)

    with timer.track("feature_goodness_analysis"):
        feature_goodness_df = compute_feature_goodness(
            train_X=train_X,
            train_y=train_part["label"],
            val_X=val_X,
            val_y=validation_part["label"],
            feature_cols=feature_cols,
        )

    with timer.track("train_and_validate_models"):
        logistic_model = LogisticRegression(max_iter=2000, random_state=SEED)
        logistic_model.fit(train_X_scaled, train_part["label"].astype(int))
        val_probs_logistic = logistic_model.predict_proba(val_X_scaled)[:, 1]
        val_preds_logistic = (val_probs_logistic >= 0.5).astype(int)

        model_metrics = {
            "logistic_validation_accuracy": float(
                accuracy_score(validation_part["label"].astype(int), val_preds_logistic)
            ),
            "logistic_validation_roc_auc": float(
                roc_auc_score(validation_part["label"].astype(int), val_probs_logistic)
            ),
            "train_rows": int(len(train_part)),
            "validation_rows": int(len(validation_part)),
        }

        xgb_model = None
        xgb_imp_df = pd.DataFrame(columns=["feature", "gain", "weight", "cover"])
        if XGBClassifier is not None:
            xgb_model = XGBClassifier(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=SEED,
                n_jobs=4,
            )
            xgb_model.fit(train_X, train_part["label"].astype(int))
            val_probs_xgb = xgb_model.predict_proba(val_X)[:, 1]
            val_preds_xgb = (val_probs_xgb >= 0.5).astype(int)
            model_metrics["xgb_validation_accuracy"] = float(
                accuracy_score(validation_part["label"].astype(int), val_preds_xgb)
            )
            model_metrics["xgb_validation_roc_auc"] = float(
                roc_auc_score(validation_part["label"].astype(int), val_probs_xgb)
            )

            booster = xgb_model.get_booster()
            gain_map = booster.get_score(importance_type="gain")
            weight_map = booster.get_score(importance_type="weight")
            cover_map = booster.get_score(importance_type="cover")

            def map_feat_name(raw_key: str) -> str:
                if raw_key in feature_cols:
                    return raw_key
                if raw_key.startswith("f") and raw_key[1:].isdigit():
                    idx = int(raw_key[1:])
                    if 0 <= idx < len(feature_cols):
                        return feature_cols[idx]
                return raw_key

            rows = []
            for feat in feature_cols:
                rows.append({"feature": feat, "gain": 0.0, "weight": 0.0, "cover": 0.0})
            xgb_imp_df = pd.DataFrame(rows).set_index("feature")
            for raw_key, value in gain_map.items():
                xgb_imp_df.loc[map_feat_name(raw_key), "gain"] = float(value)
            for raw_key, value in weight_map.items():
                xgb_imp_df.loc[map_feat_name(raw_key), "weight"] = float(value)
            for raw_key, value in cover_map.items():
                xgb_imp_df.loc[map_feat_name(raw_key), "cover"] = float(value)
            xgb_imp_df = xgb_imp_df.reset_index().sort_values("gain", ascending=False)
        else:
            model_metrics["xgb_validation_accuracy"] = -1.0
            model_metrics["xgb_validation_roc_auc"] = -1.0

        best_model_name = "logistic"
        best_auc = model_metrics["logistic_validation_roc_auc"]
        if xgb_model is not None and model_metrics["xgb_validation_roc_auc"] >= best_auc:
            best_model_name = "xgboost"
            best_auc = model_metrics["xgb_validation_roc_auc"]
        model_metrics["best_model_for_hidden_predictions"] = best_model_name

    with timer.track("feature_diagnostics_ablation_shap"):
        if best_model_name == "xgboost" and xgb_model is not None:
            best_model_obj = xgb_model
            best_scaler = None
        else:
            best_model_obj = logistic_model
            best_scaler = scaler

        perm_df = compute_permutation_auc_drop(
            model_name=best_model_name,
            model_obj=best_model_obj,
            X_val=val_X,
            y_val=validation_part["label"],
            feature_cols=feature_cols,
            scaler=best_scaler,
        )

        shap_df = compute_xgb_shap_summary(xgb_model, val_X, feature_cols)
        feature_diagnostics_df = feature_goodness_df.merge(perm_df[["feature", "perm_auc_drop"]], on="feature", how="left")
        feature_diagnostics_df = feature_diagnostics_df.merge(
            xgb_imp_df[["feature", "gain"]].rename(columns={"gain": "xgb_gain"}),
            on="feature",
            how="left",
        )
        feature_diagnostics_df = feature_diagnostics_df.merge(shap_df, on="feature", how="left")
        feature_diagnostics_df = classify_feature_status(feature_diagnostics_df)

    with timer.track("reduced_feature_retrain"):
        keep_features = feature_diagnostics_df[feature_diagnostics_df["status"] == "keep"]["feature"].tolist()
        if len(keep_features) < 8:
            keep_features = feature_diagnostics_df.sort_values("diagnostic_score", ascending=False)["feature"].head(15).tolist()

        reduced_metrics = {"feature_count": int(len(keep_features))}
        if best_model_name == "xgboost" and XGBClassifier is not None:
            reduced_model = XGBClassifier(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=SEED,
                n_jobs=4,
            )
            reduced_model.fit(train_X[keep_features], train_part["label"].astype(int))
            val_probs_reduced = reduced_model.predict_proba(val_X[keep_features])[:, 1]
        else:
            reduced_scaler = StandardScaler()
            train_red = reduced_scaler.fit_transform(train_X[keep_features])
            val_red = reduced_scaler.transform(val_X[keep_features])
            reduced_model = LogisticRegression(max_iter=2000, random_state=SEED)
            reduced_model.fit(train_red, train_part["label"].astype(int))
            val_probs_reduced = reduced_model.predict_proba(val_red)[:, 1]

        reduced_preds = (val_probs_reduced >= 0.5).astype(int)
        reduced_metrics["validation_accuracy"] = float(
            accuracy_score(validation_part["label"].astype(int), reduced_preds)
        )
        reduced_metrics["validation_roc_auc"] = float(
            roc_auc_score(validation_part["label"].astype(int), val_probs_reduced)
        )
        reduced_metrics["base_model"] = best_model_name
        reduced_metrics["full_model_roc_auc"] = float(
            model_metrics["xgb_validation_roc_auc"] if best_model_name == "xgboost" else model_metrics["logistic_validation_roc_auc"]
        )
        reduced_metrics["roc_auc_delta_vs_full"] = reduced_metrics["validation_roc_auc"] - reduced_metrics["full_model_roc_auc"]

    with timer.track("export_artifacts"):
        output_dir = paths["output_dir"]
        meta_cols = ["tconst", "primaryTitle", "originalTitle", "canonical_title", "director_ids", "writer_ids"]

        train_export = train_part[["tconst", "label"] + feature_cols].copy()
        val_export = validation_part[["tconst", "label"] + feature_cols].copy()
        val_hidden_export = validation_hidden_part[["tconst"] + feature_cols].copy()
        test_export = test_hidden_part[["tconst"] + feature_cols].copy()

        train_export.to_csv(output_dir / "train_features.csv", index=False)
        val_export.to_csv(output_dir / "validation_features.csv", index=False)
        val_hidden_export.to_csv(output_dir / "validation_hidden_features.csv", index=False)
        test_export.to_csv(output_dir / "test_features.csv", index=False)
        datatype_audit_df.to_csv(output_dir / "datatype_audit.csv", index=False)
        duplicates_df.to_csv(output_dir / "duplicates_audit.csv", index=False)
        disguised_tokens_df.to_csv(output_dir / "disguised_tokens_audit.csv", index=False)
        cleaning_missing_df.to_csv(output_dir / "cleaning_missing_before_after.csv", index=False)
        edge_clean_df.to_csv(output_dir / "edge_cleaning_impact.csv", index=False)
        cap_impact_df.to_csv(output_dir / "capping_impact.csv", index=False)
        impute_impact_df.to_csv(output_dir / "imputation_impact.csv", index=False)
        distribution_stats_df.to_csv(output_dir / "distribution_statistics.csv", index=False)
        feature_goodness_df.to_csv(output_dir / "feature_goodness.csv", index=False)
        feature_diagnostics_df.to_csv(output_dir / "feature_diagnostics.csv", index=False)
        xgb_imp_df.to_csv(output_dir / "xgb_feature_importance.csv", index=False)
        shap_df.to_csv(output_dir / "xgb_shap_summary.csv", index=False)
        (output_dir / "reduced_model_validation_metrics.json").write_text(
            json.dumps(reduced_metrics, indent=2),
            encoding="utf-8",
        )
        keep_features_df = feature_diagnostics_df[feature_diagnostics_df["status"] == "keep"][["feature", "diagnostic_score"]]
        drop_features_df = feature_diagnostics_df[feature_diagnostics_df["status"] == "drop_candidate"][
            ["feature", "diagnostic_score", "perm_auc_drop", "goodness_score"]
        ]
        keep_features_df.to_csv(output_dir / "recommended_keep_features.csv", index=False)
        drop_features_df.to_csv(output_dir / "drop_candidate_features.csv", index=False)

        train_part[meta_cols + ["label"] + feature_cols].to_csv(output_dir / "train_enriched_features.csv", index=False)
        validation_part[meta_cols + ["label"] + feature_cols].to_csv(
            output_dir / "validation_enriched_features.csv", index=False
        )
        validation_hidden_part[meta_cols + feature_cols].to_csv(
            output_dir / "validation_hidden_enriched_features.csv", index=False
        )
        test_hidden_part[meta_cols + feature_cols].to_csv(output_dir / "test_enriched_features.csv", index=False)

        val_hidden_probs_logistic = logistic_model.predict_proba(val_hidden_X_scaled)[:, 1]
        test_hidden_probs_logistic = logistic_model.predict_proba(test_hidden_X_scaled)[:, 1]
        val_hidden_preds_logistic = (val_hidden_probs_logistic >= 0.5)
        test_hidden_preds_logistic = (test_hidden_probs_logistic >= 0.5)

        (output_dir / "validation_hidden_predictions_logistic.txt").write_text(
            "\n".join("True" if x else "False" for x in val_hidden_preds_logistic) + "\n",
            encoding="utf-8",
        )
        (output_dir / "test_hidden_predictions_logistic.txt").write_text(
            "\n".join("True" if x else "False" for x in test_hidden_preds_logistic) + "\n",
            encoding="utf-8",
        )

        if xgb_model is not None:
            val_hidden_probs_xgb = xgb_model.predict_proba(val_hidden_X)[:, 1]
            test_hidden_probs_xgb = xgb_model.predict_proba(test_hidden_X)[:, 1]
            val_hidden_preds_xgb = (val_hidden_probs_xgb >= 0.5)
            test_hidden_preds_xgb = (test_hidden_probs_xgb >= 0.5)

            (output_dir / "validation_hidden_predictions_xgboost.txt").write_text(
                "\n".join("True" if x else "False" for x in val_hidden_preds_xgb) + "\n",
                encoding="utf-8",
            )
            (output_dir / "test_hidden_predictions_xgboost.txt").write_text(
                "\n".join("True" if x else "False" for x in test_hidden_preds_xgb) + "\n",
                encoding="utf-8",
            )

            pass

        if model_metrics["best_model_for_hidden_predictions"] == "xgboost" and xgb_model is not None:
            chosen_val_preds = (xgb_model.predict_proba(val_hidden_X)[:, 1] >= 0.5)
            chosen_test_preds = (xgb_model.predict_proba(test_hidden_X)[:, 1] >= 0.5)
        else:
            chosen_val_preds = val_hidden_preds_logistic
            chosen_test_preds = test_hidden_preds_logistic

        (output_dir / "validation_hidden_predictions.txt").write_text(
            "\n".join("True" if x else "False" for x in chosen_val_preds) + "\n",
            encoding="utf-8",
        )
        (output_dir / "test_hidden_predictions.txt").write_text(
            "\n".join("True" if x else "False" for x in chosen_test_preds) + "\n",
            encoding="utf-8",
        )

        (output_dir / "model_validation_metrics.json").write_text(
            json.dumps(model_metrics, indent=2),
            encoding="utf-8",
        )

        conversion_metrics = build_comparison_metrics(paths["project_root"])
        pd.DataFrame([conversion_metrics]).to_csv(output_dir / "json_many_to_many_comparison.csv", index=False)

    timings_df = pd.DataFrame(timer.timings)
    timings_df.to_csv(paths["output_dir"] / "step_timings.csv", index=False)
    generate_styled_visuals_and_tables(
        output_dir=paths["output_dir"],
        timings_df=timings_df,
        model_metrics=model_metrics,
        comparison_metrics=conversion_metrics,
        feature_goodness_df=feature_goodness_df,
        feature_diagnostics_df=feature_diagnostics_df,
        cleaning_missing_df=cleaning_missing_df,
        edge_clean_df=edge_clean_df,
        cap_impact_df=cap_impact_df,
        impute_impact_df=impute_impact_df,
        datatype_audit_df=datatype_audit_df,
        duplicates_df=duplicates_df,
        disguised_tokens_df=disguised_tokens_df,
        distribution_stats_df=distribution_stats_df,
    )
    report_path = paths["member_root"] / "FIX_COMPARISON_REPORT.md"
    write_comparison_report(
        report_path=report_path,
        issue_fix_log=ISSUE_FIX_LOG,
        metrics=conversion_metrics,
        model_metrics=model_metrics,
        timing_path=paths["output_dir"] / "step_timings.csv",
    )
    cleaning_report_path = paths["member_root"] / "CLEANING_PIPELINE_DOC.md"
    write_cleaning_pipeline_report(
        report_path=cleaning_report_path,
        summary=cleaning_summary,
        conversion_metrics=conversion_metrics,
        timings_df=timings_df,
        output_dir=paths["output_dir"],
    )
    pd.DataFrame(
        [
            {
                "step": step,
                "engine": STEP_ENGINE_USAGE.get(step, ("Unspecified", ""))[0],
                "why_engine": STEP_ENGINE_USAGE.get(step, ("Unspecified", "No explicit mapping set."))[1],
            }
            for step in timings_df["step"].tolist()
        ]
    ).to_csv(paths["output_dir"] / "engine_usage_map.csv", index=False)
    pd.DataFrame(
        [{"metric": k, "value": v} for k, v in cleaning_summary.items()]
    ).to_csv(paths["output_dir"] / "cleaning_audit_metrics.csv", index=False)

    print("Pipeline completed successfully.")
    print(f"Artifacts: {paths['output_dir']}")
    print(f"Comparison report: {paths['member_root'] / 'FIX_COMPARISON_REPORT.md'}")
    print(f"Cleaning pipeline report: {cleaning_report_path}")


if __name__ == "__main__":
    main()
