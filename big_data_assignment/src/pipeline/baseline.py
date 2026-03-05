from __future__ import annotations

import glob
import os
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.data.dataloaders import load_train_data, load_validation_data
from src.models import baseline_model, logistic_regression, xgboost_model
from src.utils.config import load_config, resolve_path_from_config
from src.utils.grid_search import expand_model_grid, flatten_model_params

# After run_baselines(), the best model (by validation accuracy) is stored here for make_submission().
_best_baseline: Dict[str, Any] | None = None


def _text_to_matrix(X) -> np.ndarray:
    """Convert text series to a numeric matrix (no BoW/TF-IDF). Uses a single length feature per sample."""
    return np.array([[len(str(s))] for s in X], dtype=np.float64)


def _get_director_writer_tokens(csv_dir: str) -> Tuple[pd.Series | None, pd.Series | None]:
    """Load and aggregate director/writer tokens per tconst from movie_directors.csv and movie_writers.csv."""
    director_tokens = None
    writer_tokens = None
    directors_path = os.path.join(csv_dir, "movie_directors.csv")
    writers_path = os.path.join(csv_dir, "movie_writers.csv")
    if os.path.exists(directors_path):
        directors_df = pd.read_csv(directors_path)
        if {"tconst", "director"}.issubset(directors_df.columns):
            director_tokens = (
                directors_df.groupby("tconst")["director"]
                .apply(lambda s: " ".join(f"director_{d}" for d in s.astype(str)))
                .rename("director_tokens")
            )
    if os.path.exists(writers_path):
        writers_df = pd.read_csv(writers_path)
        if {"tconst", "writer"}.issubset(writers_df.columns):
            writer_tokens = (
                writers_df.groupby("tconst")["writer"]
                .apply(lambda s: " ".join(f"writer_{w}" for w in s.astype(str)))
                .rename("writer_tokens")
            )
    return director_tokens, writer_tokens


def _enrich_df_text(df: pd.DataFrame, director_tokens: pd.Series | None, writer_tokens: pd.Series | None) -> pd.Series:
    """Build enriched text series from df (primaryTitle + director_tokens + writer_tokens)."""
    title_col = "primaryTitle" if "primaryTitle" in df.columns else None
    if title_col is None:
        raise ValueError(f"Expected 'primaryTitle' column, found {df.columns.tolist()}")
    base_text = df[title_col].fillna("").astype(str)
    if director_tokens is not None:
        df = df.merge(director_tokens, on="tconst", how="left")
    else:
        df = df.copy()
        df["director_tokens"] = ""
    if writer_tokens is not None:
        df = df.merge(writer_tokens, on="tconst", how="left")
    else:
        df["writer_tokens"] = ""
    enriched = (base_text + " " + df["director_tokens"].fillna("") + " " + df["writer_tokens"].fillna("")).str.strip()
    return enriched


def _load_train_val_from_raw_chunks(config: Dict[str, Any]) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Load training and validation data directly from the raw CSV chunks in
    ``data/raw/csv``.

    This uses:
      - train-1.csv ... train-8.csv as the labeled data
      - movie_directors.csv and movie_writers.csv to enrich each movie with
        director / writer tokens

    A train/validation split is created using the ``training.validation_size``
    and ``training.random_seed`` entries from the config.
    """
    raw_dir = resolve_path_from_config(config, "data", "raw_dir")
    csv_dir = os.path.join(raw_dir, "csv")

    if not os.path.isdir(csv_dir):
        raise FileNotFoundError(f"Expected CSV directory at {csv_dir} but it does not exist.")

    # 1) Load all train-*.csv chunks and concatenate
    train_paths = sorted(glob.glob(os.path.join(csv_dir, "train-*.csv")))
    if not train_paths:
        raise FileNotFoundError(f"No train-*.csv files found in {csv_dir}")

    frames = [pd.read_csv(p) for p in train_paths]
    train_df = pd.concat(frames, ignore_index=True)

    # Ensure required columns are present
    if "tconst" not in train_df.columns or "label" not in train_df.columns:
        raise ValueError(
            f"Expected 'tconst' and 'label' columns in train CSVs, "
            f"found {train_df.columns.tolist()}"
        )

    director_tokens, writer_tokens = _get_director_writer_tokens(csv_dir)
    enriched_text = _enrich_df_text(train_df, director_tokens, writer_tokens)
    y = train_df["label"]

    train_cfg = config.get("training", {})
    test_size = float(train_cfg.get("validation_size", 0.2))
    random_state = int(train_cfg.get("random_seed", 42))

    X_train, X_val, y_train, y_val = train_test_split(
        enriched_text,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_val, y_train, y_val


def _load_hidden_from_raw_chunks(config: Dict[str, Any], file_key: str) -> pd.Series:
    """
    Load validation_hidden or test_hidden from data/raw/csv and return enriched text series
    (same order as rows in the CSV). file_key should be "validation_hidden_file" or "test_hidden_file".
    """
    raw_dir = resolve_path_from_config(config, "data", "raw_dir")
    csv_dir = os.path.join(raw_dir, "csv")
    filename = config.get("data", {}).get(file_key, "")
    if not filename:
        raise KeyError(f"Config key data.{file_key} not found.")
    path = os.path.join(csv_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Hidden file not found: {path}")
    df = pd.read_csv(path)
    if "tconst" not in df.columns:
        raise ValueError(f"Expected 'tconst' in {path}, found {df.columns.tolist()}")
    director_tokens, writer_tokens = _get_director_writer_tokens(csv_dir)
    return _enrich_df_text(df, director_tokens, writer_tokens)


def run_baselines() -> None:
    """
    Simple shared baseline:
    - prefers using raw CSV chunks in data/raw/csv (train-*.csv + movie_* files)
    - falls back to the original config-based CSVs if those are not present
    - trains/evaluates logistic regression, XGBoost and a baseline model (no BoW/TF-IDF preprocessing).

    If config has training.grid_search.enabled: true, runs grid search over
    the current model type only (model.type) using list values in config;
    reports best params and accuracy. Otherwise uses first value of any list.
    """
    global _best_baseline
    config = load_config()

    try:
        # Preferred: use the raw CSV shards + movie_directors / movie_writers
        X_train, X_val, y_train, y_val = _load_train_val_from_raw_chunks(config)
        print("Loaded data from data/raw/csv (train-*.csv + movie_directors/movie_writers).")
    except Exception as e:
        print(
            "Falling back to config-based CSV loading via src.data.dataloaders "
            f"because raw-chunk loading failed with: {e}"
        )
        X_train, y_train = load_train_data(config)
        X_val, y_val = load_validation_data(config)

    # No BoW/TF-IDF: convert text to a minimal numeric matrix for the classifiers
    X_train_vec = _text_to_matrix(X_train)
    X_val_vec = _text_to_matrix(X_val)

    models = {
        "logistic_regression": logistic_regression,
        "xgboost": xgboost_model,
        "baseline": baseline_model,
    }

    grid_enabled = config.get("training", {}).get("grid_search", {}).get("enabled", False)
    model_type = config.get("model", {}).get("type", "logistic_regression")

    if grid_enabled:
        # Run grid search for the selected model type only; store best for make_submission()
        best_acc = -1.0
        best_params = None
        best_model = None
        grid_results = []
        n = 0
        for flat_config in expand_model_grid(config):
            n += 1
            params = flat_config.get("model", {}).get(model_type, {})
            print(f"Grid point {n}: {params}")
            module = models[model_type]
            model = module.train(X_train_vec, y_train, X_val_vec, y_val, flat_config)
            y_val_pred = module.predict(model, X_val_vec)
            acc = accuracy_score(y_val, y_val_pred)
            grid_results.append((params, acc))
            print(f"  Validation accuracy: {acc:.4f}\n")
            if acc > best_acc:
                best_acc = acc
                best_params = params
                best_model = model
        print("Summary (all grid configs -> validation accuracy):")
        for i, (params, acc) in enumerate(grid_results, 1):
            print(f"  {i}. {params} -> {acc:.4f}")
        print(f"Best: {best_params} -> accuracy {best_acc:.4f}")
        _best_baseline = {
            "model_type": model_type,
            "model": best_model,
            "transform": _text_to_matrix,
            "config": config,
        }
        return

    # Single run: all three models (lists in config use first value); store best for make_submission()
    results = {}
    summary_configs = {}
    best_acc = -1.0
    best_name = None
    best_model = None
    for name, module in models.items():
        params = config.get("model", {}).get(name, {})
        flat_params = flatten_model_params(params) if isinstance(params, dict) else {}
        summary_configs[name] = flat_params
        print(f"Training {name}...")
        model = module.train(X_train_vec, y_train, X_val_vec, y_val, config)
        y_val_pred = module.predict(model, X_val_vec)
        acc = accuracy_score(y_val, y_val_pred)
        results[name] = acc
        print(f"  config: {flat_params}")
        print(f"  Validation accuracy ({name}): {acc:.4f}\n")
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = model

    print("Summary (model -> config -> validation accuracy):")
    for name in results:
        print(f"  {name}: {summary_configs[name]} -> {results[name]:.4f}")
    print(f"Best model for submission: {best_name} (validation accuracy {best_acc:.4f})")
    _best_baseline = {
        "model_type": best_name,
        "model": best_model,
        "transform": _text_to_matrix,
        "config": config,
    }


def make_submission() -> str:
    """
    Generate submission files using the **best model from the last run_baselines()** call.

    You must call run_baselines() first. make_submission() then uses the model that had
    the highest validation accuracy (and the same feature transform) to predict on
    validation_hidden.csv and test_hidden.csv.

    - Loads validation_hidden.csv and test_hidden.csv from data/raw/csv.
    - Writes two text files in the `submissions/` folder with model name and timestamp:
      - validation_predictions_<model_type>_<YYYY-MM-DD_HH-MM-SS>.txt
      - test_predictions_<model_type>_<YYYY-MM-DD_HH-MM-SS>.txt
    Each line is "True" or "False".

    Returns the path to the submissions directory.

    Raises
    ------
    RuntimeError
        If run_baselines() has not been called yet (no best model stored).
    """
    global _best_baseline
    if _best_baseline is None:
        raise RuntimeError(
            "No best model stored. Call run_baselines() first, then make_submission()."
        )

    config = _best_baseline["config"]
    model_type = _best_baseline["model_type"]
    model = _best_baseline["model"]
    transform = _best_baseline["transform"]

    # Load hidden sets and convert to same numeric representation as at training time
    X_val_hidden = _load_hidden_from_raw_chunks(config, "validation_hidden_file")
    X_test_hidden = _load_hidden_from_raw_chunks(config, "test_hidden_file")
    X_val_hidden_vec = transform(X_val_hidden)
    X_test_hidden_vec = transform(X_test_hidden)

    # Predict with the best model
    module = {
        "logistic_regression": logistic_regression,
        "xgboost": xgboost_model,
        "baseline": baseline_model,
    }[model_type]
    y_val_pred = module.predict(model, X_val_hidden_vec)
    y_test_pred = module.predict(model, X_test_hidden_vec)

    def to_submission_line(b: Any) -> str:
        return "True" if b else "False"

    val_lines = [to_submission_line(b) for b in y_val_pred]
    test_lines = [to_submission_line(b) for b in y_test_pred]

    base_submissions = resolve_path_from_config(config, "paths", "submissions_dir")
    os.makedirs(base_submissions, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    val_path = os.path.join(base_submissions, f"validation_predictions_{model_type}_{timestamp}.txt")
    test_path = os.path.join(base_submissions, f"test_predictions_{model_type}_{timestamp}.txt")
    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val_lines) + "\n")
    with open(test_path, "w", encoding="utf-8") as f:
        f.write("\n".join(test_lines) + "\n")

    print(f"Submission written to: {base_submissions} (best model: {model_type})")
    print(f"  - {val_path} ({len(val_lines)} lines)")
    print(f"  - {test_path} ({len(test_lines)} lines)")
    return base_submissions


if __name__ == "__main__":
    run_baselines()

