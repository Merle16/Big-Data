from __future__ import annotations

import glob
import os
from datetime import datetime
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.data.dataloaders import load_train_data, load_validation_data
from src.models import baseline_model, logistic_regression, xgboost_model
from src.utils.config import load_config, resolve_path_from_config
from src.utils.grid_search import expand_model_grid


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
    - builds a basic bag-of-words representation
    - trains/evaluates logistic regression, XGBoost and a baseline model.

    If config has training.grid_search.enabled: true, runs grid search over
    the current model type only (model.type) using list values in config;
    reports best params and accuracy. Otherwise uses first value of any list.
    """
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

    vectorizer = CountVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train.astype(str))
    X_val_vec = vectorizer.transform(X_val.astype(str))

    models = {
        "logistic_regression": logistic_regression,
        "xgboost": xgboost_model,
        "baseline": baseline_model,
    }

    grid_enabled = config.get("training", {}).get("grid_search", {}).get("enabled", False)
    model_type = config.get("model", {}).get("type", "logistic_regression")

    if grid_enabled:
        # Run grid search for the selected model type only
        best_acc = -1.0
        best_params = None
        n = 0
        for flat_config in expand_model_grid(config):
            n += 1
            params = flat_config.get("model", {}).get(model_type, {})
            print(f"Grid point {n}: {params}")
            module = models[model_type]
            model = module.train(X_train_vec, y_train, X_val_vec, y_val, flat_config)
            y_val_pred = module.predict(model, X_val_vec)
            acc = accuracy_score(y_val, y_val_pred)
            print(f"  Validation accuracy: {acc:.4f}\n")
            if acc > best_acc:
                best_acc = acc
                best_params = params
        print(f"Grid search done ({n} points). Best: {best_params} -> accuracy {best_acc:.4f}")
        return

    # Single run: all three models (lists in config use first value)
    results = {}
    for name, module in models.items():
        print(f"Training {name}...")
        model = module.train(X_train_vec, y_train, X_val_vec, y_val, config)
        y_val_pred = module.predict(model, X_val_vec)
        acc = accuracy_score(y_val, y_val_pred)
        results[name] = acc
        print(f"Validation accuracy ({name}): {acc:.4f}\n")

    print("Summary:", results)


def make_submission() -> str:
    """
    Generate submission files for the course server.

    - Loads training data from data/raw/csv (train-*.csv + movie_directors/movie_writers).
    - Loads validation_hidden.csv and test_hidden.csv from data/raw/csv.
    - Trains the model specified in config (model.type: logistic_regression, xgboost, or baseline).
    - Writes two text files in the `submissions/` folder with model name and timestamp in the filename:
      - validation_predictions_<model_type>_<YYYY-MM-DD_HH-MM-SS>.txt
      - test_predictions_<model_type>_<YYYY-MM-DD_HH-MM-SS>.txt
    Each line is the string "True" or "False" for the corresponding row in the hidden CSV.

    Returns the path to the submissions directory.
    """
    config = load_config()

    # Load training data (same as run_baselines)
    try:
        X_train, X_val, y_train, y_val = _load_train_val_from_raw_chunks(config)
    except Exception as e:
        raise RuntimeError(
            "make_submission() requires data/raw/csv with train-*.csv and hidden files. "
            f"Raw chunk loading failed: {e}"
        ) from e

    # Combine train + val for final model
    X_full = pd.concat([X_train, X_val], ignore_index=True)
    y_full = pd.concat([y_train, y_val], ignore_index=True)

    # Load hidden sets (same enrichment as train)
    X_val_hidden = _load_hidden_from_raw_chunks(config, "validation_hidden_file")
    X_test_hidden = _load_hidden_from_raw_chunks(config, "test_hidden_file")

    # Vectorize
    vectorizer = CountVectorizer(max_features=5000)
    X_full_vec = vectorizer.fit_transform(X_full.astype(str))
    X_val_hidden_vec = vectorizer.transform(X_val_hidden.astype(str))
    X_test_hidden_vec = vectorizer.transform(X_test_hidden.astype(str))

    # Train the model type from config
    model_type = config.get("model", {}).get("type", "logistic_regression")
    if model_type == "logistic_regression":
        module = logistic_regression
    elif model_type == "xgboost":
        module = xgboost_model
    elif model_type == "baseline":
        module = baseline_model
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = module.train(X_full_vec, y_full, X_full_vec[:1], y_full[:1], config)

    # Predict
    y_val_pred = module.predict(model, X_val_hidden_vec)
    y_test_pred = module.predict(model, X_test_hidden_vec)

    # Map to "True" / "False" strings (course requirement)
    def to_submission_line(b: Any) -> str:
        return "True" if b else "False"

    val_lines = [to_submission_line(b) for b in y_val_pred]
    test_lines = [to_submission_line(b) for b in y_test_pred]

    # Output files: submissions/ with model type + timestamp in filenames
    base_submissions = resolve_path_from_config(config, "paths", "submissions_dir")
    os.makedirs(base_submissions, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    val_path = os.path.join(base_submissions, f"validation_predictions_{model_type}_{timestamp}.txt")
    test_path = os.path.join(base_submissions, f"test_predictions_{model_type}_{timestamp}.txt")
    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val_lines) + "\n")
    with open(test_path, "w", encoding="utf-8") as f:
        f.write("\n".join(test_lines) + "\n")

    print(f"Submission written to: {base_submissions}")
    print(f"  - {val_path} ({len(val_lines)} lines)")
    print(f"  - {test_path} ({len(test_lines)} lines)")
    return base_submissions


if __name__ == "__main__":
    run_baselines()

