from __future__ import annotations

import glob
import os
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.data.dataloaders import load_train_data, load_validation_data
from src.models import baseline_model, logistic_regression, xgboost_model
from src.utils.config import load_config, resolve_path_from_config


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

    # 2) Load movie_directors and movie_writers and aggregate to tokens per tconst
    directors_path = os.path.join(csv_dir, "movie_directors.csv")
    writers_path = os.path.join(csv_dir, "movie_writers.csv")

    director_tokens = None
    writer_tokens = None

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

    if director_tokens is not None:
        train_df = train_df.merge(director_tokens, on="tconst", how="left")
    else:
        train_df["director_tokens"] = ""

    if writer_tokens is not None:
        train_df = train_df.merge(writer_tokens, on="tconst", how="left")
    else:
        train_df["writer_tokens"] = ""

    # 3) Build a simple text field from the movie title plus director/writer tokens
    #    (there is no 'review' column in these CSVs).
    title_col = "primaryTitle" if "primaryTitle" in train_df.columns else None
    if title_col is None:
        raise ValueError(
            f"Expected 'primaryTitle' column in train CSVs, found {train_df.columns.tolist()}"
        )

    base_text = train_df[title_col].fillna("").astype(str)
    enriched_text = (
        base_text
        + " "
        + train_df["director_tokens"].fillna("")
        + " "
        + train_df["writer_tokens"].fillna("")
    ).str.strip()

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


def run_baselines() -> None:
    """
    Simple shared baseline:
    - prefers using raw CSV chunks in data/raw/csv (train-*.csv + movie_* files)
    - falls back to the original config-based CSVs if those are not present
    - builds a basic bag-of-words representation
    - trains/evaluates logistic regression, XGBoost and a baseline model
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

    results = {}
    for name, module in models.items():
        print(f"Training {name}...")
        model = module.train(X_train_vec, y_train, X_val_vec, y_val, config)
        y_val_pred = module.predict(model, X_val_vec)
        acc = accuracy_score(y_val, y_val_pred)
        results[name] = acc
        print(f"Validation accuracy ({name}): {acc:.4f}\n")

    print("Summary:", results)


if __name__ == "__main__":
    run_baselines()

