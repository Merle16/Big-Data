from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd

from src.utils.config import resolve_path_from_config


def _load_csv(config: Dict[str, Any], file_key: str, has_labels: bool = True) -> Tuple[pd.Series, pd.Series] | pd.Series:
    data_cfg = config.get("data", {})
    text_col = data_cfg.get("text_column", "review")
    label_col = data_cfg.get("label_column", "label")

    csv_path = resolve_path_from_config(config, "data", file_key)
    df = pd.read_csv(csv_path)

    if has_labels:
        if text_col not in df.columns or label_col not in df.columns:
            raise ValueError(f"Expected columns '{text_col}' and '{label_col}' in {csv_path}, found {df.columns.tolist()}")
        return df[text_col], df[label_col]

    if text_col not in df.columns:
        raise ValueError(f"Expected text column '{text_col}' in {csv_path}, found {df.columns.tolist()}")
    return df[text_col]


def load_train_data(config: Dict[str, Any]) -> Tuple[pd.Series, pd.Series]:
    """
    Load labeled training data as (texts, labels).
    """
    return _load_csv(config, "train_file", has_labels=True)  # type: ignore[return-value]


def load_validation_data(config: Dict[str, Any]) -> Tuple[pd.Series, pd.Series]:
    """
    Load labeled validation data as (texts, labels).
    """
    return _load_csv(config, "validation_file", has_labels=True)  # type: ignore[return-value]


def load_test_data(config: Dict[str, Any]) -> Tuple[pd.Series, pd.Series]:
    """
    Load labeled test data (if labels are available) as (texts, labels).
    """
    return _load_csv(config, "test_file", has_labels=True)  # type: ignore[return-value]


def load_validation_hidden(config: Dict[str, Any]) -> pd.Series:
    """
    Load validation_hidden.csv (no labels).
    """
    return _load_csv(config, "validation_hidden_file", has_labels=False)  # type: ignore[return-value]


def load_test_hidden(config: Dict[str, Any]) -> pd.Series:
    """
    Load test_hidden.csv (no labels).
    """
    return _load_csv(config, "test_hidden_file", has_labels=False)  # type: ignore[return-value]

