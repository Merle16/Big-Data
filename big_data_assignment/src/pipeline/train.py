from __future__ import annotations

from typing import Any, Dict

from src.data.dataloaders import load_train_data, load_validation_data
from src.utils.config import load_config


def _select_model_module(model_type: str):
    if model_type == "logistic_regression":
        from src.models import logistic_regression as module
    elif model_type == "xgboost":
        from src.models import xgboost_model as module
    elif model_type == "baseline":
        from src.models import baseline_model as module
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return module


def train_from_config(config: Dict[str, Any]) -> None:
    # Intentionally left without a shared feature/cleaning pipeline.
    # Each team member should experiment with their own feature engineering
    # in their personal folder (see members/<name>/).
    _ = load_train_data(config)
    _ = load_validation_data(config)
    raise NotImplementedError(
        "Shared training pipeline does not define features yet. "
        "Use member-specific scripts (e.g. members/<name>/baseline.py) "
        "to design and run your own cleaning/feature pipelines."
    )


def main() -> None:
    config = load_config()
    train_from_config(config)


if __name__ == "__main__":
    main()

