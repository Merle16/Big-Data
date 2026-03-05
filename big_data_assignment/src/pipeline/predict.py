from __future__ import annotations

from typing import Any, Dict

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


def predict_from_config(config: Dict[str, Any]) -> None:
    # Intentionally left without a shared feature/cleaning pipeline.
    # Once the team agrees on a final pipeline, this function can be
    # implemented to load the chosen model and generate submission files.
    raise NotImplementedError(
        "Shared prediction pipeline is not implemented yet. "
        "Use your member-specific code to generate predictions "
        "until a final pipeline is agreed upon."
    )


def main() -> None:
    config = load_config()
    predict_from_config(config)


if __name__ == "__main__":
    main()

