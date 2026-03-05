from __future__ import annotations

from src.utils.config import load_config
from src.data.dataloaders import load_train_data, load_validation_data
from src.models import logistic_regression, xgboost_model, baseline_model

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score


def run_baselines() -> None:
    """
    Simple shared baseline:
    - loads train/validation data via shared dataloaders
    - builds a basic bag-of-words representation
    - trains/evaluates logistic regression, XGBoost and a baseline model
    """
    config = load_config()

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

