## IMDB project repo layout

- **`config/`**: central YAML config (`config.yaml`) that controls data paths, feature settings, model type/hyper‑params, and output folders.
- **`src/`**: reusable project code.
  - **`src/utils/config.py`**: `load_config()` reads `config/config.yaml` automatically; `resolve_path_from_config()` turns config entries (e.g. `data.raw_dir`) into absolute paths under the project root.
  - **`src/data/dataloaders.py`**: shared loaders (`load_train_data`, `load_validation_data`, `load_test_data`, `load_validation_hidden`, `load_test_hidden`) that all take a `config` dict.
  - **`src/data/json_to_tabular.py`**: shared JSON → CSV helpers (`convert_json_to_tabular`, `convert_all_from_config`) driven by the `json_conversion` section in the config.
  - **`src/pipeline/train.py`**: entrypoint `main()` that calls `load_config()` and `train_from_config(config)` (team‑shared training pipeline scaffold).
  - **`src/pipeline/predict.py`**: entrypoint `main()` that calls `load_config()` and `predict_from_config(config)` (team‑shared prediction scaffold).
  - **`src/models/`**: place model implementations used by the shared pipeline (e.g. `logistic_regression`, `xgboost`, `baseline`).
- **`data/`**: IMDB data (`raw/`, `processed/`) and CSVs referenced from `config/config.yaml`.
- **`members/`**: one folder per member for notebooks and experimental scripts that build on the shared loaders/config.

### How config wiring works

- **Central config**: edit `config/config.yaml`; no extra flags or env vars are needed.
- **Automatic loading**: any script can do `from src.utils.config import load_config` and call `config = load_config()`.
- **Shared I/O helpers**: pass that `config` into the reusable helpers in `src/data/dataloaders.py` and `src/data/json_to_tabular.py` so all team members use the same paths and schema.
- **Paths stay consistent**: `resolve_path_from_config()` always resolves paths relative to the project root (inside or outside Docker), so the same config works locally and in containers.

### Docker docs

- **Docker overview and commands** live in `READMEs/README_docker.md`.

