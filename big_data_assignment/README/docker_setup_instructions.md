# Docker setup for the IMDB project

### From the root repository:

0. Start Docker desktop and confirm it is on and running (If you have it installed, simply open the application)

1. Go to the folder location where this repo is located
- something like `cd your\own\ocation\to\...\big_data_assignment` 

2. Open a local terminal and run the following to build the docker:
  - `docker build -t big-data-imdb .`   

  (!) including the dot 

  (!) Make sure there is enough disk space

3. After last command has ran (might take a while) run terminal inside the container with the local data/:

  - `docker run --rm -it -v "$(pwd)/data:/app/data" big-data-imdb bash` 

  if it does not work, try:
  - `docker run --rm -it -v "$PWD/data:/app/data" big-data-imdb bash`

You should see it running in your Docker desktop application

### Optional: running Jupyter inside Docker instead of a local virtual environment:
  - `docker run --rm -it -p 8888:8888 -v "$(pwd)/app:/app" big-data-imdb bash`


.

.

.

.

.

.

.

Everything below explains the details of this setup.

- **Image base**: `python:3.11-slim` with Java (`openjdk-17-jre-headless`) for Spark and basic tools.
- **Project layout in the image** (all under `/app`):
  - `requirements.txt`
  - `config/` (including `config/config.yaml`)
  - `src/`
  - `data/`
- **Python path**: `PYTHONPATH=/app/src`, so you can run modules like `python -m src.pipeline.train`.

### Build the image

- **From the repo root**:
  - `cd big_data_assignment`
  - `docker build -t big-data-imdb .`

### Run a container for interactive work

- **Minimal run with data mounted**:
  - `docker run --rm -it -v "$(pwd)/data:/app/data" big-data-imdb bash`
- This opens a shell at `/app` inside the container so you can:
  - Inspect data under `/app/data`
  - Run Python modules, e.g.:
    - `python -m src.pipeline.train`
    - `python -m src.pipeline.predict`

### Volumes and persistence

- To keep models, logs, and artifacts on the host, mount extra folders:
  - `docker run --rm -it \`
  - `  -v "$(pwd)/data:/app/data" \`
  - `  -v "$(pwd)/models:/app/models" \`
  - `  -v "$(pwd)/artifacts:/app/artifacts" \`
  - `  -v "$(pwd)/logs:/app/logs" \`
  - `  big-data-imdb bash`
- These paths line up with the defaults in `config/config.yaml`:
  - `paths.models_dir`, `paths.artifacts_dir`, `paths.logs_dir`.

### How config is picked up automatically in Docker

- The image copies `config/config.yaml` into `/app/config/config.yaml`.
- `src/utils/config.py`:
  - Computes the project root from its own location.
  - Looks for `config/config.yaml` under that root.
- Because the container’s working directory is `/app` and `PYTHONPATH=/app/src`, any code that imports `load_config()` will automatically read `/app/config/config.yaml`:
  - `from src.utils.config import load_config`
  - `config = load_config()`
- No extra environment variables, CLI flags, or hard-coded paths are needed; the same config file drives:
  - Data paths (e.g. `data.raw_dir`, `data.train_file`, etc.)
  - Feature settings
  - Model selection and hyperparameters
  - Output folders (`paths.models_dir`, `paths.artifacts_dir`, `paths.logs_dir`)

### Typical Docker workflow

- **1. Prepare config and data on the host**
  - Edit `config/config.yaml` as needed.
  - Place raw/processed data under `data/` to match the config paths.

- **2. Build or rebuild the image**
  - `docker build -t big-data-imdb .`

- **3. Start a container**
  - With data and outputs mounted as volumes (see above).

- **4. Run pipelines inside the container**
  - Training scaffold:
    - `python -m src.pipeline.train`
  - Prediction scaffold:
    - `python -m src.pipeline.predict`

- **5. Inspect results on the host**
  - Models, logs, and artifacts are written into the mounted host folders, so you can open them directly in your IDE.

