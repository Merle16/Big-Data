#!/usr/bin/env python3
"""Pipeline orchestrator — IMDB movie-hit prediction.

Stages
------
  cleaning   Raw CSVs → cleaned parquets (DuckDB + MICE imputation)
  features   Parquets → feature matrix  (candidate → select → quality)
  models     Features → trained models  (XGBoost + Logistic, diagnostics, export)
  report     All outputs → full_pipeline_report.html

Enrichments (opt-in, run between cleaning and features)
------
  --genre      Genre labels from Movies_by_Genre/
  --rt-oscar   Rotten Tomatoes scores + Oscar nominations

Graph features (opt-in, run between features and models)
------
  --pagerank   Collaboration-network PageRank via Spark GraphFrames (or NetworkX
               fallback). Builds a success-weighted co-credit graph from
               title_principals.csv and computes PageRank for every director,
               writer, and actor. Four features are merged into
               features_train_prepped before model training:
                 director_pagerank  — mean PageRank of the film's director(s)
                 writer_pagerank    — mean PageRank of the film's writer(s)
                 top_actor_pagerank — PageRank of the top-billed actor
                 avg_cast_pagerank  — mean PageRank of the top-3 actors
               Scores are derived from the training split only (no leakage).
               Films with no crew record receive the global-median fallback.
               Requires: pyspark>=3.5, graphframes (pip install graphframes).
               Falls back to networkx automatically if Spark is unavailable.
               Parameters are tunable in config.yaml → graph.pagerank.

Usage
-----
  python pipeline/run.py                              # full pipeline, no enrichment
  python pipeline/run.py --genre --rt-oscar           # with both enrichments
  python pipeline/run.py --genre --rt-oscar --pagerank  # all enrichments + graph
  python pipeline/run.py --pagerank                   # graph features only (needs existing feature parquets)
  python pipeline/run.py --genre /custom/path         # custom genre folder
  python pipeline/run.py --from features              # skip cleaning, use existing parquets
  python pipeline/run.py --from models                # skip cleaning + features
  python pipeline/run.py --only models                # just retrain
  python pipeline/run.py --only report                # just regenerate HTML
  python pipeline/run.py --stages features models     # explicit stage list
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("pipeline")

# Make `pipeline.*` importable when invoked as `python pipeline/run.py` from project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_STAGES            = ("cleaning", "features", "models", "report")
_DEFAULT_GENRE_DIR = Path(__file__).resolve().parents[1] / "data" / "Movies_by_Genre"


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the IMDB movie-hit prediction pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Stage selection (mutually exclusive)
    stage_group = p.add_mutually_exclusive_group()
    stage_group.add_argument(
        "--from", dest="from_stage", choices=_STAGES, metavar="STAGE",
        help="Start from this stage (skip earlier ones).",
    )
    stage_group.add_argument(
        "--only", dest="only_stage", choices=_STAGES, metavar="STAGE",
        help="Run only this single stage.",
    )
    stage_group.add_argument(
        "--stages", nargs="+", choices=_STAGES, metavar="STAGE",
        help="Explicit list of stages to run (canonical order enforced).",
    )

    # Enrichment toggles
    p.add_argument(
        "--genre", dest="genre", nargs="?", const="", default=None, metavar="PATH",
        help=(
            "Enable genre enrichment from Movies_by_Genre/. "
            "Omit PATH to use the default (data/Movies_by_Genre/)."
        ),
    )
    p.add_argument(
        "--rt-oscar", dest="rt_oscar", action="store_true", default=False,
        help="Enable Rotten Tomatoes + Oscar enrichment.",
    )
    p.add_argument(
        "--pagerank", dest="pagerank", action="store_true", default=False,
        help=(
            "Enable collaboration-network PageRank features. Computes PageRank "
            "on the IMDB co-credit graph (title_principals.csv) using Spark "
            "GraphFrames, then merges 4 features into features_train_prepped "
            "before model training. Falls back to NetworkX if Spark is unavailable. "
            "Parameters (reset_prob, max_iter, spark_cores) are set in config.yaml "
            "under graph.pagerank. Runs after the features stage."
        ),
    )

    # Misc
    p.add_argument(
        "--param-search", dest="param_search", action="store_true",
        help="Run hyperparameter search in the models stage (placeholder).",
    )

    # Keep --external-dataset as a hidden alias for --genre (backwards compat)
    p.add_argument(
        "--external-dataset", dest="genre", nargs="?", const="", default=None,
        help=argparse.SUPPRESS,
    )

    return p.parse_args()


def _resolve_stages(args: argparse.Namespace) -> list[str]:
    if args.only_stage:
        return [args.only_stage]
    if args.stages:
        return [s for s in _STAGES if s in args.stages]
    if args.from_stage:
        return list(_STAGES[_STAGES.index(args.from_stage):])
    return list(_STAGES)


def _resolve_enrichments(args: argparse.Namespace) -> list[tuple]:
    """Return ordered list of (name, *params) tuples for enabled enrichments."""
    enrichments = []
    if args.genre is not None:
        genre_dir = Path(args.genre) if args.genre else _DEFAULT_GENRE_DIR
        enrichments.append(("genre", genre_dir))
    if args.rt_oscar:
        enrichments.append(("rt_oscar",))
    return enrichments


# ── Stage runners ─────────────────────────────────────────────────────────────

def _run_cleaning() -> None:
    from pipeline.data_cleaning import run_pipeline
    paths = run_pipeline()
    log.info("cleaning complete.")
    for split, path in paths.items():
        log.info("  %s: %s", split, path)


def _run_genre_enrichment(state: dict, genre_dir: Path) -> dict:
    from pipeline.enrichment.genre import run as _enrich
    state = _enrich(state, genre_dir=genre_dir)
    log.info("genre enrichment complete.")
    return state


def _run_rt_oscar_enrichment(state: dict) -> dict:
    from pipeline.enrichment.rt_oscar import run as _enrich_rt
    state = _enrich_rt(state)
    log.info("RT + Oscar enrichment complete.")
    return state


_PR_COLS = [
    "director_pagerank",
    "writer_pagerank",
    "top_actor_pagerank",
    "avg_cast_pagerank",
]


def _merge_pagerank_into_parquets(state: dict, strict: bool = False) -> dict:
    """
    Merge pre-computed PageRank features (from pipeline/outputs/graph/) into
    the train/val/test feature parquets.  Called both after g1_pagerank.run()
    AND at the start of the models stage when --pagerank is set but the features
    stage was skipped (e.g. --from models --pagerank).

    Parameters
    ----------
    strict : bool
        When True (e.g. called from the models stage with an explicit --pagerank
        flag) raise FileNotFoundError if the graph parquets are missing instead
        of silently skipping.
    """
    import pandas as pd

    ROOT         = Path(__file__).resolve().parents[1]
    OUT_FEATURES = ROOT / "pipeline" / "outputs" / "features"
    OUT_GRAPH    = ROOT / "pipeline" / "outputs" / "graph"

    pr_train_fp = OUT_GRAPH / "features_pagerank.parquet"
    if not pr_train_fp.exists():
        msg = (
            "features_pagerank.parquet not found — skipping PageRank merge. "
            "Run with the features stage included (or drop --pagerank) to produce it."
        )
        if strict:
            raise FileNotFoundError(
                f"{pr_train_fp} not found. "
                "You requested --pagerank but graph parquets have not been computed yet. "
                "Re-run including the features stage (e.g. `python pipeline/run.py --from features --pagerank`) "
                "to generate them first."
            )
        log.warning(msg)
        return state

    pr_train       = pd.read_parquet(pr_train_fp)[["tconst"] + _PR_COLS]
    global_medians = {col: float(pr_train[col].median()) for col in _PR_COLS}

    def _merge(feat_path: Path, pr_path: Path) -> pd.DataFrame:
        feat = pd.read_parquet(feat_path)
        feat = feat.drop(columns=[c for c in _PR_COLS if c in feat.columns])
        pr   = pd.read_parquet(pr_path)[["tconst"] + _PR_COLS]
        out  = feat.merge(pr, on="tconst", how="left")
        for col in _PR_COLS:
            out[col] = out[col].fillna(global_medians[col])
        out.to_parquet(feat_path, index=False)
        return out

    merged = _merge(
        OUT_FEATURES / "features_train_prepped.parquet",
        OUT_GRAPH    / "features_pagerank.parquet",
    )
    for split in ("val", "test"):
        feat_fp = OUT_FEATURES / f"features_{split}.parquet"
        pr_fp   = OUT_GRAPH    / f"features_pagerank_{split}.parquet"
        if feat_fp.exists() and pr_fp.exists():
            _merge(feat_fp, pr_fp)

    state["pagerank_features"]      = _PR_COLS
    state["features_train_prepped"] = merged

    # Re-run feature quality so PageRank features appear in goodness heatmap,
    # AUC bar, MI bar, PSI bar, and status figures alongside the base features.
    try:
        from pipeline.feature_engineering.f3_feature_quality import run as _f3
        log.info("Re-running feature quality diagnostics to include PageRank features...")
        state = _f3(state)
    except Exception as _e:
        log.warning("Feature quality re-run failed (%s) — figures may be stale.", _e)

    return state


def _run_pagerank_enrichment(state: dict) -> dict:
    """
    Run g1_pagerank (compute PageRank scores) then merge 4 features into
    train/val/test parquets.

    What this does
    --------------
    1. Builds a success-weighted co-credit graph: every pair of people who
       worked on the same training-set film gets an edge weighted by that
       film's label (1=hit, 0=flop).
    2. Runs PageRank on that graph (Spark GraphFrames if available, else NetworkX).
    3. For each film, derives: director_pagerank, writer_pagerank,
       top_actor_pagerank, avg_cast_pagerank.
    4. Merges those 4 columns into train/val/test feature parquets so that
       the models stage picks them up automatically.

    No leakage: PageRank is computed on training labels only. Val/test films
    receive scores via a lookup — unknown crew get the global-median fallback.

    Parameters are read from config.yaml → graph.pagerank.
    """
    from pipeline.graph import g1_pagerank

    try:
        from pipeline.config import CFG as _CFG
    except ImportError:
        _CFG = {}
    _gr = _CFG.get("graph", {}).get("pagerank", {})

    g1_pagerank.run(
        reset_prob=_gr.get("reset_prob", 0.15),
        tol=_gr.get("tol", 0.01),
        max_iter=_gr.get("max_iter", 20),
        spark_cores=_gr.get("spark_cores", 4),
        force_networkx=_gr.get("force_networkx", False),
    )

    # Detect which engine was used and record in state
    try:
        import pyspark  # noqa: F401
        HAS_SPARK = True
    except ImportError:
        HAS_SPARK = False
    state["graph_engine"] = "spark" if HAS_SPARK else "networkx"

    state = _merge_pagerank_into_parquets(state)
    log.info(
        "PageRank enrichment complete — %d features merged into train/val/test parquets",
        len(_PR_COLS),
    )
    return state


def _run_features() -> dict:
    from pipeline.feature_engineering import run_feature_pipeline
    state = run_feature_pipeline()
    log.info("features complete.")
    return state


def _run_models(state: dict | None = None, *, param_search: bool = False) -> dict:
    from pipeline.models import run_model_pipeline
    if param_search:
        log.warning("--param-search: not yet implemented — running standard training.")
    state = run_model_pipeline(state)
    log.info("models complete.")
    return state


def _run_report() -> None:
    from pipeline.reporting import make_full_report
    out = make_full_report.run()
    log.info("report written → %s", out)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _section(label: str) -> None:
    log.info("=" * 60)
    log.info("── %s ──", label)
    log.info("=" * 60)


# ── Orchestrator ──────────────────────────────────────────────────────────────

def main() -> None:
    args        = _parse_args()
    stages      = _resolve_stages(args)
    enrichments = _resolve_enrichments(args)

    # ── Run ID + config hash (FIX 4) ──────────────────────────────────────────
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ROOT = Path(__file__).resolve().parents[1]
    cfg_path = ROOT / "config.yaml"
    cfg_bytes = cfg_path.read_bytes() if cfg_path.exists() else b""
    cfg_hash = hashlib.sha256(cfg_bytes).hexdigest()[:8]

    log.info("run_id=%s  config_hash=%s", run_id, cfg_hash)
    log.info("stages:      %s", " → ".join(stages))
    if enrichments:
        log.info("enrichments: %s", " + ".join(e[0] for e in enrichments))
    if args.pagerank:
        log.info("graph:       pagerank (runs after features, before models)")

    state: dict = {}

    pipeline_start = time.time()

    for stage in stages:
        _section(stage.upper())
        t0 = time.time()

        if stage == "cleaning":
            _run_cleaning()
            for enrich in enrichments:
                _section(f"ENRICHMENT · {enrich[0].upper()}")
                if enrich[0] == "genre":
                    state = _run_genre_enrichment(state, enrich[1])
                elif enrich[0] == "rt_oscar":
                    state = _run_rt_oscar_enrichment(state)

        elif stage == "features":
            state = _run_features()
            if args.pagerank:
                _section("GRAPH ENRICHMENT · PAGERANK")
                state = _run_pagerank_enrichment(state)

        elif stage == "models":
            # If --pagerank was set but the features stage didn't run (e.g. --from models),
            # the graph parquets already exist — just merge them into the feature files.
            if args.pagerank and "features" not in stages:
                _section("GRAPH ENRICHMENT · PAGERANK (merge only)")
                state = _merge_pagerank_into_parquets(state, strict=True)
            state = _run_models(state or None, param_search=args.param_search)

        elif stage == "report":
            _run_report()

        elapsed = time.time() - t0
        log.info("Stage '%s' completed in %.1fs", stage, elapsed)

    # ── Save run manifest (FIX 4 + FIX 6) ────────────────────────────────────
    import yaml as _yaml
    cfg_data: dict = _yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    pipeline_version = cfg_data.get("pipeline", {}).get("version", "unknown")

    manifest = {
        "run_id":           run_id,
        "timestamp":        datetime.datetime.now().isoformat(),
        "stages":           stages,
        "enrichments":      [e[0] for e in enrichments],
        "config_hash":      cfg_hash,
        "pipeline_version": pipeline_version,
        "python_version":   sys.version,
        "graph_engine":     state.get("graph_engine", None),
    }

    out_dir = ROOT / "pipeline" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("run_manifest written → %s", manifest_path)

    total_elapsed = time.time() - pipeline_start
    log.info(
        "Pipeline done in %.1fs — stages: %s",
        total_elapsed,
        ", ".join(stages),
    )


if __name__ == "__main__":
    main()
