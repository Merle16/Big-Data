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

Usage
-----
  python pipeline/run.py                              # full pipeline, no enrichment
  python pipeline/run.py --genre --rt-oscar           # with both enrichments
  python pipeline/run.py --genre /custom/path         # custom genre folder
  python pipeline/run.py --from features              # skip cleaning, use existing parquets
  python pipeline/run.py --from models                # skip cleaning + features
  python pipeline/run.py --only models                # just retrain
  python pipeline/run.py --only report                # just regenerate HTML
  python pipeline/run.py --stages features models     # explicit stage list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    print("\n[run] cleaning complete.")
    for split, path in paths.items():
        print(f"  {split}: {path}")


def _run_genre_enrichment(state: dict, genre_dir: Path) -> dict:
    from pipeline.enrichment.genre import run as _enrich
    state = _enrich(state, genre_dir=genre_dir)
    print("\n[run] genre enrichment complete.")
    return state


def _run_rt_oscar_enrichment(state: dict) -> dict:
    from pipeline.enrichment.rt_oscar import run as _enrich_rt
    state = _enrich_rt(state)
    print("\n[run] RT + Oscar enrichment complete.")
    return state


def _run_features() -> dict:
    from pipeline.feature_engineering import run_feature_pipeline
    state = run_feature_pipeline()
    print("\n[run] features complete.")
    return state


def _run_models(state: dict | None = None, *, param_search: bool = False) -> dict:
    from pipeline.models import run_model_pipeline
    if param_search:
        print("[run] --param-search: not yet implemented — running standard training.")
    state = run_model_pipeline(state)
    print("\n[run] models complete.")
    return state


def _run_report() -> None:
    from pipeline.reporting import make_full_report
    out = make_full_report.run()
    print(f"\n[run] report written → {out}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _section(label: str) -> None:
    print("=" * 60)
    print(f"[run] ── {label} ──")
    print("=" * 60)


# ── Orchestrator ──────────────────────────────────────────────────────────────

def main() -> None:
    args        = _parse_args()
    stages      = _resolve_stages(args)
    enrichments = _resolve_enrichments(args)

    print(f"[run] stages:      {' → '.join(stages)}")
    if enrichments:
        print(f"[run] enrichments: {' + '.join(e[0] for e in enrichments)}")
    print()

    state: dict = {}

    for stage in stages:
        _section(stage.upper())

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

        elif stage == "models":
            state = _run_models(state or None, param_search=args.param_search)

        elif stage == "report":
            _run_report()

    print(f"\n[run] done.  Completed: {', '.join(stages)}")


if __name__ == "__main__":
    main()
