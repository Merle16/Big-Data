#!/usr/bin/env python3
"""Pipeline orchestrator.

Usage
-----
  python pipeline/run.py                               # full pipeline
  python pipeline/run.py --from features              # skip cleaning, use existing parquets
  python pipeline/run.py --from models                # skip cleaning + features
  python pipeline/run.py --only models                # just retrain
  python pipeline/run.py --only report                # just regenerate HTML
  python pipeline/run.py --stages features models     # explicit list (canonical order enforced)
  python pipeline/run.py --external-dataset           # run genre enrichment (default path)
  python pipeline/run.py --external-dataset /p/Movies_by_Genre  # custom genre folder
  python pipeline/run.py --from models --param-search # hyperparam search (not yet wired)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `pipeline.*` importable when run as `python pipeline/run.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_STAGES      = ("cleaning", "features", "models", "report")
_DEFAULT_GENRE_DIR = Path(__file__).resolve().parents[1] / "data" / "Movies_by_Genre"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the IMDB movie-hit prediction pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--from",
        dest="from_stage",
        choices=_STAGES,
        metavar="STAGE",
        help="Start from this stage (skip earlier ones).",
    )
    group.add_argument(
        "--only",
        dest="only_stage",
        choices=_STAGES,
        metavar="STAGE",
        help="Run only this single stage.",
    )
    group.add_argument(
        "--stages",
        nargs="+",
        choices=_STAGES,
        metavar="STAGE",
        help="Explicit list of stages to run (canonical order enforced).",
    )
    p.add_argument(
        "--external-dataset",
        dest="external_dataset",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help=(
            "Enable genre enrichment from Movies_by_Genre. "
            "Omit PATH to use the default (data/Movies_by_Genre/). "
            "Runs after cleaning and before features."
        ),
    )
    p.add_argument(
        "--param-search",
        dest="param_search",
        action="store_true",
        help="Run hyperparameter search in the models stage (placeholder).",
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


def _run_cleaning() -> None:
    from pipeline.data_cleaning import run_pipeline
    paths = run_pipeline()
    print("\n[run] cleaning complete.")
    for split, path in paths.items():
        print(f"  {split}: {path}")


def _run_enrichment(state: dict, genre_dir: Path) -> dict:
    from pipeline.enrich_genre import run as _enrich
    state = _enrich(state, genre_dir=genre_dir)
    print("\n[run] genre enrichment complete.")
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
    from pipeline import make_full_report
    out = make_full_report.run()
    print(f"\n[run] report written → {out}")


def main() -> None:
    args   = _parse_args()
    stages = _resolve_stages(args)

    genre_dir: Path | None = None
    if args.external_dataset is not None:
        genre_dir = Path(args.external_dataset) if args.external_dataset else _DEFAULT_GENRE_DIR
        print(f"[run] genre enrichment enabled  ({genre_dir})\n")

    print(f"[run] stages: {' → '.join(stages)}\n")

    state: dict = {}
    for stage in stages:
        print("=" * 60)
        print(f"[run] ── {stage.upper()} ──")
        print("=" * 60)

        if stage == "cleaning":
            _run_cleaning()
            if genre_dir is not None:
                print("=" * 60)
                print("[run] ── ENRICHMENT ──")
                print("=" * 60)
                state = _run_enrichment(state, genre_dir)

        elif stage == "features":
            state = _run_features()

        elif stage == "models":
            state = _run_models(state or None, param_search=args.param_search)

        elif stage == "report":
            _run_report()

    print(f"\n[run] done.  Completed: {', '.join(stages)}")


if __name__ == "__main__":
    main()
