"""
feature_engineering — Modular Feature Pipeline
===============================================
Orchestrates three sequential steps:

  f1  Candidate feature generation
      Reads  : data/processed/train_clean.parquet
      Writes : data/processed/features_train.parquet + .csv
               data/processed/feature_figures/01–06_*.png

  f2  Feature selection & matrix preparation
      Reads  : data/processed/features_train.parquet
      Writes : data/processed/features_train_prepped.parquet + .csv
               data/processed/feature_figures/07–11_*.png

  f3  Feature quality diagnostics
      Reads  : data/processed/features_train_prepped.parquet
      Writes : data/processed/feature_goodness.csv
               data/processed/feature_figures/12–16_*.png

Public API
----------
  run_feature_pipeline(out_dir=None) -> dict
      Runs f1 → f2 → f3 and returns the final state dict.

  FEATURE_MOTIVATION    — per-feature motivation strings (from f1)
  FEATURE_GROUPS        — grouping of features by type (from f1)
  DISPOSITION_REGISTRY  — per-feature action/imputation/scaling registry (from f2)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import f1_candidate_features as f1
from . import f2_feature_selection as f2
from . import f3_feature_quality as f3

# Re-export key registries so callers can import them from the package directly
FEATURE_MOTIVATION   = f1.FEATURE_MOTIVATION
FEATURE_GROUPS       = f1.FEATURE_GROUPS
DISPOSITION_REGISTRY = f2.DISPOSITION_REGISTRY


def run_feature_pipeline(out_dir: Optional[Path] = None) -> dict:
    """
    Run the complete feature engineering pipeline: f1 → f2 → f3.

    Parameters
    ----------
    out_dir : Path, optional
        Override the default output directory (data/processed/).
        When provided, the module-level PROC paths in f1/f2/f3 are NOT patched;
        this parameter is reserved for future use and currently informational only.

    Returns
    -------
    state : dict
        Final state dict containing all intermediate and final outputs:
          train_feat              — feature-engineered DataFrame (f1 output)
          dir_lookup              — director OOF lookup (f1)
          dir_gm                  — director global mean (f1)
          wr_lookup               — writer OOF lookup (f1)
          wr_gm                   — writer global mean (f1)
          features_train_prepped  — imputed/capped feature matrix (f2 output)
          final_feat_cols         — list of retained feature columns (f2)
          cap_bounds              — quantile cap bounds fitted on train (f2)
          medians                 — imputation medians fitted on train (f2)
          feature_goodness        — per-feature quality diagnostics DataFrame (f3)
          feat_cols_quality       — list of evaluated feature columns (f3)
    """
    state: dict = {}

    print("=" * 60)
    print("Feature Engineering Pipeline — Step f1: Candidate Features")
    print("=" * 60)
    state = f1.run(state)

    print("=" * 60)
    print("Feature Engineering Pipeline — Step f2: Feature Selection")
    print("=" * 60)
    state = f2.run(state)

    print("=" * 60)
    print("Feature Engineering Pipeline — Step f3: Feature Quality")
    print("=" * 60)
    state = f3.run(state)

    print("=" * 60)
    print("Feature Engineering Pipeline — Complete")
    print("=" * 60)
    return state


__all__ = [
    "run_feature_pipeline",
    "FEATURE_MOTIVATION",
    "FEATURE_GROUPS",
    "DISPOSITION_REGISTRY",
    "f1",
    "f2",
    "f3",
]
