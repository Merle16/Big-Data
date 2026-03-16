#!/usr/bin/env python3
"""
pipeline/models — Model training, diagnostics, reduction, and export.

Orchestrates the four model steps in sequence:
  m1_train        → train Logistic Regression & XGBoost
  m2_diagnostics  → permutation importance, SHAP, calibration, overfitting check
  m3_reduced_model→ ablation curve, reduced feature set, retrain
  m4_export       → predictions, manifest, summary figures

Public API
----------
run_model_pipeline(state=None) -> dict
    Execute all four steps in order, threading state through each.
    Returns the final accumulated state dict.

SEED = 42
    Shared random seed used by all steps.
"""
from __future__ import annotations

from . import m1_train, m2_diagnostics, m3_reduced_model, m4_export

SEED: int = 42


def run_model_pipeline(state: dict | None = None) -> dict:
    """
    Run the full model pipeline: m1 → m2 → m3 → m4.

    Parameters
    ----------
    state : dict or None
        Optional initial state dict.  Pass None (or omit) to start fresh.
        Typically should contain ``features_train_prepped`` (pd.DataFrame)
        produced by the feature-engineering pipeline.

    Returns
    -------
    dict
        Final state dict carrying all models, metrics, and DataFrames
        produced during the pipeline run.
    """
    if state is None:
        state = {}

    print("[model_pipeline] ── Step 1/4: m1_train ──────────────────────────")
    state = m1_train.run(state)

    print("[model_pipeline] ── Step 2/4: m2_diagnostics ───────────────────")
    state = m2_diagnostics.run(state)

    print("[model_pipeline] ── Step 3/4: m3_reduced_model ─────────────────")
    state = m3_reduced_model.run(state)

    print("[model_pipeline] ── Step 4/4: m4_export ────────────────────────")
    state = m4_export.run(state)

    print("[model_pipeline] All steps complete.")
    return state


__all__ = ["run_model_pipeline", "SEED", "m1_train", "m2_diagnostics",
           "m3_reduced_model", "m4_export"]
