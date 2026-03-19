#!/usr/bin/env python3
"""
temporal_holdout_eval.py — Temporal Train/Holdout Evaluation + Final Submission
================================================================================

Uses the pipeline's already-computed feature parquets — no raw CSV reading.

Design rationale
----------------
A random 80/20 internal split inflates AUC because directors/writers seen in
the internal val set are also in training. Using startYear ≤ 2013 as train and
> 2013 as temporal holdout gives a true out-of-distribution estimate that
correlates with leaderboard performance.

Youden's J threshold is calibrated on the temporal holdout (hit rate ~41%),
NOT on the training distribution (hit rate ~53%). This prevents over-prediction
of True labels on the leaderboard.

Run from the project root:
    python pipeline/temporal_holdout_eval.py
    python pipeline/temporal_holdout_eval.py --tune
    python pipeline/temporal_holdout_eval.py --tune --n-trials 40
    python pipeline/temporal_holdout_eval.py --threshold 0.55

Flags
-----
  --tune          Random hyperparameter search scored on temporal holdout AUC.
  --n-trials N    Number of param combinations to try (default: 30).
  --threshold T   Override Youden's J with a fixed threshold.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[1]   # big_data_assignment/
PIPE_OUT     = ROOT / "pipeline" / "outputs"

TRAIN_FEAT   = PIPE_OUT / "features" / "features_train_prepped.parquet"
TEST_FEAT    = PIPE_OUT / "features" / "features_test.parquet"

SUBMISSIONS  = ROOT / "submissions"
SUBMISSIONS.mkdir(parents=True, exist_ok=True)

TEMPORAL_CUTOFF = 2013
SEED            = 42

PARAM_GRID = {
    "n_estimators":     [200, 300, 500, 700, 1000],
    "max_depth":        [3, 4, 5, 6, 7],
    "learning_rate":    [0.01, 0.03, 0.05, 0.10],
    "subsample":        [0.70, 0.80, 0.85, 0.90],
    "colsample_bytree": [0.60, 0.75, 0.85, 0.90],
    "min_child_weight": [1, 2, 3, 5],
    "reg_lambda":       [0.5, 1.0, 2.0],
}

DEFAULT_PARAMS = {
    "n_estimators": 500, "max_depth": 5, "learning_rate": 0.05,
    "subsample": 0.85, "colsample_bytree": 0.85,
    "min_child_weight": 1, "reg_lambda": 1.0,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def youden_threshold(y_true, probs):
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j = tpr - fpr
    return float(thresholds[j.argmax()])


def make_model(params: dict) -> XGBClassifier:
    return XGBClassifier(
        **params,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", n_jobs=4, random_state=SEED, verbosity=0,
    )


def load_splits():
    """Load pipeline feature parquets and do temporal split.

    RT and Oscar features are excluded from the submission model.
    Reason: rt_match_flag=0 (71% of test films) causes the model to assign
    near-zero probability regardless of other signals, because in training
    unmatched films are overwhelmingly non-hits (MNAR). This creates a
    catastrophic bimodal score distribution on the test set (80% of films
    score <0.02). The internal pipeline report can still use RT/Oscar for
    diagnostics, but the submission model uses only RT-agnostic features.
    """
    train_df = pd.read_parquet(TRAIN_FEAT)
    test_df  = pd.read_parquet(TEST_FEAT)

    # Exclude RT, Oscar, and constant features
    RT_OSCAR_PREFIX = ("rt_", "oscar_")
    skip = {"tconst", "label", "primaryTitle", "canonical_title"}
    feat_cols = [c for c in train_df.columns if c not in skip
                 and not any(c.startswith(p) for p in RT_OSCAR_PREFIX)
                 and train_df[c].notna().sum() > 0
                 and train_df[c].std() > 1e-9]   # drop constants

    # Align test to same columns (test has no label)
    feat_cols = [c for c in feat_cols if c in test_df.columns]
    print(f"  RT/Oscar features excluded from submission model (bimodal test distribution)")
    print(f"  RT/Oscar features remain in pipeline report for diagnostics")

    print(f"  Features used: {len(feat_cols)}")

    # Temporal split on labeled data
    year = pd.to_numeric(train_df["startYear"], errors="coerce").fillna(0)
    train_mask = year <= TEMPORAL_CUTOFF
    hold_mask  = year >  TEMPORAL_CUTOFF

    X_train = train_df.loc[train_mask, feat_cols].fillna(0)
    y_train = train_df.loc[train_mask, "label"].astype(int)
    X_hold  = train_df.loc[hold_mask,  feat_cols].fillna(0)
    y_hold  = train_df.loc[hold_mask,  "label"].astype(int)
    X_all   = train_df[feat_cols].fillna(0)
    y_all   = train_df["label"].astype(int)
    X_test  = test_df[feat_cols].fillna(0)
    tconst_test = test_df["tconst"].values

    n_tr, n_ho = train_mask.sum(), hold_mask.sum()
    hr_tr = y_train.mean()
    hr_ho = y_hold.mean()
    print(f"  Train (≤{TEMPORAL_CUTOFF}): {n_tr} films, hit rate {hr_tr:.1%}")
    print(f"  Holdout (>{TEMPORAL_CUTOFF}): {n_ho} films, hit rate {hr_ho:.1%}")
    print(f"  Test (hidden): {len(X_test)} films")

    return X_train, y_train, X_hold, y_hold, X_all, y_all, X_test, tconst_test, feat_cols


def tune(X_train, y_train, X_hold, y_hold, n_trials: int) -> dict:
    rng = np.random.default_rng(SEED)
    best_auc    = -1.0
    best_params = DEFAULT_PARAMS.copy()

    print(f"\n[Tune] {n_trials} random trials scored on temporal holdout AUC …")
    for i in range(n_trials):
        params = {k: rng.choice(v).item() for k, v in PARAM_GRID.items()}
        m = make_model(params)
        m.fit(X_train, y_train, verbose=False)
        auc = float(roc_auc_score(y_hold, m.predict_proba(X_hold)[:, 1]))
        marker = " ← best" if auc > best_auc else ""
        print(f"  trial {i+1:3d}: holdout AUC={auc:.4f}{marker}  {params}")
        if auc > best_auc:
            best_auc    = auc
            best_params = params

    print(f"\n[Tune] Best holdout AUC: {best_auc:.4f}")
    print(f"[Tune] Best params: {best_params}")
    return best_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune",       action="store_true")
    parser.add_argument("--n-trials",   type=int, default=30)
    parser.add_argument("--threshold",  type=float, default=None)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print("=" * 70)
    print("  TEMPORAL HOLDOUT EVALUATION + FINAL SUBMISSION GENERATOR")
    print(f"  Run timestamp: {ts}")
    print("=" * 70)

    # ── 1. Load ────────────────────────────────────────────────────────────────
    print("\n[1] Loading pipeline features …")
    X_train, y_train, X_hold, y_hold, X_all, y_all, X_test, tconst_test, feat_cols = load_splits()

    # ── 2. Hyperparameter search (optional) ────────────────────────────────────
    if args.tune:
        params = tune(X_train, y_train, X_hold, y_hold, args.n_trials)
    else:
        params = DEFAULT_PARAMS.copy()
        print(f"\n[2] Using default params: {params}")

    # ── 3. Evaluate on temporal holdout ────────────────────────────────────────
    print("\n[3] Training on ≤2013, evaluating on >2013 …")
    model_holdout = make_model(params)
    model_holdout.fit(X_train, y_train, verbose=False)
    hold_probs = model_holdout.predict_proba(X_hold)[:, 1]
    hold_auc   = float(roc_auc_score(y_hold, hold_probs))
    youden_thr = youden_threshold(y_hold.to_numpy(), hold_probs)

    print(f"  Temporal holdout AUC : {hold_auc:.4f}")
    print(f"  Youden's J threshold : {youden_thr:.4f}")
    print(f"  (hit rate in holdout : {y_hold.mean():.1%})")

    # ── 4. Retrain on ALL labeled data ─────────────────────────────────────────
    print("\n[4] Retraining on all labeled data …")
    model_final = make_model(params)
    model_final.fit(X_all, y_all, verbose=False)

    # ── 5. Predict on hidden test ──────────────────────────────────────────────
    thr = args.threshold if args.threshold is not None else youden_thr
    print(f"\n[5] Predicting with threshold={thr:.4f} …")
    test_probs  = model_final.predict_proba(X_test)[:, 1]
    test_labels = (test_probs >= thr).astype(int)
    n_true = int(test_labels.sum())
    print(f"  True predictions: {n_true}/{len(test_labels)} ({n_true/len(test_labels):.1%})")

    # ── 6. Save submission ─────────────────────────────────────────────────────
    tag  = f"thr{thr:.2f}_auc{hold_auc:.4f}_{ts}"
    out  = SUBMISSIONS / f"submission_{tag}.csv"
    pred = pd.DataFrame({"tconst": tconst_test, "predicted_label": test_labels})
    pred.to_csv(out, index=False)
    print(f"\n[6] Submission saved → {out}")
    print(f"    True={n_true}  False={len(test_labels)-n_true}")
    print(f"\n  Holdout AUC={hold_auc:.4f}  |  Threshold={thr:.4f}  |  True={n_true}")


if __name__ == "__main__":
    main()
