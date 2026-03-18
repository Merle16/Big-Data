#!/usr/bin/env python3
"""
Step m1: Model training — Logistic Regression vs XGBoost.
Reads features_train_prepped, trains both models, saves models.pkl and figures.

Figures saved to data/processed/model_figures/:
  01_roc_curves.png
  02_confusion_logistic.png
  03_confusion_xgboost.png     (only if XGBoost available)
  04_model_comparison.png
  05_score_distributions.png
  06_threshold_sweep_logistic.png
  07_threshold_sweep_xgboost.png  (only if available)
  08_xgb_importance.png           (only if available)
  09_logistic_coefs.png

Artifacts saved to data/processed/:
  models.pkl
  model_results.csv
  threshold_analysis.csv
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional

log = logging.getLogger("pipeline.m1_train")

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    log.warning("XGBoost not available — falling back to logistic regression only")

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]   # big_data_assignment/
PIPELINE   = ROOT / "pipeline"
OUT_FEAT   = PIPELINE / "outputs" / "features"
OUT_MODELS = PIPELINE / "outputs" / "models"
MODEL_FIG_DIR = OUT_MODELS
OUT_MODELS.mkdir(parents=True, exist_ok=True)

# ── config ─────────────────────────────────────────────────────────────────────
try:
    from ..config import CFG as _CFG
except ImportError:
    import yaml as _yaml
    _cfg_path = ROOT / "config.yaml"
    _CFG: dict = _yaml.safe_load(_cfg_path.read_text(encoding="utf-8")) if _cfg_path.exists() else {}

# ── constants ──────────────────────────────────────────────────────────────────
SEED = _CFG.get("global", {}).get("seed", 42)
BG  = "#0a0a0a"
CRD = "#111111"
BDR = "#252525"
TXT = "#e8e8e8"
MUT = "#666666"
Y   = "#F5C518"
GRN = "#2ecc71"
RED = "#e74c3c"
ORG = "#f39c12"
BLU = "#1848f5"
B   = BLU  # legacy alias

# ── matplotlib dark theme ──────────────────────────────────────────────────────
mpl.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    CRD,
    "axes.edgecolor":    BDR,
    "axes.labelcolor":   TXT,
    "text.color":        TXT,
    "xtick.color":       MUT,
    "ytick.color":       MUT,
    "grid.color":        BDR,
    "grid.linewidth":    0.4,
    "legend.facecolor":  CRD,
    "legend.edgecolor":  BDR,
    "legend.fontsize":   9,
    "legend.framealpha": 0.15,
    "font.family":       "sans-serif",
})


# ── data helpers ───────────────────────────────────────────────────────────────

def _load_feat_df(state: dict) -> pd.DataFrame:
    feat_df = state.get("features_train_prepped")
    if feat_df is None:
        for fname in ["features_train_prepped.parquet", "features_train_prepped.csv"]:
            fp = OUT_FEAT / fname
            if fp.exists():
                feat_df = pd.read_parquet(fp) if fname.endswith(".parquet") else pd.read_csv(fp)
                break
    if feat_df is None:
        raise FileNotFoundError("features_train_prepped not found — run feature pipeline first.")
    for col in feat_df.columns:
        if col not in ("tconst", "primaryTitle", "canonical_title"):
            feat_df[col] = pd.to_numeric(feat_df[col], errors="coerce")
    return feat_df


def _prep_splits(feat_df: pd.DataFrame, feat_cols: list):
    """Three-way stratified split: 60% train / 20% val / 20% test.

    - train : used for model fitting only
    - val   : used for threshold calibration (Youden's J) and model selection
    - test  : held completely out — only used to report final honest AUC
    """
    if "label" not in feat_df.columns:
        raise ValueError("label column missing.")
    label_num = pd.to_numeric(feat_df["label"], errors="coerce")
    feat_df = feat_df.loc[label_num.notna()].copy()
    feat_df["label"] = label_num.loc[label_num.notna()].astype(int).to_numpy()
    for col in feat_cols:
        med = float(feat_df[col].median()) if feat_df[col].notna().sum() > 0 else 0.0
        feat_df[col] = feat_df[col].fillna(med)

    # Step 1: carve out 20% as held-out test (never used for any decision)
    trainval_idx, test_idx = train_test_split(
        feat_df.index,
        test_size=0.20,
        random_state=SEED,
        stratify=feat_df["label"].astype(int),
    )
    # Step 2: split the remaining 80% into 75% train / 25% val → 60% / 20% overall
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=0.25,
        random_state=SEED,
        stratify=feat_df.loc[trainval_idx, "label"].astype(int),
    )
    X_tr = feat_df.loc[train_idx,  feat_cols]
    X_vl = feat_df.loc[val_idx,    feat_cols]
    X_te = feat_df.loc[test_idx,   feat_cols]
    y_tr = feat_df.loc[train_idx,  "label"].astype(int)
    y_vl = feat_df.loc[val_idx,    "label"].astype(int)
    y_te = feat_df.loc[test_idx,   "label"].astype(int)
    return feat_df, X_tr, X_vl, X_te, y_tr, y_vl, y_te, list(train_idx), list(val_idx), list(test_idx)


# ── threshold helpers ──────────────────────────────────────────────────────────

def _youden_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple:
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j = tpr - fpr
    finite = np.isfinite(thresholds)
    if finite.any():
        idx = int(np.argmax(np.where(finite, j, -np.inf)))
    else:
        idx = int(np.argmax(j))
    threshold = thresholds[idx]
    if not np.isfinite(threshold):
        threshold = 0.5
    return float(np.clip(threshold, 0.0, 1.0)), float(j[idx])


def _threshold_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision   = tp / (tp + fp) if (tp + fp) else 0.0
    return {
        "threshold":   float(threshold),
        "accuracy":    float(accuracy_score(y_true, preds)),
        "f1":          float(f1_score(y_true, preds, zero_division=0)),
        "precision":   float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "youden_j":    float(sensitivity + specificity - 1.0),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


# ── figure functions ───────────────────────────────────────────────────────────

def _save(fig, fname: str) -> None:
    fig.savefig(MODEL_FIG_DIR / fname, dpi=130, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)


def _fig_roc(log_probs, xgb_probs, y_val) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot([0, 1], [0, 1], color=MUT, linestyle="--", linewidth=1)

    fpr, tpr, _ = roc_curve(y_val, log_probs)
    log_auc = roc_auc_score(y_val, log_probs)
    ax.fill_between(fpr, tpr, alpha=0.08, color=BLU)
    ax.plot(fpr, tpr, color=BLU, linewidth=2.5, label=f"Logistic  AUC={log_auc:.4f}")

    if xgb_probs is not None:
        fpr2, tpr2, _ = roc_curve(y_val, xgb_probs)
        xgb_auc = roc_auc_score(y_val, xgb_probs)
        ax.fill_between(fpr2, tpr2, alpha=0.08, color=Y)
        ax.plot(fpr2, tpr2, color=Y, linewidth=2.5, label=f"XGBoost   AUC={xgb_auc:.4f}")

    ax.set_xlabel("False Positive Rate", fontsize=11, labelpad=8)
    ax.set_ylabel("True Positive Rate", fontsize=11, labelpad=8)
    ax.set_title("ROC curves — validation set",
                 fontsize=13, fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    ax.tick_params(axis="x", labelsize=9, labelcolor=MUT)
    ax.tick_params(axis="y", labelsize=9, labelcolor=MUT)
    ax.legend()
    fig.tight_layout(pad=2.5)
    _save(fig, "01_roc_curves.png")


def _fig_confusion(probs, y_val, threshold=0.5, title="Confusion matrix",
                   fname="02_confusion_logistic.png") -> None:
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_val, preds)
    fig, ax = plt.subplots(figsize=(14, 5))
    cmap = ListedColormap([CRD, Y])
    im = ax.imshow(cm, cmap=cmap)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=9)
    ax.set_yticklabels(["True 0", "True 1"], fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    # Diagonal cells use the yellow colormap end (bright) → dark text;
    # off-diagonal cells use the dark colormap end → white text.
    for i in range(2):
        for j in range(2):
            cell_color = TXT if i != j else BG
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=cell_color, fontsize=14, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=8, colors=MUT)
    fig.tight_layout(pad=2.5)
    _save(fig, fname)


def _fig_score_dist(log_probs, xgb_probs, y_val) -> None:
    ncols = 2 if xgb_probs is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(16, 5) if ncols == 2 else (14, 5))
    if ncols == 1:
        axes = [axes]
    pairs = [(log_probs, "Logistic")]
    if xgb_probs is not None:
        pairs.append((xgb_probs, "XGBoost"))
    for ax, (probs, name) in zip(axes, pairs):
        ax.hist(probs[y_val == 0], bins=40, color=RED, alpha=0.35, edgecolor="none", label="label=0", density=True)
        ax.hist(probs[y_val == 1], bins=40, color=GRN, alpha=0.35, edgecolor="none", label="label=1", density=True)
        ax.axvline(0.5, color=MUT, linestyle="--", linewidth=1)
        ax.set_title(f"{name} — score distribution by class",
                     fontsize=13, fontweight="bold", color=TXT, pad=12)
        ax.title.set_position([0.5, 1.02])
        ax.legend()
        ax.tick_params(axis="x", labelsize=9, labelcolor=MUT)
        ax.tick_params(axis="y", labelsize=9, labelcolor=MUT)
    fig.tight_layout(pad=2.5)
    _save(fig, "05_score_distributions.png")


def _fig_xgb_importance(xgb_model, feat_cols: List[str]) -> None:
    booster = xgb_model.get_booster()
    score_map = booster.get_score(importance_type="gain")

    def _resolve(k):
        if k in feat_cols:
            return k
        if k.startswith("f") and k[1:].isdigit():
            idx = int(k[1:])
            if 0 <= idx < len(feat_cols):
                return feat_cols[idx]
        return k

    imp = pd.Series({_resolve(k): v for k, v in score_map.items()}).reindex(feat_cols, fill_value=0)
    imp = imp.sort_values(ascending=True).tail(20)

    n_feat = len(imp)
    fig, ax = plt.subplots(figsize=(14, max(6, n_feat * 0.55)))
    bars = ax.barh(imp.index.tolist(), imp.values, color=Y, alpha=0.88, edgecolor="none")
    x_max = ax.get_xlim()[1]
    for bar in bars:
        w = bar.get_width()
        if w > 0:
            ax.text(w + 0.005 * x_max, bar.get_y() + bar.get_height() / 2,
                    f"{w:.1f}", va="center", ha="left", fontsize=8, color=TXT)
    ax.set_xlabel("Gain importance", fontsize=11, labelpad=8)
    ax.set_title("XGBoost feature importance — gain (top 20)",
                 fontsize=13, fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    ax.tick_params(axis="y", labelsize=9, labelcolor=MUT)
    ax.tick_params(axis="x", labelsize=9, labelcolor=MUT)
    fig.tight_layout(pad=2.5)
    _save(fig, "08_xgb_importance.png")


def _fig_logistic_coefs(log_model: LogisticRegression, feat_cols: List[str]) -> None:
    coefs = pd.Series(log_model.coef_[0], index=feat_cols)
    top = coefs.abs().nlargest(20).index
    coefs_top = coefs[top].sort_values()
    colors = [GRN if v > 0 else RED for v in coefs_top.values]
    n_feat = len(coefs_top)
    fig, ax = plt.subplots(figsize=(14, max(6, n_feat * 0.55)))
    bars = ax.barh(coefs_top.index.tolist(), coefs_top.values, color=colors, alpha=0.88, edgecolor="none")
    ax.axvline(0, color=MUT, linewidth=1)
    x_max = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]))
    for bar in bars:
        w = bar.get_width()
        x_pos = w + 0.005 * x_max if w >= 0 else w - 0.005 * x_max
        ha = "left" if w >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", va="center", ha=ha, fontsize=8, color=TXT)
    ax.set_xlabel("Coefficient (standardized features)", fontsize=11, labelpad=8)
    ax.set_title("Logistic regression coefficients (top 20 by |coef|)",
                 fontsize=13, fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    ax.tick_params(axis="y", labelsize=9, labelcolor=MUT)
    ax.tick_params(axis="x", labelsize=9, labelcolor=MUT)
    fig.tight_layout(pad=2.5)
    _save(fig, "09_logistic_coefs.png")


def _fig_model_comparison(log_auc, xgb_auc, log_acc, xgb_acc) -> None:
    metrics   = ["ROC-AUC", "Accuracy"]
    log_vals  = [log_auc, log_acc]
    xgb_vals  = [xgb_auc, xgb_acc] if xgb_auc is not None else None
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(14, 5))
    bars_log = ax.bar(x - 0.2, log_vals, width=0.35, color=BLU, alpha=0.88, edgecolor="none", label="Logistic")
    if xgb_vals:
        bars_xgb = ax.bar(x + 0.2, xgb_vals, width=0.35, color=Y, alpha=0.88, edgecolor="none", label="XGBoost")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11, labelpad=8)
    ax.set_title("Model comparison — validation set",
                 fontsize=13, fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    ax.tick_params(axis="x", labelsize=9, labelcolor=MUT)
    ax.tick_params(axis="y", labelsize=9, labelcolor=MUT)
    ax.legend()
    y_max = ax.get_ylim()[1]
    for bar in bars_log:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005 * y_max,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9, color=TXT)
    if xgb_vals:
        for bar in bars_xgb:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005 * y_max,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9, color=TXT)
    fig.tight_layout(pad=2.5)
    _save(fig, "04_model_comparison.png")


def _fig_threshold_sweep(y_true: np.ndarray, probs: np.ndarray, model_name: str,
                         youden_thr: float, fname: str) -> None:
    thresholds   = np.linspace(0.01, 0.99, 99)
    sensitivity  = []
    specificity  = []
    f1_vals      = []
    youden_vals  = []
    for thr in thresholds:
        m = _threshold_metrics(y_true, probs, float(thr))
        sensitivity.append(m["sensitivity"])
        specificity.append(m["specificity"])
        f1_vals.append(m["f1"])
        youden_vals.append(m["youden_j"])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(thresholds, sensitivity,  color=GRN, linewidth=2, label="Sensitivity")
    ax.plot(thresholds, specificity,  color=ORG, linewidth=2, label="Specificity")
    ax.plot(thresholds, f1_vals,      color=BLU, linewidth=2, label="F1")
    ax.plot(thresholds, youden_vals,  color=Y,   linewidth=2, label="Youden J")
    ax.axvline(0.5, color=MUT, linestyle="--", linewidth=1.4, label="Threshold 0.5")
    ax.axvline(youden_thr, color=RED, linestyle="--", linewidth=1.4,
               label=f"Youden {youden_thr:.3f}")
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Threshold", fontsize=11, labelpad=8)
    ax.set_ylabel("Metric value", fontsize=11, labelpad=8)
    ax.set_title(f"{model_name} — threshold effect on validation metrics",
                 fontsize=13, fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    ax.tick_params(axis="x", labelsize=9, labelcolor=MUT)
    ax.tick_params(axis="y", labelsize=9, labelcolor=MUT)
    ax.legend(ncol=2, loc="lower center")
    fig.tight_layout(pad=2.5)
    _save(fig, fname)


# ── main entry point ───────────────────────────────────────────────────────────

def run(state: dict) -> dict:
    """Train Logistic Regression and XGBoost, produce figures, save artifacts."""
    feat_df = _load_feat_df(state)

    feat_cols = [c for c in feat_df.columns
                 if c not in ("tconst", "label", "primaryTitle", "canonical_title")
                 and feat_df[c].notna().sum() > 0]

    # Guard against silent schema drift when enrichment files are missing.
    # The full pipeline (--genre --rt-oscar --pagerank) produces 60+ features.
    # Fewer than 20 means at least one enrichment stage was silently skipped.
    _MIN_FEATURES = 20
    if len(feat_cols) < _MIN_FEATURES:
        raise RuntimeError(
            f"[m1_train] Only {len(feat_cols)} features found — expected at least {_MIN_FEATURES}. "
            f"Likely cause: enrichment files missing (run with --genre --rt-oscar --pagerank). "
            f"Found: {feat_cols}"
        )
    print(f"[m1_train] {len(feat_cols)} features loaded.")

    feat_df, X_tr, X_vl, X_te, y_tr, y_vl, y_te, train_idx, val_idx, test_idx = _prep_splits(feat_df, feat_cols)
    y_val_np  = y_vl.to_numpy()
    y_test_np = y_te.to_numpy()

    # ── scale for logistic ─────────────────────────────────────────────────────
    scaler = StandardScaler()
    train_Xs = scaler.fit_transform(X_tr)
    val_Xs   = scaler.transform(X_vl)
    test_Xs  = scaler.transform(X_te)

    # ── logistic regression ────────────────────────────────────────────────────
    print("[m1_train] Fitting logistic regression...")
    _lcfg     = _CFG.get("models", {}).get("logistic", {})
    log_model = LogisticRegression(
        max_iter=int(_lcfg.get("max_iter", 2000)),
        C=float(_lcfg.get("C", 1.0)),
        random_state=SEED,
    )
    log_model.fit(train_Xs, y_tr)
    log_probs      = log_model.predict_proba(val_Xs)[:, 1]
    log_probs_test = log_model.predict_proba(test_Xs)[:, 1]
    log_auc        = float(roc_auc_score(y_vl, log_probs))
    log_auc_test   = float(roc_auc_score(y_te, log_probs_test))
    log_acc        = float(accuracy_score(y_vl, (log_probs >= 0.5).astype(int)))
    log_acc_test   = float(accuracy_score(y_te, (log_probs_test >= 0.5).astype(int)))

    # ── xgboost ────────────────────────────────────────────────────────────────
    xgb_model = None; xgb_probs = None; xgb_probs_test = None
    xgb_auc = None; xgb_auc_test = None; xgb_acc = None; xgb_acc_test = None
    if HAS_XGB:
        print("[m1_train] Fitting XGBoost...")
        _xcfg     = _CFG.get("models", {}).get("xgboost", {})
        xgb_model = XGBClassifier(
            n_estimators=int(_xcfg.get("n_estimators", 500)),
            max_depth=int(_xcfg.get("max_depth", 5)),
            learning_rate=float(_xcfg.get("learning_rate", 0.05)),
            subsample=float(_xcfg.get("subsample", 0.85)),
            colsample_bytree=float(_xcfg.get("colsample_bytree", 0.85)),
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", n_jobs=4, random_state=SEED, verbosity=0,
        )
        xgb_model.fit(X_tr, y_tr, verbose=False)
        xgb_probs      = xgb_model.predict_proba(X_vl)[:, 1]
        xgb_probs_test = xgb_model.predict_proba(X_te)[:, 1]
        xgb_auc        = float(roc_auc_score(y_vl, xgb_probs))
        xgb_auc_test   = float(roc_auc_score(y_te, xgb_probs_test))
        xgb_acc        = float(accuracy_score(y_vl, (xgb_probs >= 0.5).astype(int)))
        xgb_acc_test   = float(accuracy_score(y_te, (xgb_probs_test >= 0.5).astype(int)))

    # ── threshold analysis ─────────────────────────────────────────────────────
    log_thr_youden, log_youden_j = _youden_threshold(y_val_np, log_probs)
    log_m_05     = _threshold_metrics(y_val_np, log_probs, 0.5)
    log_m_youden = _threshold_metrics(y_val_np, log_probs, log_thr_youden)

    xgb_thr_youden = xgb_youden_j = xgb_m_05 = xgb_m_youden = None
    if xgb_probs is not None:
        xgb_thr_youden, xgb_youden_j = _youden_threshold(y_val_np, xgb_probs)
        xgb_m_05     = _threshold_metrics(y_val_np, xgb_probs, 0.5)
        xgb_m_youden = _threshold_metrics(y_val_np, xgb_probs, xgb_thr_youden)

    best     = "XGBoost" if (xgb_auc is not None and xgb_auc >= log_auc) else "Logistic"
    best_auc = xgb_auc if best == "XGBoost" else log_auc

    # ── figures ────────────────────────────────────────────────────────────────
    print("[m1_train] Saving figures...")
    _fig_roc(log_probs, xgb_probs, y_val_np)
    # Val confusion (threshold calibrated here via Youden's J)
    _fig_confusion(log_probs, y_val_np,
                   title=f"Logistic — val  AUC={log_auc:.4f}",
                   fname="02_confusion_logistic.png")
    # Test confusion (held-out — no decisions made on this set)
    _fig_confusion(log_probs_test, y_test_np,
                   title=f"Logistic — test  AUC={log_auc_test:.4f}",
                   fname="02b_confusion_logistic_test.png")
    if xgb_probs is not None:
        _fig_confusion(xgb_probs, y_val_np,
                       title=f"XGBoost — val  AUC={xgb_auc:.4f}",
                       fname="03_confusion_xgboost.png")
        _fig_confusion(xgb_probs_test, y_test_np,
                       title=f"XGBoost — test  AUC={xgb_auc_test:.4f}",
                       fname="03b_confusion_xgboost_test.png")
    _fig_model_comparison(log_auc, xgb_auc, log_acc, xgb_acc)
    _fig_score_dist(log_probs, xgb_probs, y_val_np)
    _fig_threshold_sweep(y_val_np, log_probs, "Logistic", log_thr_youden,
                         "06_threshold_sweep_logistic.png")
    if xgb_probs is not None and xgb_thr_youden is not None:
        _fig_threshold_sweep(y_val_np, xgb_probs, "XGBoost", xgb_thr_youden,
                             "07_threshold_sweep_xgboost.png")
    if xgb_model is not None:
        _fig_xgb_importance(xgb_model, feat_cols)
    _fig_logistic_coefs(log_model, feat_cols)

    # ── threshold DataFrame ────────────────────────────────────────────────────
    threshold_rows = [
        {"model": "logistic", "rule": "fixed_0.5",  **log_m_05,
         "youden_threshold": log_thr_youden, "youden_j_at_optimum": log_youden_j},
        {"model": "logistic", "rule": "youden_opt", **log_m_youden,
         "youden_threshold": log_thr_youden, "youden_j_at_optimum": log_youden_j},
    ]
    if xgb_m_05 is not None and xgb_m_youden is not None:
        threshold_rows.extend([
            {"model": "xgboost", "rule": "fixed_0.5",  **xgb_m_05,
             "youden_threshold": xgb_thr_youden, "youden_j_at_optimum": xgb_youden_j},
            {"model": "xgboost", "rule": "youden_opt", **xgb_m_youden,
             "youden_threshold": xgb_thr_youden, "youden_j_at_optimum": xgb_youden_j},
        ])
    threshold_df = pd.DataFrame(threshold_rows)

    # ── model results DataFrame ────────────────────────────────────────────────
    model_results = pd.DataFrame([{
        "model": "logistic",
        "validation_auc": log_auc, "validation_accuracy": log_acc,
        "test_auc": log_auc_test,  "test_accuracy": log_acc_test,
        "threshold_fixed": 0.5, "threshold_youden": log_thr_youden,
        "accuracy_youden": log_m_youden["accuracy"],
        "f1_fixed": log_m_05["f1"], "f1_youden": log_m_youden["f1"],
        "selected": best == "Logistic",
    }])
    if xgb_auc is not None:
        model_results = pd.concat([model_results, pd.DataFrame([{
            "model": "xgboost",
            "validation_auc": xgb_auc, "validation_accuracy": xgb_acc,
            "test_auc": xgb_auc_test,  "test_accuracy": xgb_acc_test,
            "threshold_fixed": 0.5, "threshold_youden": xgb_thr_youden,
            "accuracy_youden": None if xgb_m_youden is None else xgb_m_youden["accuracy"],
            "f1_fixed":   None if xgb_m_05    is None else xgb_m_05["f1"],
            "f1_youden":  None if xgb_m_youden is None else xgb_m_youden["f1"],
            "selected": best == "XGBoost",
        }])], ignore_index=True)

    # ── save CSVs ──────────────────────────────────────────────────────────────
    model_results.to_csv(OUT_MODELS / "model_results.csv", index=False)
    threshold_df.to_csv(OUT_MODELS / "threshold_analysis.csv", index=False)

    # ── save models.pkl ────────────────────────────────────────────────────────
    with open(OUT_MODELS / "models.pkl", "wb") as f:
        pickle.dump({
            "log_model":  log_model,
            "xgb_model":  xgb_model,
            "scaler":     scaler,
            "feat_cols":  feat_cols,
            "best_model": best,
            "thresholds": {
                "logistic": {"fixed_0_5": 0.5, "youden": log_thr_youden},
                "xgboost":  None if xgb_thr_youden is None
                            else {"fixed_0_5": 0.5, "youden": xgb_thr_youden},
            },
            "train_idx": train_idx,
            "val_idx":   val_idx,
            "test_idx":  test_idx,
        }, f)

    # ── update state ───────────────────────────────────────────────────────────
    state.update({
        "log_model":          log_model,
        "xgb_model":          xgb_model,
        "scaler":             scaler,
        "feat_cols":          feat_cols,
        "best_model":         best,
        "threshold_analysis": threshold_df,
        "train_idx":          train_idx,
        "val_idx":            val_idx,
        "test_idx":           test_idx,
        "log_auc":            log_auc,
        "xgb_auc":            xgb_auc,
        "log_auc_test":       log_auc_test,
        "xgb_auc_test":       xgb_auc_test,
    })

    print(f"[m1_train] Logistic  val AUC={log_auc:.4f}  test AUC={log_auc_test:.4f}")
    print(f"[m1_train] XGBoost   val AUC={xgb_auc:.4f}  test AUC={xgb_auc_test:.4f}  Best={best}")
    print(f"[m1_train] Figures saved to {MODEL_FIG_DIR}")
    print(f"[m1_train] Artifacts saved to {OUT_MODELS}")
    return state


if __name__ == "__main__":
    run({})
    print("[m1_train] Done.")
