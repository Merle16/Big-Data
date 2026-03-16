#!/usr/bin/env python3
"""
Step m2: Model diagnostics — permutation AUC drop, SHAP, calibration, overfitting check.

Figures saved to data/processed/model_figures/:
  10_perm_auc_drop.png
  11_shap_importance.png   (only if XGBoost available)
  12_calibration_curve.png
  13_auc_gap.png
  14_diagnostic_scatter.png

Artifacts saved to data/processed/:
  feature_diagnostics.csv
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    import xgboost as xgb_lib
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]   # big_data_assignment/
PIPELINE   = ROOT / "pipeline"
OUT_FEAT   = PIPELINE / "outputs" / "features"
OUT_MODELS = PIPELINE / "outputs" / "models"
MODEL_FIG_DIR = OUT_MODELS
OUT_MODELS.mkdir(parents=True, exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────────
SEED = 42
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
    if "label" not in feat_df.columns:
        raise ValueError("label column missing.")
    label_num = pd.to_numeric(feat_df["label"], errors="coerce")
    feat_df = feat_df.loc[label_num.notna()].copy()
    feat_df["label"] = label_num.loc[label_num.notna()].astype(int).to_numpy()
    for col in feat_cols:
        med = float(feat_df[col].median()) if feat_df[col].notna().sum() > 0 else 0.0
        feat_df[col] = feat_df[col].fillna(med)
    train_idx, val_idx = train_test_split(
        feat_df.index, test_size=0.20, random_state=SEED,
        stratify=feat_df["label"].astype(int),
    )
    X_tr = feat_df.loc[train_idx, feat_cols]
    X_vl = feat_df.loc[val_idx, feat_cols]
    y_tr = feat_df.loc[train_idx, "label"].astype(int)
    y_vl = feat_df.loc[val_idx, "label"].astype(int)
    return feat_df, X_tr, X_vl, y_tr, y_vl, list(train_idx), list(val_idx)


# ── diagnostic computation ─────────────────────────────────────────────────────

def _predict_proba(model_name: str, model_obj, X, scaler=None) -> np.ndarray:
    """Unified predict_proba: XGBoost skips scaler, logistic applies it."""
    if model_name == "xgboost":
        return model_obj.predict_proba(X)[:, 1]
    return model_obj.predict_proba(scaler.transform(X))[:, 1]


def compute_permutation_drop(model_name: str, model_obj, X_val: pd.DataFrame,
                              y_val: pd.Series, feat_cols: list,
                              scaler=None) -> pd.DataFrame:
    """Shuffle each feature independently; measure AUC drop vs baseline."""
    y = y_val.astype(int).to_numpy()
    base_probs   = _predict_proba(model_name, model_obj, X_val[feat_cols], scaler)
    baseline_auc = float(roc_auc_score(y, base_probs))
    rng  = np.random.default_rng(SEED)
    rows = []
    for feat in feat_cols:
        shuffled = X_val[feat_cols].copy()
        shuffled[feat] = rng.permutation(shuffled[feat].to_numpy())
        perm_probs = _predict_proba(model_name, model_obj, shuffled, scaler)
        perm_auc   = float(roc_auc_score(y, perm_probs))
        rows.append({
            "feature":      feat,
            "baseline_auc": baseline_auc,
            "permuted_auc": perm_auc,
            "perm_auc_drop": baseline_auc - perm_auc,
        })
    return (pd.DataFrame(rows)
              .sort_values("perm_auc_drop", ascending=False)
              .reset_index(drop=True))


def compute_shap(xgb_model, X_val: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    """Compute mean absolute SHAP values via XGBoost's built-in contribs."""
    if xgb_model is None or not HAS_XGB:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])
    booster  = xgb_model.get_booster()
    dval     = xgb_lib.DMatrix(X_val[feat_cols], feature_names=feat_cols)
    contrib  = booster.predict(dval, pred_contribs=True)
    mean_abs = np.abs(contrib[:, :-1]).mean(axis=0)
    return (pd.DataFrame({"feature": feat_cols, "mean_abs_shap": mean_abs})
              .sort_values("mean_abs_shap", ascending=False))


# ── save helper ────────────────────────────────────────────────────────────────

def _save(fig, fname: str) -> None:
    fig.savefig(MODEL_FIG_DIR / fname, dpi=130, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)


# ── figure functions ───────────────────────────────────────────────────────────

def _fig_perm_drop(perm_df: pd.DataFrame, model_name: str) -> None:
    top    = perm_df.head(20).sort_values("perm_auc_drop", ascending=True)
    colors = [GRN if v > 0.005 else ORG if v > 0 else RED for v in top["perm_auc_drop"]]
    n_feat = len(top)
    fig, ax = plt.subplots(figsize=(14, max(6, n_feat * 0.55)))
    bars = ax.barh(top["feature"], top["perm_auc_drop"], color=colors, alpha=0.88, edgecolor="none")
    ax.axvline(0,     color=MUT, linewidth=1)
    ax.axvline(0.002, color=GRN, linestyle="--", linewidth=1)
    x_max = ax.get_xlim()[1]
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.001 * x_max, bar.get_y() + bar.get_height() / 2,
                f"{w:.4f}", va="center", ha="left", fontsize=8, color=TXT)
    if len(top) > 0:
        ax.text(0.003, -1, "▶ 0.002 threshold", color=GRN, fontsize=7)
    ax.set_xlabel("AUC drop when feature is shuffled", fontsize=11, labelpad=8)
    ax.set_title(f"Permutation AUC drop — {model_name} (top 20)",
                 fontsize=13, fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    ax.tick_params(axis="y", labelsize=9, labelcolor=MUT)
    ax.tick_params(axis="x", labelsize=9, labelcolor=MUT)
    fig.tight_layout(pad=2.5)
    _save(fig, "10_perm_auc_drop.png")


def _fig_shap(shap_df: pd.DataFrame) -> None:
    if shap_df.empty:
        print("[m2_diagnostics] SHAP skipped — XGBoost not available.")
        return
    top = shap_df.head(20).sort_values("mean_abs_shap", ascending=True)
    n_feat = len(top)
    fig, ax = plt.subplots(figsize=(14, max(6, n_feat * 0.55)))
    bars = ax.barh(top["feature"], top["mean_abs_shap"], color=Y, alpha=0.88, edgecolor="none")
    x_max = ax.get_xlim()[1]
    for bar in bars:
        w = bar.get_width()
        if w > 0:
            ax.text(w + 0.005 * x_max, bar.get_y() + bar.get_height() / 2,
                    f"{w:.4f}", va="center", ha="left", fontsize=8, color=TXT)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11, labelpad=8)
    ax.set_title("XGBoost SHAP importance (top 20 features)",
                 fontsize=13, fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    ax.tick_params(axis="y", labelsize=9, labelcolor=MUT)
    ax.tick_params(axis="x", labelsize=9, labelcolor=MUT)
    fig.tight_layout(pad=2.5)
    _save(fig, "11_shap_importance.png")


def _fig_calibration(log_probs, xgb_probs, y_val) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot([0, 1], [0, 1], color=MUT, linestyle="--", linewidth=1,
            label="Perfect calibration")

    frac_pos, mean_pred = calibration_curve(y_val, log_probs, n_bins=10)
    ax.plot(mean_pred, frac_pos, marker="o", color=BLU, linewidth=2, label="Logistic")

    if xgb_probs is not None:
        frac_pos2, mean_pred2 = calibration_curve(y_val, xgb_probs, n_bins=10)
        ax.plot(mean_pred2, frac_pos2, marker="s", color=Y, linewidth=2, label="XGBoost")

    ax.set_xlabel("Mean predicted probability", fontsize=11, labelpad=8)
    ax.set_ylabel("Fraction of positives", fontsize=11, labelpad=8)
    ax.set_title("Calibration curve (reliability diagram)",
                 fontsize=13, fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    ax.tick_params(axis="x", labelsize=9, labelcolor=MUT)
    ax.tick_params(axis="y", labelsize=9, labelcolor=MUT)
    ax.legend()
    fig.tight_layout(pad=2.5)
    _save(fig, "12_calibration_curve.png")


def _fig_auc_gap(log_train_auc: float, log_val_auc: float,
                 xgb_train_auc: Optional[float], xgb_val_auc: Optional[float]) -> None:
    """One figure with one or two grouped bars (logistic + optional XGBoost)."""
    models  = ["Logistic"]
    tr_vals = [log_train_auc]
    vl_vals = [log_val_auc]
    if xgb_train_auc is not None and xgb_val_auc is not None:
        models.append("XGBoost")
        tr_vals.append(xgb_train_auc)
        vl_vals.append(xgb_val_auc)

    x   = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(14, 5))
    bars_tr = ax.bar(x - 0.2, tr_vals, width=0.35, color=Y,   alpha=0.88, edgecolor="none", label="Train AUC")
    bars_vl = ax.bar(x + 0.2, vl_vals, width=0.35, color=GRN, alpha=0.88, edgecolor="none", label="Val AUC")
    ax.set_ylim(0.5, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("ROC-AUC", fontsize=11, labelpad=8)
    ax.set_title("Train vs Validation AUC — overfitting check",
                 fontsize=13, fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    ax.tick_params(axis="x", labelsize=9, labelcolor=MUT)
    ax.tick_params(axis="y", labelsize=9, labelcolor=MUT)
    ax.legend()
    y_max = ax.get_ylim()[1]
    for bar in bars_tr:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005 * y_max, f"{h:.4f}",
                ha="center", va="bottom", fontsize=9, color=TXT, fontweight="bold")
    for i, (bar, v, tv) in enumerate(zip(bars_vl, vl_vals, tr_vals)):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005 * y_max, f"{v:.4f}",
                ha="center", va="bottom", fontsize=9, color=TXT, fontweight="bold")
        gap = tv - v
        ax.text(x[i], 0.52, f"gap={gap:.4f}", ha="center",
                color=RED if gap > 0.05 else GRN, fontsize=9, fontweight="bold")
    fig.tight_layout(pad=2.5)
    _save(fig, "13_auc_gap.png")


def _fig_status_scatter(diag_df: pd.DataFrame) -> None:
    sc = {"keep": GRN, "review": ORG, "drop_candidate": RED}
    fig, ax = plt.subplots(figsize=(14, 5))
    for status, color in sc.items():
        subset = diag_df[diag_df["status"] == status]
        ax.scatter(
            subset["perm_auc_drop"].clip(lower=-0.01),
            subset.get("goodness_score", pd.Series(0, index=subset.index)).fillna(0),
            color=color, alpha=0.7, s=40, linewidths=0, label=status, zorder=3,
        )
        for _, row in subset.iterrows():
            ax.annotate(
                row["feature"],
                (row["perm_auc_drop"],
                 row.get("goodness_score", 0) if not pd.isna(row.get("goodness_score", 0)) else 0),
                fontsize=6, color=TXT, alpha=0.7,
            )
    ax.axvline(0.002, color=GRN, linestyle="--", linewidth=1)
    ax.axvline(0,     color=MUT, linewidth=1)
    ax.set_xlabel("Permutation AUC drop", fontsize=11, labelpad=8)
    ax.set_ylabel("Goodness score", fontsize=11, labelpad=8)
    ax.set_title("Feature diagnostic scatter — keep / review / drop_candidate",
                 fontsize=13, fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    ax.tick_params(axis="x", labelsize=9, labelcolor=MUT)
    ax.tick_params(axis="y", labelsize=9, labelcolor=MUT)
    ax.legend()
    fig.tight_layout(pad=2.5)
    _save(fig, "14_diagnostic_scatter.png")


# ── main entry point ───────────────────────────────────────────────────────────

def run(state: dict) -> dict:
    """Compute diagnostics: permutation drop, SHAP, calibration, overfitting check."""
    feat_df = _load_feat_df(state)

    # ── load models from state or disk ────────────────────────────────────────
    log_model  = state.get("log_model")
    xgb_model  = state.get("xgb_model")
    scaler     = state.get("scaler")
    feat_cols  = state.get("feat_cols")
    best_model = state.get("best_model", "logistic")

    if log_model is None:
        pkl = OUT_MODELS / "models.pkl"
        if pkl.exists():
            with open(pkl, "rb") as f:
                art = pickle.load(f)
            log_model  = art.get("log_model")
            xgb_model  = art.get("xgb_model")
            scaler     = art.get("scaler")
            feat_cols  = art.get("feat_cols")
            best_model = art.get("best_model", "logistic")

    if log_model is None:
        raise FileNotFoundError("models.pkl not found — run m1_train first.")

    feat_cols = [c for c in (feat_cols or []) if c in feat_df.columns]
    if not feat_cols:
        feat_cols = [c for c in feat_df.columns
                     if c not in ("tconst", "label", "primaryTitle", "canonical_title")]

    feat_df, X_tr, X_vl, y_tr, y_vl, _, _ = _prep_splits(feat_df, feat_cols)

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(X_tr)

    # ── AUC on train and val ───────────────────────────────────────────────────
    log_train_probs = log_model.predict_proba(scaler.transform(X_tr))[:, 1]
    log_val_probs   = log_model.predict_proba(scaler.transform(X_vl))[:, 1]
    log_train_auc   = float(roc_auc_score(y_tr, log_train_probs))
    log_val_auc     = float(roc_auc_score(y_vl, log_val_probs))

    xgb_val_probs = xgb_train_auc = xgb_val_auc = None
    if xgb_model is not None:
        xgb_val_probs   = xgb_model.predict_proba(X_vl)[:, 1]
        xgb_train_probs = xgb_model.predict_proba(X_tr)[:, 1]
        xgb_val_auc     = float(roc_auc_score(y_vl, xgb_val_probs))
        xgb_train_auc   = float(roc_auc_score(y_tr, xgb_train_probs))

    # ── permutation importance ─────────────────────────────────────────────────
    active_name = best_model.lower()
    if active_name == "xgboost" and xgb_model is not None:
        active_obj  = xgb_model
        active_sclr = None
        active_nm   = "xgboost"
    else:
        active_obj  = log_model
        active_sclr = scaler
        active_nm   = "logistic"

    print(f"[m2_diagnostics] Computing permutation drops for {active_nm} "
          f"({len(feat_cols)} features)...")
    perm_df = compute_permutation_drop(active_nm, active_obj, X_vl, y_vl,
                                        feat_cols, active_sclr)

    # ── SHAP ──────────────────────────────────────────────────────────────────
    shap_df = compute_shap(xgb_model, X_vl, feat_cols)

    # ── combined diagnostics ───────────────────────────────────────────────────
    diag_df = perm_df.copy()
    if not shap_df.empty:
        diag_df = diag_df.merge(shap_df[["feature", "mean_abs_shap"]], on="feature", how="left")
    else:
        diag_df["mean_abs_shap"] = 0.0

    # Try to merge goodness scores from a prior feature-quality step
    goodness_fp = OUT_FEAT / "feature_goodness.csv"
    if goodness_fp.exists():
        good_df = pd.read_csv(goodness_fp)[["feature", "goodness_score", "status"]].copy()
        diag_df = diag_df.merge(good_df, on="feature", how="left")
    else:
        diag_df["goodness_score"] = 0.5
        diag_df["status"] = "review"

    # Re-classify with diagnostic rules
    good_med = float(diag_df["goodness_score"].fillna(0.5).median())
    diag_df["status"] = "review"
    diag_df.loc[
        (diag_df["perm_auc_drop"] <= 0) &
        (diag_df["goodness_score"].fillna(0) < good_med),
        "status",
    ] = "drop_candidate"
    diag_df.loc[
        (diag_df["perm_auc_drop"] >= 0.002) |
        (diag_df["goodness_score"].fillna(0) >= 0.60),
        "status",
    ] = "keep"

    # ── figures ────────────────────────────────────────────────────────────────
    print("[m2_diagnostics] Saving figures...")
    _fig_perm_drop(perm_df, active_nm)
    _fig_shap(shap_df)
    _fig_calibration(log_val_probs, xgb_val_probs, y_vl.to_numpy())
    _fig_auc_gap(log_train_auc, log_val_auc, xgb_train_auc, xgb_val_auc)
    _fig_status_scatter(diag_df)

    # ── save CSV ───────────────────────────────────────────────────────────────
    diag_df.to_csv(OUT_MODELS / "feature_diagnostics.csv", index=False)

    keeps = int((diag_df["status"] == "keep").sum())
    drops = int((diag_df["status"] == "drop_candidate").sum())
    print(f"[m2_diagnostics] feature_diagnostics.csv saved  "
          f"({len(diag_df)} features | keep={keeps} drop_candidate={drops})")

    state["feature_diagnostics"] = diag_df
    return state


if __name__ == "__main__":
    run({})
    print("[m2_diagnostics] Done.")
