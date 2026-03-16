#!/usr/bin/env python3
"""
Step m3: Reduced model selection via ablation curve.

Figures saved to data/processed/model_figures/:
  15_ablation_curve.png
  16_roc_full_vs_reduced.png
  17_dropped_vs_kept.png

Artifacts saved to data/processed/:
  keep_set_features.txt
  reduced_model.pkl   — {model, scaler, feat_cols, model_type}
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
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


# ── model evaluation helper ────────────────────────────────────────────────────

def _fit_eval(X_tr: pd.DataFrame, y_tr: pd.Series,
              X_vl: pd.DataFrame, y_vl: pd.Series,
              feat_cols: list, model_type: str = "logistic") -> float:
    """Fit a fresh model on feat_cols subset and return validation AUC."""
    X_tr_sub = X_tr[feat_cols]
    X_vl_sub = X_vl[feat_cols]
    if model_type == "xgboost" and HAS_XGB:
        m = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, random_state=SEED,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", n_jobs=4, verbosity=0,
        )
        m.fit(X_tr_sub, y_tr.astype(int), verbose=False)
        probs = m.predict_proba(X_vl_sub)[:, 1]
    else:
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr_sub)
        X_vl_sc = sc.transform(X_vl_sub)
        m = LogisticRegression(max_iter=2000, random_state=SEED)
        m.fit(X_tr_sc, y_tr.astype(int))
        probs = m.predict_proba(X_vl_sc)[:, 1]
    return float(roc_auc_score(y_vl.astype(int), probs))


# ── save helper ────────────────────────────────────────────────────────────────

def _save(fig, fname: str) -> None:
    fig.savefig(MODEL_FIG_DIR / fname, dpi=130, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)


# ── figure functions ───────────────────────────────────────────────────────────

def _fig_ablation(X_tr: pd.DataFrame, y_tr: pd.Series,
                  X_vl: pd.DataFrame, y_vl: pd.Series,
                  feat_cols_ranked: list, model_type: str) -> tuple:
    """
    AUC vs number of features kept (greedy top-down by diagnostic rank).
    Saves 15_ablation_curve.png and returns (best_n, best_auc, aucs).
    """
    steps = list(range(1, min(len(feat_cols_ranked) + 1, 26)))
    aucs  = []
    for n in steps:
        sub = feat_cols_ranked[:n]
        try:
            auc = _fit_eval(X_tr, y_tr, X_vl, y_vl, sub, model_type)
        except Exception:
            auc = float("nan")
        aucs.append(auc)
        if n % 5 == 0:
            print(f"[m3_reduced_model] ablation n={n}  AUC={auc:.4f}")

    best_n   = steps[int(np.nanargmax(aucs))]
    best_auc = float(np.nanmax(aucs))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(steps, aucs, color=Y, marker="o", markersize=5, linewidth=2)
    ax.axvline(best_n, color=GRN, linestyle="--", linewidth=1.5)
    ax.text(best_n + 0.3, min(a for a in aucs if not np.isnan(a)) + 0.001,
            f"Best: n={best_n}\nAUC={best_auc:.4f}", color=GRN, fontsize=9)
    ax.set_xlabel("Number of features (top-N by diagnostic rank)", fontsize=11, labelpad=8)
    ax.set_ylabel("Validation AUC", fontsize=11, labelpad=8)
    ax.set_title(f"Ablation curve — {model_type}: AUC vs feature count",
                 fontsize=13, fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    ax.tick_params(axis="x", labelsize=9, labelcolor=MUT)
    ax.tick_params(axis="y", labelsize=9, labelcolor=MUT)
    fig.tight_layout(pad=2.5)
    _save(fig, "15_ablation_curve.png")
    return best_n, best_auc, aucs


def _fig_roc_compare(full_probs: np.ndarray, red_probs: np.ndarray,
                     y_val: pd.Series) -> None:
    y = y_val.astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot([0, 1], [0, 1], color=MUT, linestyle="--", linewidth=1)
    fpr1, tpr1, _ = roc_curve(y, full_probs)
    full_auc = roc_auc_score(y, full_probs)
    ax.fill_between(fpr1, tpr1, alpha=0.08, color=BLU)
    ax.plot(fpr1, tpr1, color=BLU, linewidth=2.5,
            label=f"Full model  AUC={full_auc:.4f}")
    fpr2, tpr2, _ = roc_curve(y, red_probs)
    red_auc = roc_auc_score(y, red_probs)
    ax.fill_between(fpr2, tpr2, alpha=0.08, color=Y)
    ax.plot(fpr2, tpr2, color=Y, linewidth=2.5,
            label=f"Reduced model  AUC={red_auc:.4f}")
    ax.set_xlabel("FPR", fontsize=11, labelpad=8)
    ax.set_ylabel("TPR", fontsize=11, labelpad=8)
    ax.set_title("ROC: full vs reduced model",
                 fontsize=13, fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    ax.tick_params(axis="x", labelsize=9, labelcolor=MUT)
    ax.tick_params(axis="y", labelsize=9, labelcolor=MUT)
    ax.legend()
    fig.tight_layout(pad=2.5)
    _save(fig, "16_roc_full_vs_reduced.png")


def _fig_dropped_vs_kept(diag_df: pd.DataFrame, keep_set: List[str]) -> None:
    all_feats = diag_df["feature"].tolist()
    diag_df   = diag_df.copy()
    diag_df["in_keep_set"] = [f in keep_set for f in all_feats]
    drop_df = diag_df[~diag_df["in_keep_set"]].sort_values("perm_auc_drop", ascending=True)
    keep_df = diag_df[ diag_df["in_keep_set"]].sort_values("perm_auc_drop", ascending=True)

    n_rows = max(len(drop_df), len(keep_df), 1)
    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, n_rows * 0.55)))
    for ax, df, title, color in zip(
        axes,
        [drop_df, keep_df],
        ["Dropped features (perm AUC drop)", "Kept features (perm AUC drop)"],
        [RED, GRN],
    ):
        if len(df) > 0:
            bars = ax.barh(df["feature"], df["perm_auc_drop"].clip(lower=-0.005),
                           color=color, alpha=0.88, edgecolor="none")
            ax.axvline(0, color=MUT, linewidth=1)
            x_max = ax.get_xlim()[1]
            for bar in bars:
                w = bar.get_width()
                ax.text(w + 0.002 * x_max, bar.get_y() + bar.get_height() / 2,
                        f"{w:.4f}", va="center", ha="left", fontsize=7, color=TXT)
        ax.set_title(title, fontsize=13, fontweight="bold", color=TXT, pad=12)
        ax.title.set_position([0.5, 1.02])
        ax.tick_params(axis="y", labelsize=9, labelcolor=MUT)
        ax.tick_params(axis="x", labelsize=9, labelcolor=MUT)
    fig.tight_layout(pad=2.5)
    _save(fig, "17_dropped_vs_kept.png")


# ── main entry point ───────────────────────────────────────────────────────────

def run(state: dict) -> dict:
    """Select reduced feature set via ablation, retrain, save artifacts."""
    feat_df = _load_feat_df(state)

    # ── load diagnostics ───────────────────────────────────────────────────────
    diag_df = state.get("feature_diagnostics")
    if diag_df is None:
        fp = OUT_MODELS / "feature_diagnostics.csv"
        if fp.exists():
            diag_df = pd.read_csv(fp)
    if diag_df is None:
        fp2 = OUT_FEAT / "feature_goodness.csv"
        if fp2.exists():
            diag_df = pd.read_csv(fp2)

    # ── load models ────────────────────────────────────────────────────────────
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

    feat_cols = [c for c in (feat_cols or []) if c in feat_df.columns]
    if not feat_cols:
        feat_cols = [c for c in feat_df.columns
                     if c not in ("tconst", "label", "primaryTitle", "canonical_title")]

    feat_df, X_tr, X_vl, y_tr, y_vl, _, _ = _prep_splits(feat_df, feat_cols)

    active_type = "xgboost" if best_model.lower() == "xgboost" and xgb_model else "logistic"

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(X_tr)

    # ── full model predictions ─────────────────────────────────────────────────
    if active_type == "xgboost":
        full_probs = xgb_model.predict_proba(X_vl)[:, 1]
    else:
        full_probs = log_model.predict_proba(scaler.transform(X_vl))[:, 1]
    full_auc = float(roc_auc_score(y_vl.astype(int), full_probs))

    # ── determine feature ranking from diagnostics ─────────────────────────────
    if diag_df is not None and "feature" in diag_df.columns:
        sort_col = "perm_auc_drop" if "perm_auc_drop" in diag_df.columns else "goodness_score"
        diag_sorted   = diag_df.sort_values(sort_col, ascending=False)
        feats_ranked  = [f for f in diag_sorted["feature"].tolist() if f in feat_cols]
        if "status" in diag_df.columns:
            keep_set = [
                f for f in feats_ranked
                if diag_df.loc[diag_df["feature"] == f, "status"].values[0] != "drop_candidate"
            ]
        else:
            keep_set = feats_ranked[:max(5, len(feats_ranked) // 2)]
    else:
        feats_ranked = feat_cols
        keep_set     = feat_cols

    keep_set = [f for f in keep_set if f in feat_cols]
    if len(keep_set) < 3:
        keep_set = feat_cols  # safety fallback

    # ── ablation curve ─────────────────────────────────────────────────────────
    print(f"[m3_reduced_model] Running ablation ({len(feats_ranked)} features)...")
    best_n, best_auc, abl_aucs = _fig_ablation(X_tr, y_tr, X_vl, y_vl,
                                                feats_ranked, active_type)

    ablation_set = feats_ranked[:best_n]
    final_keep   = ablation_set if len(ablation_set) >= 2 else keep_set

    # ── retrain reduced model ──────────────────────────────────────────────────
    print(f"[m3_reduced_model] Retraining reduced model on {len(final_keep)} features...")
    if active_type == "xgboost" and HAS_XGB:
        red_model = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, random_state=SEED,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", n_jobs=4, verbosity=0,
        )
        red_model.fit(X_tr[final_keep], y_tr.astype(int), verbose=False)
        red_probs  = red_model.predict_proba(X_vl[final_keep])[:, 1]
        red_scaler = None
    else:
        red_scaler = StandardScaler()
        X_tr_red   = red_scaler.fit_transform(X_tr[final_keep])
        X_vl_red   = red_scaler.transform(X_vl[final_keep])
        red_model  = LogisticRegression(max_iter=2000, random_state=SEED)
        red_model.fit(X_tr_red, y_tr.astype(int))
        red_probs  = red_model.predict_proba(X_vl_red)[:, 1]

    red_auc   = float(roc_auc_score(y_vl.astype(int), red_probs))
    auc_delta = red_auc - full_auc

    # ── figures ────────────────────────────────────────────────────────────────
    print("[m3_reduced_model] Saving figures...")
    _fig_roc_compare(full_probs, red_probs, y_vl)

    fallback_diag = diag_df if diag_df is not None else pd.DataFrame({
        "feature":      feat_cols,
        "perm_auc_drop": [0.0] * len(feat_cols),
    })
    _fig_dropped_vs_kept(fallback_diag, final_keep)

    # ── save artifacts ─────────────────────────────────────────────────────────
    (OUT_MODELS / "keep_set_features.txt").write_text("\n".join(final_keep), encoding="utf-8")
    with open(OUT_MODELS / "reduced_model.pkl", "wb") as f:
        pickle.dump({
            "model":      red_model,
            "scaler":     red_scaler,
            "feat_cols":  final_keep,
            "model_type": active_type,
        }, f)

    print(f"[m3_reduced_model] Full AUC={full_auc:.4f}  "
          f"Reduced AUC={red_auc:.4f}  "
          f"Delta={auc_delta:+.4f}  "
          f"Keep={len(final_keep)} features")

    state.update({
        "keep_set":  final_keep,
        "red_model": red_model,
        "red_auc":   red_auc,
    })
    return state


if __name__ == "__main__":
    run({})
    print("[m3_reduced_model] Done.")
