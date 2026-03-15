#!/usr/bin/env python3
"""
f3_feature_quality.py — Feature Quality Diagnostics
=====================================================
Pipeline step: feature_engineering / f3

Reads
-----
  data/processed/features_train_prepped.parquet   (preferred)
  data/processed/features_train_prepped.csv       (CSV fallback)

Writes
------
  data/processed/feature_goodness.csv
  data/processed/feature_figures/12_goodness_heatmap.png
  data/processed/feature_figures/13_auc_bar.png
  data/processed/feature_figures/14_mi_bar.png
  data/processed/feature_figures/15_psi_bar.png
  data/processed/feature_figures/16_status_bar.png

State keys consumed
-------------------
  features_train_prepped  — output of f2_feature_selection

State keys produced
-------------------
  feature_goodness     — DataFrame with per-feature quality metrics and status
  feat_cols_quality    — list of feature columns that were evaluated
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# ── paths ──────────────────────────────────────────────────────────────────────
PIPELINE = Path(__file__).resolve().parent.parent
PROC     = PIPELINE.parent / "data" / "processed"   # clean parquets (read-only)
OUT_FEAT = PIPELINE / "outputs" / "features"
FIG_DIR  = OUT_FEAT
OUT_FEAT.mkdir(parents=True, exist_ok=True)

# ── style ──────────────────────────────────────────────────────────────────────
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

SEED = 42

_SKIP_META = {"tconst", "label", "primaryTitle", "canonical_title"}


# ── quality computation ────────────────────────────────────────────────────────

def _safe_auc(y: np.ndarray, x: pd.Series) -> float:
    """Univariate ROC-AUC, reflected so result is always >= 0.5."""
    if x.nunique() <= 1 or len(np.unique(y)) != 2:
        return float("nan")
    auc = float(roc_auc_score(y, x))
    return max(auc, 1.0 - auc)


def compute_psi(
    train_vals: pd.Series,
    val_vals: pd.Series,
    n_bins: int = 10,
) -> float:
    """
    Population Stability Index (PSI) between two distributions.

    Interpretation
    --------------
    < 0.10  : stable
    0.10-0.25: moderate shift, worth monitoring
    > 0.25  : high shift, feature may be unstable at deployment
    """
    tr = pd.to_numeric(train_vals, errors="coerce").dropna()
    vl = pd.to_numeric(val_vals,   errors="coerce").dropna()
    if tr.empty or vl.empty:
        return float("nan")

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges     = np.unique(np.quantile(tr, quantiles))
    if len(edges) < 3:
        return float("nan")

    tr_dist = pd.cut(tr, bins=edges, include_lowest=True).value_counts(normalize=True).sort_index()
    vl_dist = pd.cut(vl, bins=edges, include_lowest=True).value_counts(normalize=True).sort_index()
    al = pd.concat([tr_dist, vl_dist], axis=1).fillna(1e-6).clip(lower=1e-6)
    al.columns = ["t", "v"]
    return float(((al["v"] - al["t"]) * np.log(al["v"] / al["t"])).sum())


def compute_goodness(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    val_X: pd.DataFrame,
    val_y: pd.Series,
    feat_cols: List[str],
) -> pd.DataFrame:
    """
    Compute a composite goodness score for each feature.

    Metrics
    -------
    univariate_auc_train/val : one-feature ROC-AUC (reflected >= 0.5)
    mutual_info              : sklearn MI against binary label (train)
    spearman_train/val       : Spearman rank correlation with label
    abs_spearman_val         : |spearman_val| for ranking
    psi_train_vs_val         : distribution stability

    Goodness score
    --------------
    0.35 * AUC_rank + 0.25 * MI_rank + 0.20 * |Spearman|_rank
    + 0.10 * (1-PSI)_rank + 0.10 * (1-miss)_rank

    Status rules
    ------------
    keep           : goodness >= 0.60
    drop_candidate : AUC_val < 0.52 AND MI < median AND goodness < median
    review         : everything else (including high-PSI features)
    """
    y_tr = train_y.astype(int).to_numpy()
    y_vl = val_y.astype(int).to_numpy()
    rows = []

    for feat in feat_cols:
        x_tr = (pd.to_numeric(train_X[feat], errors="coerce")
                if feat in train_X.columns else pd.Series(dtype=float))
        x_vl = (pd.to_numeric(val_X[feat], errors="coerce")
                if feat in val_X.columns else pd.Series(dtype=float))

        med      = float(x_tr.median()) if not x_tr.dropna().empty else 0.0
        x_tr_f   = x_tr.fillna(med)
        x_vl_f   = x_vl.fillna(med)

        auc_tr   = _safe_auc(y_tr, x_tr_f)
        auc_vl   = _safe_auc(y_vl, x_vl_f)
        mi       = float(mutual_info_classif(
            x_tr_f.to_numpy().reshape(-1, 1), y_tr,
            discrete_features=False, random_state=SEED,
        )[0])
        spear_tr = (float(pd.Series(x_tr_f).corr(pd.Series(y_tr), method="spearman"))
                    if x_tr_f.nunique() > 1 else float("nan"))
        spear_vl = (float(pd.Series(x_vl_f).corr(pd.Series(y_vl), method="spearman"))
                    if x_vl_f.nunique() > 1 else float("nan"))
        psi      = compute_psi(x_tr, x_vl)

        rows.append({
            "feature":              feat,
            "missing_rate_train":   float(x_tr.isna().mean()),
            "missing_rate_val":     float(x_vl.isna().mean()),
            "std_train":            float(x_tr_f.std(ddof=0)),
            "univariate_auc_train": auc_tr,
            "univariate_auc_val":   auc_vl,
            "mutual_info":          mi,
            "spearman_train":       spear_tr,
            "spearman_val":         spear_vl,
            "abs_spearman_val":     abs(spear_vl) if not np.isnan(spear_vl) else float("nan"),
            "psi_train_vs_val":     psi,
        })

    df = pd.DataFrame(rows)

    # Composite score via percentile ranking
    auc_r   = df["univariate_auc_val"].fillna(0.5).rank(pct=True)
    mi_r    = df["mutual_info"].fillna(0).rank(pct=True)
    sp_r    = df["abs_spearman_val"].fillna(0).rank(pct=True)
    psi_max = df["psi_train_vs_val"].max(skipna=True)
    psi_max = psi_max if not pd.isna(psi_max) else 1.0
    psi_r   = (-df["psi_train_vs_val"].fillna(psi_max)).rank(pct=True)
    miss_r  = (-df["missing_rate_train"].fillna(1.0)).rank(pct=True)
    df["goodness_score"] = (
        0.35 * auc_r + 0.25 * mi_r + 0.20 * sp_r + 0.10 * psi_r + 0.10 * miss_r
    )

    median_good = float(df["goodness_score"].median())

    # Status assignment
    df["status"] = "review"
    df.loc[df["psi_train_vs_val"].fillna(0) > 0.2, "status"] = "review"
    df.loc[df["goodness_score"] >= 0.60, "status"] = "keep"
    df.loc[
        (df["univariate_auc_val"].fillna(0.5) < 0.52)
        & (df["mutual_info"].fillna(0) < df["mutual_info"].median())
        & (df["goodness_score"] < median_good),
        "status",
    ] = "drop_candidate"

    return df.sort_values("goodness_score", ascending=False).reset_index(drop=True)


# ── figures ────────────────────────────────────────────────────────────────────

def _fig_goodness_heatmap(diag: pd.DataFrame) -> None:
    import seaborn as sns
    metric_cols = [
        "univariate_auc_val", "mutual_info", "abs_spearman_val",
        "psi_train_vs_val", "goodness_score",
    ]
    metric_cols = [c for c in metric_cols if c in diag.columns]
    df   = diag.set_index("feature")[metric_cols].fillna(0)
    df_n = (df - df.min()) / (df.max() - df.min() + 1e-9)

    # Invert PSI: lower drift → better score
    if "psi_train_vs_val" in df_n.columns:
        df_n["psi_train_vs_val"] = 1.0 - df_n["psi_train_vs_val"]

    n_features = len(df)
    fig, ax = plt.subplots(figsize=(14, max(6, n_features * 0.55)))
    cmap = sns.diverging_palette(15, 145, s=80, l=40, as_cmap=True)
    im = ax.imshow(df_n.values, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(metric_cols)))
    ax.set_xticklabels(metric_cols, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index.tolist(), fontsize=9)
    ax.set_title(
        "Feature quality heatmap (green=good signal/stable, red=poor/high-drift)",
        fontsize=13, fontweight="bold", color=TXT, pad=12,
    )
    ax.title.set_position([0.5, 1.02])
    for i in range(len(df)):
        for j in range(len(metric_cols)):
            raw_val  = df.iloc[i, j]
            ax.text(j, i, f"{raw_val:.2f}", ha="center", va="center",
                    fontsize=8, color=TXT)
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Normalized score", fontsize=9)
    cbar.ax.tick_params(labelsize=8, colors=MUT)
    fig.tight_layout(pad=2.5)
    fig.savefig(FIG_DIR / "12_goodness_heatmap.png", dpi=130,
                bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close(fig)


def _fig_auc_bar(diag: pd.DataFrame) -> None:
    df = diag.dropna(subset=["univariate_auc_val"]).sort_values("univariate_auc_val", ascending=True)
    n_features = len(df)
    fig, ax = plt.subplots(figsize=(14, max(6, n_features * 0.55)))
    colors = [
        GRN if v >= 0.6 else Y if v >= 0.55 else ORG if v >= 0.52 else RED
        for v in df["univariate_auc_val"]
    ]
    bars = ax.barh(df["feature"], df["univariate_auc_val"], color=colors, alpha=0.88, edgecolor="none")
    ax.axvline(0.5,  color=MUT, linestyle="--", linewidth=1)
    ax.axvline(0.55, color=ORG, linestyle="--", linewidth=1)
    ax.axvline(0.60, color=GRN, linestyle="--", linewidth=1)
    x_max = ax.get_xlim()[1]
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", va="center", ha="left", fontsize=8, color=TXT)
    ax.set_xlabel("Univariate ROC-AUC (validation split)", fontsize=11, labelpad=8)
    ax.set_title("Feature univariate predictive power",
                 fontsize=13, fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    ax.tick_params(axis="y", labelsize=9, labelcolor=MUT)
    ax.tick_params(axis="x", labelsize=9, labelcolor=MUT)
    fig.tight_layout(pad=2.5)
    fig.savefig(FIG_DIR / "13_auc_bar.png", dpi=130,
                bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close(fig)


def _fig_mi_bar(diag: pd.DataFrame) -> None:
    df = diag.sort_values("mutual_info", ascending=True)
    n_features = len(df)
    fig, ax = plt.subplots(figsize=(14, max(6, n_features * 0.55)))
    bars = ax.barh(df["feature"], df["mutual_info"], color=BLU, alpha=0.88, edgecolor="none")
    x_max = ax.get_xlim()[1]
    for bar in bars:
        w = bar.get_width()
        if w > 0:
            ax.text(w + 0.001 * x_max, bar.get_y() + bar.get_height() / 2,
                    f"{w:.4f}", va="center", ha="left", fontsize=8, color=TXT)
    ax.set_xlabel("Mutual Information (train)", fontsize=11, labelpad=8)
    ax.set_title("Mutual information with label",
                 fontsize=13, fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    ax.tick_params(axis="y", labelsize=9, labelcolor=MUT)
    ax.tick_params(axis="x", labelsize=9, labelcolor=MUT)
    fig.tight_layout(pad=2.5)
    fig.savefig(FIG_DIR / "14_mi_bar.png", dpi=130,
                bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close(fig)


def _fig_psi_bar(diag: pd.DataFrame) -> None:
    df = diag.dropna(subset=["psi_train_vs_val"]).sort_values(
        "psi_train_vs_val", ascending=False
    )
    n_features = len(df)
    fig, ax = plt.subplots(figsize=(14, max(6, n_features * 0.55)))
    colors = [RED if v > 0.25 else ORG if v > 0.1 else GRN for v in df["psi_train_vs_val"]]
    bars = ax.barh(df["feature"], df["psi_train_vs_val"], color=colors, alpha=0.88, edgecolor="none")
    ax.axvline(0.1,  color=ORG, linestyle="--", linewidth=1)
    ax.axvline(0.25, color=RED, linestyle="--", linewidth=1)
    x_max = ax.get_xlim()[1]
    for bar in bars:
        w = bar.get_width()
        if w > 0:
            ax.text(w + 0.001 * x_max, bar.get_y() + bar.get_height() / 2,
                    f"{w:.3f}", va="center", ha="left", fontsize=8, color=TXT)
    ax.set_xlabel("PSI (train vs val)", fontsize=11, labelpad=8)
    ax.set_title("Population Stability Index — distribution drift between train and validation",
                 fontsize=13, fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    ax.tick_params(axis="y", labelsize=9, labelcolor=MUT)
    ax.tick_params(axis="x", labelsize=9, labelcolor=MUT)

    # Threshold labels — only add if data range is wide enough
    if len(df) > 0 and float(df["psi_train_vs_val"].max()) > 0.15:
        ax.text(0.11, -0.5, "caution",    color=ORG, fontsize=8)
        ax.text(0.26, -0.5, "high drift", color=RED, fontsize=8)

    fig.tight_layout(pad=2.5)
    fig.savefig(FIG_DIR / "15_psi_bar.png", dpi=130,
                bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close(fig)


def _fig_status_pie(diag: pd.DataFrame) -> None:
    """Feature status distribution — shown as a horizontal bar chart."""
    counts     = diag["status"].value_counts().sort_values(ascending=True)
    colors_map = {"keep": GRN, "review": ORG, "drop_candidate": RED}
    labels     = counts.index.tolist()
    vals       = counts.values.tolist()
    clrs       = [colors_map.get(l, MUT) for l in labels]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.barh(labels, vals, color=clrs, alpha=0.88, edgecolor="none")
    x_max = ax.get_xlim()[1]
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005 * x_max, bar.get_y() + bar.get_height() / 2,
                str(int(w)), va="center", ha="left", fontsize=9, color=TXT)
    ax.set_xlabel("# features", fontsize=11, labelpad=8)
    ax.set_title("Feature status distribution",
                 fontsize=13, fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    ax.tick_params(axis="y", labelsize=9, labelcolor=MUT)
    ax.tick_params(axis="x", labelsize=9, labelcolor=MUT)
    fig.tight_layout(pad=2.5)
    fig.savefig(FIG_DIR / "16_status_bar.png", dpi=130,
                bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close(fig)


# ── main run ───────────────────────────────────────────────────────────────────

def run(state: dict) -> dict:
    """
    Run feature quality diagnostics.

    Consumes state keys: features_train_prepped
    Produces state keys: feature_goodness, feat_cols_quality
    """
    # ── 1. Load prepped feature matrix ────────────────────────────────────────
    feat_df: pd.DataFrame | None = state.get("features_train_prepped")

    if feat_df is None:
        for fp in [OUT_FEAT / "features_train_prepped.parquet", OUT_FEAT / "features_train_prepped.csv"]:
            if fp.exists():
                feat_df = (
                    pd.read_parquet(fp) if fp.suffix == ".parquet"
                    else pd.read_csv(fp)
                )
                print(f"[f3] Loaded {fp.name}  shape={feat_df.shape}")
                break

    if feat_df is None:
        raise FileNotFoundError(
            "features_train_prepped.parquet/.csv not found — run f2_feature_selection first."
        )

    # Coerce numeric
    for col in feat_df.columns:
        if col not in _SKIP_META:
            feat_df[col] = pd.to_numeric(feat_df[col], errors="coerce")

    if "label" not in feat_df.columns:
        raise ValueError("label column missing from features_train_prepped — cannot compute quality metrics.")

    # Drop rows with missing label
    label_num = pd.to_numeric(feat_df["label"], errors="coerce")
    bad_label_rows = int(label_num.isna().sum())
    if bad_label_rows:
        print(f"[f3] Dropping {bad_label_rows} rows with non-numeric label.")
    feat_df        = feat_df.loc[label_num.notna()].copy()
    feat_df["label"] = label_num.loc[label_num.notna()].astype(int).to_numpy()

    # ── 2. Feature columns to evaluate ───────────────────────────────────────
    feat_cols = [
        c for c in feat_df.columns
        if c not in _SKIP_META and feat_df[c].notna().sum() > 0
    ]

    # ── 3. Internal 80/20 stratified split ────────────────────────────────────
    train_idx, val_idx = train_test_split(
        feat_df.index,
        test_size=0.20,
        random_state=SEED,
        stratify=feat_df["label"].astype(int),
    )
    train_X = feat_df.loc[train_idx, feat_cols].reset_index(drop=True)
    val_X   = feat_df.loc[val_idx,   feat_cols].reset_index(drop=True)
    train_y = feat_df.loc[train_idx, "label"].reset_index(drop=True)
    val_y   = feat_df.loc[val_idx,   "label"].reset_index(drop=True)

    print(f"[f3] Computing goodness metrics for {len(feat_cols)} features...")
    diag = compute_goodness(train_X, train_y, val_X, val_y, feat_cols)

    keeps  = int((diag["status"] == "keep").sum())
    drops  = int((diag["status"] == "drop_candidate").sum())
    review = int((diag["status"] == "review").sum())
    top3   = ", ".join(diag.head(3)["feature"].tolist())
    print(
        f"[f3] Status summary — keep={keeps}  drop_candidate={drops}  review={review}"
    )
    print(f"[f3] Top-3 features by goodness: {top3}")

    # ── 4. Figures ────────────────────────────────────────────────────────────
    print("[f3] Saving figures...")
    _fig_goodness_heatmap(diag)
    _fig_auc_bar(diag)
    _fig_mi_bar(diag)
    _fig_psi_bar(diag)
    _fig_status_pie(diag)

    # ── 5. Save outputs ───────────────────────────────────────────────────────
    diag.to_csv(OUT_FEAT / "feature_goodness.csv", index=False)
    print(f"[f3] Saved feature_goodness.csv  ({len(diag)} features)")

    state["feature_goodness"]    = diag
    state["feat_cols_quality"]   = feat_cols
    return state


if __name__ == "__main__":
    run({})
    print("[f3] Done.")
