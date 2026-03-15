#!/usr/bin/env python3
"""
f2_feature_selection.py — Feature Selection & Matrix Preparation
=================================================================
Pipeline step: feature_engineering / f2

Reads
-----
  data/processed/features_train.parquet   (preferred)
  data/processed/features_train.csv       (CSV fallback)

Writes
------
  data/processed/features_train_prepped.parquet
  data/processed/features_train_prepped.csv
  data/processed/feature_figures/07_action_summary.png
  data/processed/feature_figures/08_endyear_evidence.png
  data/processed/feature_figures/09_capping_runtimeMinutes.png
  data/processed/feature_figures/10_capping_numVotes_log1p.png
  data/processed/feature_figures/11_nan_audit.png

State keys consumed
-------------------
  features_train   or   train_feat  — output of f1_candidate_features

State keys produced
-------------------
  features_train_prepped  — imputed, capped, dropped-column feature DataFrame
  final_feat_cols         — list of feature column names retained in the matrix
  cap_bounds              — {col: (lo, hi)} quantile caps fitted on train
  medians                 — {col: median_value} used for median imputation
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

# ── disposition registry ───────────────────────────────────────────────────────

DISPOSITION_REGISTRY = [
    {"feature": "startYear",               "action": "keep_numeric",     "imputation": "median_train", "scaling": "standard",  "struct_miss": "no",      "justification": "Release era matters; low missingness; recover with train median."},
    {"feature": "endYear",                 "action": "drop",             "imputation": "none",         "scaling": "none",      "struct_miss": "yes",     "justification": "~90% missing. Captured by end_missing + year_span. Median imputation would be misleading."},
    {"feature": "runtimeMinutes",          "action": "cap_then_keep",    "imputation": "median_train", "scaling": "standard",  "struct_miss": "no",      "justification": "Runtime informative but heavy-tailed; cap+impute."},
    {"feature": "numVotes_log1p",          "action": "cap_then_keep",    "imputation": "median_train", "scaling": "standard",  "struct_miss": "no",      "justification": "Popularity proxy; log + capping handles skew."},
    {"feature": "title_len",               "action": "keep_numeric",     "imputation": "none",         "scaling": "standard",  "struct_miss": "no",      "justification": "Always computable from title; no missingness."},
    {"feature": "title_word_count",        "action": "keep_numeric",     "imputation": "none",         "scaling": "standard",  "struct_miss": "no",      "justification": "Always computable."},
    {"feature": "title_has_digit",         "action": "keep_numeric",     "imputation": "none",         "scaling": "none",      "struct_miss": "no",      "justification": "Binary flag."},
    {"feature": "title_has_colon",         "action": "keep_numeric",     "imputation": "none",         "scaling": "none",      "struct_miss": "no",      "justification": "Binary flag."},
    {"feature": "title_has_question",      "action": "keep_numeric",     "imputation": "none",         "scaling": "none",      "struct_miss": "no",      "justification": "Binary flag."},
    {"feature": "title_upper_ratio",       "action": "keep_numeric",     "imputation": "none",         "scaling": "standard",  "struct_miss": "no",      "justification": "Bounded [0,1]; no imputation needed."},
    {"feature": "has_original_title",      "action": "keep_numeric",     "imputation": "none",         "scaling": "none",      "struct_miss": "no",      "justification": "Binary flag."},
    {"feature": "runtime_missing",         "action": "keep_flag_only",   "imputation": "none",         "scaling": "none",      "struct_miss": "partial", "justification": "Missingness is signal; flag retained."},
    {"feature": "votes_missing",           "action": "keep_flag_only",   "imputation": "none",         "scaling": "none",      "struct_miss": "partial", "justification": "Missingness is signal."},
    {"feature": "start_missing",           "action": "keep_flag_only",   "imputation": "none",         "scaling": "none",      "struct_miss": "partial", "justification": "Missingness is signal."},
    {"feature": "end_missing",             "action": "keep_flag_only",   "imputation": "none",         "scaling": "none",      "struct_miss": "yes",     "justification": "endYear structurally missing; flag is informative."},
    {"feature": "year_span",               "action": "keep_numeric",     "imputation": "none",         "scaling": "standard",  "struct_miss": "no",      "justification": "0 when either year absent; captures lifecycle without leakage."},
    {"feature": "num_directors",           "action": "keep_numeric",     "imputation": "zero_fill",    "scaling": "standard",  "struct_miss": "no",      "justification": "Count; 0 for no edge data."},
    {"feature": "num_unique_directors",    "action": "keep_numeric",     "imputation": "zero_fill",    "scaling": "standard",  "struct_miss": "no",      "justification": "Count."},
    {"feature": "num_writers",             "action": "keep_numeric",     "imputation": "zero_fill",    "scaling": "standard",  "struct_miss": "no",      "justification": "Count."},
    {"feature": "num_unique_writers",      "action": "keep_numeric",     "imputation": "zero_fill",    "scaling": "standard",  "struct_miss": "no",      "justification": "Count."},
    {"feature": "is_auteur",               "action": "keep_numeric",     "imputation": "none",         "scaling": "none",      "struct_miss": "no",      "justification": "Binary flag derived from counts."},
    {"feature": "director_hit_rate",       "action": "encode_then_keep", "imputation": "global_mean",  "scaling": "optional",  "struct_miss": "no",      "justification": "OOF-safe; global mean fallback for unseen."},
    {"feature": "writer_hit_rate",         "action": "encode_then_keep", "imputation": "global_mean",  "scaling": "optional",  "struct_miss": "no",      "justification": "OOF-safe."},
    {"feature": "canonical_title_hit_rate","action": "encode_then_keep", "imputation": "global_mean",  "scaling": "optional",  "struct_miss": "no",      "justification": "OOF-safe."},
    {"feature": "title_group_size_train",  "action": "keep_numeric",     "imputation": "zero_fill",    "scaling": "standard",  "struct_miss": "no",      "justification": "Count; 0 for unseen."},
    {"feature": "title_unique_years_train","action": "keep_numeric",     "imputation": "zero_fill",    "scaling": "standard",  "struct_miss": "no",      "justification": "Count."},
    {"feature": "title_conflicting_years", "action": "keep_flag_only",   "imputation": "none",         "scaling": "none",      "struct_miss": "no",      "justification": "Binary conflict flag."},
    {"feature": "title_sim_to_hit",        "action": "keep_numeric",     "imputation": "zero_fill",    "scaling": "standard",  "struct_miss": "no",      "justification": "Cosine similarity; bounded; 0 is valid default."},
    {"feature": "title_sim_to_non_hit",    "action": "keep_numeric",     "imputation": "zero_fill",    "scaling": "standard",  "struct_miss": "no",      "justification": "Cosine similarity."},
    {"feature": "title_sim_margin",        "action": "keep_numeric",     "imputation": "zero_fill",    "scaling": "standard",  "struct_miss": "no",      "justification": "Margin = hit_sim - non_hit_sim."},
]

ACTION_COLOR = {
    "drop":            RED,
    "keep_numeric":    GRN,
    "keep_flag_only":  GRN,
    "cap_then_keep":   ORG,
    "encode_then_keep": Y,
}

CAP_COLS = ["runtimeMinutes", "numVotes_log1p"]

_SKIP_META = {"tconst", "label", "primaryTitle", "canonical_title"}


# ── imputation ─────────────────────────────────────────────────────────────────

def _apply_imputation(df: pd.DataFrame, medians: Dict[str, float]) -> pd.DataFrame:
    """Apply per-feature imputation according to DISPOSITION_REGISTRY."""
    out = df.copy()
    for rec in DISPOSITION_REGISTRY:
        feat   = rec["feature"]
        policy = rec["imputation"]
        if feat not in out.columns:
            continue
        if policy == "none":
            pass
        elif policy == "median_train":
            out[feat] = out[feat].fillna(medians.get(feat, 0.0))
        elif policy == "zero_fill":
            out[feat] = out[feat].fillna(0.0)
        elif policy == "global_mean":
            gm = medians.get(feat, float(out[feat].mean()))
            out[feat] = out[feat].fillna(gm)
    return out


# ── figures ────────────────────────────────────────────────────────────────────

def _fig_action_summary() -> None:
    counts = Counter(r["action"] for r in DISPOSITION_REGISTRY)
    labels = list(counts.keys())
    vals   = [counts[l] for l in labels]
    colors = [ACTION_COLOR.get(l, MUT) for l in labels]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(labels, vals, color=colors, alpha=0.88, edgecolor="none")
    y_max = ax.get_ylim()[1]
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005 * y_max,
                str(int(h)), ha="center", va="bottom", fontsize=9, color=TXT)
    ax.set_ylabel("# features", fontsize=11, labelpad=8)
    ax.set_title("Action distribution across feature disposition registry",
                 fontsize=13, fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    ax.tick_params(axis="x", labelsize=9, labelcolor=MUT)
    ax.tick_params(axis="y", labelsize=9, labelcolor=MUT)
    fig.tight_layout(pad=2.5)
    fig.savefig(FIG_DIR / "07_action_summary.png", dpi=130,
                bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close(fig)


def _fig_endyear_evidence(df_feat: pd.DataFrame) -> None:
    """Visual evidence for the decision to drop endYear."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: missing rates for a selection of columns
    ax = axes[0]
    probe_cols = ["startYear", "endYear", "runtimeMinutes", "numVotes_log1p", "title_len"]
    probe_cols = [c for c in probe_cols if c in df_feat.columns]
    miss  = [df_feat[c].isna().mean() for c in probe_cols]
    clrs  = [RED if m > 0.5 else ORG if m > 0.1 else GRN for m in miss]
    ax.barh(probe_cols, miss, color=clrs, alpha=0.88, edgecolor="none")
    ax.set_xlabel("Missing rate", fontsize=11, labelpad=8)
    ax.set_title("Missing rates — endYear vs others",
                 fontsize=13, fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    ax.tick_params(axis="y", labelsize=9, labelcolor=MUT)
    ax.tick_params(axis="x", labelsize=9, labelcolor=MUT)
    ax.axvline(0.9, color=RED, linestyle="--", linewidth=1.5)
    ax.text(0.92, 0, "90%", color=RED, va="bottom", fontsize=8)

    # Right: distribution of non-null endYear values
    ax = axes[1]
    if "endYear" in df_feat.columns:
        vals = pd.to_numeric(df_feat["endYear"], errors="coerce").dropna()
        if len(vals) > 0:
            ax.hist(vals, bins=30, color=RED, alpha=0.35, edgecolor="none")
            ax.set_title(f"endYear distribution ({len(vals)} non-null rows)",
                         fontsize=13, fontweight="bold", color=TXT, pad=12)
        else:
            ax.text(0.5, 0.5, "All endYear values are null",
                    ha="center", va="center", color=RED,
                    transform=ax.transAxes, fontsize=12)
            ax.set_title("endYear — no non-null values present",
                         fontsize=13, fontweight="bold", color=TXT, pad=12)
    else:
        ax.text(0.5, 0.5, "endYear column not in feature matrix",
                ha="center", va="center", color=MUT,
                transform=ax.transAxes, fontsize=11)
        ax.set_title("endYear not present (already dropped upstream)",
                     fontsize=13, fontweight="bold", color=TXT, pad=12)
    ax.tick_params(axis="x", labelsize=9, labelcolor=MUT)
    ax.tick_params(axis="y", labelsize=9, labelcolor=MUT)
    for ax_ in axes:
        ax_.title.set_position([0.5, 1.02])

    fig.suptitle(
        "Evidence for dropping endYear: structural missingness + sparse remaining data",
        color=TXT, fontsize=13, fontweight="bold",
    )
    fig.tight_layout(pad=2.5)
    fig.savefig(FIG_DIR / "08_endyear_evidence.png", dpi=130,
                bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close(fig)


def _fig_capping(
    df_feat: pd.DataFrame,
    cap_col: str,
    filename: str,
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> None:
    """Before/after capping distribution for a single column."""
    if cap_col not in df_feat.columns:
        return
    vals = pd.to_numeric(df_feat[cap_col], errors="coerce").dropna()
    if vals.empty:
        return

    lo     = float(vals.quantile(q_low))
    hi     = float(vals.quantile(q_high))
    capped = vals.clip(lo, hi)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, data, title, color in zip(
        axes,
        [vals, capped],
        [
            f"{cap_col} — BEFORE capping",
            f"{cap_col} — AFTER capping [p{q_low*100:.0f}, p{q_high*100:.0f}]",
        ],
        [RED, GRN],
    ):
        ax.hist(data, bins=40, color=color, alpha=0.35, edgecolor="none")
        ax.set_title(title, fontsize=13, fontweight="bold", color=TXT, pad=12)
        ax.title.set_position([0.5, 1.02])
        ax.tick_params(axis="x", labelsize=9, labelcolor=MUT)
        ax.tick_params(axis="y", labelsize=9, labelcolor=MUT)
        ax.set_xlabel(
            f"mean={data.mean():.2f}  std={data.std():.2f}  max={data.max():.1f}",
            color=MUT, fontsize=9, labelpad=8,
        )
    fig.tight_layout(pad=2.5)
    fig.savefig(FIG_DIR / filename, dpi=130,
                bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close(fig)


def _fig_nan_audit(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    feat_cols: List[str],
) -> None:
    """Side-by-side NaN count bars before and after imputation."""
    cols_with_nan = [
        c for c in feat_cols
        if c in df_before.columns and df_before[c].isna().sum() > 0
    ]
    if not cols_with_nan:
        print("[f2] No missing values in selected feature columns — skipping NaN audit figure.")
        return

    before = [df_before[c].isna().sum() for c in cols_with_nan]
    after  = [df_after[c].isna().sum() if c in df_after.columns else 0 for c in cols_with_nan]
    x = np.arange(len(cols_with_nan))

    fig, ax = plt.subplots(figsize=(14, 5))
    bars_b = ax.bar(x - 0.2, before, width=0.4, color=RED, alpha=0.88, edgecolor="none", label="before imputation")
    bars_a = ax.bar(x + 0.2, after,  width=0.4, color=GRN, alpha=0.88, edgecolor="none", label="after imputation")
    y_max = ax.get_ylim()[1]
    for bar in bars_b:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005 * y_max,
                    str(int(h)), ha="center", va="bottom", fontsize=8, color=TXT)
    for bar in bars_a:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005 * y_max,
                    str(int(h)), ha="center", va="bottom", fontsize=8, color=TXT)
    ax.set_xticks(x)
    ax.set_xticklabels(cols_with_nan, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("NaN count", fontsize=11, labelpad=8)
    ax.set_title("NaN counts before vs after imputation",
                 fontsize=13, fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    ax.tick_params(axis="y", labelsize=9, labelcolor=MUT)
    ax.legend()
    fig.tight_layout(pad=2.5)
    fig.savefig(FIG_DIR / "11_nan_audit.png", dpi=130,
                bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close(fig)


# ── main run ───────────────────────────────────────────────────────────────────

def run(state: dict) -> dict:
    """
    Run feature selection and matrix preparation.

    Consumes state keys: features_train | train_feat
    Produces state keys: features_train_prepped, final_feat_cols, cap_bounds, medians
    """
    # ── 1. Load feature matrix ────────────────────────────────────────────────
    feat_df: Optional[pd.DataFrame] = state.get("features_train")
    if feat_df is None:
        feat_df = state.get("train_feat")

    if feat_df is None:
        for fp in [OUT_FEAT / "features_train.parquet", OUT_FEAT / "features_train.csv"]:
            if fp.exists():
                feat_df = (
                    pd.read_parquet(fp) if fp.suffix == ".parquet"
                    else pd.read_csv(fp)
                )
                print(f"[f2] Loaded {fp.name}  shape={feat_df.shape}")
                break

    if feat_df is None:
        raise FileNotFoundError(
            "features_train.parquet/.csv not found — run f1_candidate_features first."
        )

    # Coerce numeric columns
    for col in feat_df.columns:
        if col not in _SKIP_META:
            feat_df[col] = pd.to_numeric(feat_df[col], errors="coerce")

    # ── 2. Determine feature columns ──────────────────────────────────────────
    drop_feats   = {r["feature"] for r in DISPOSITION_REGISTRY if r["action"] == "drop"}
    all_candidate = [c for c in feat_df.columns if c not in _SKIP_META]
    final_feat_cols = [c for c in all_candidate if c not in drop_feats]

    # ── 3. Capping (fit on train only) ────────────────────────────────────────
    cap_bounds: Dict[str, Tuple[float, float]] = {}
    feat_capped = feat_df.copy()
    for col in CAP_COLS:
        if col in feat_capped.columns:
            vals = pd.to_numeric(feat_capped[col], errors="coerce").dropna()
            if len(vals) > 0:
                lo, hi = float(vals.quantile(0.01)), float(vals.quantile(0.99))
                cap_bounds[col] = (lo, hi)
                feat_capped[col] = pd.to_numeric(feat_capped[col], errors="coerce").clip(lo, hi)

    # ── 4. Imputation medians (fit on train) ──────────────────────────────────
    medians: Dict[str, float] = {}
    for col in final_feat_cols:
        if col in feat_capped.columns:
            v = pd.to_numeric(feat_capped[col], errors="coerce")
            medians[col] = float(v.median()) if not v.dropna().empty else 0.0

    meta_cols = ["tconst"] + (["label"] if "label" in feat_capped.columns else [])
    available_feat = [c for c in final_feat_cols if c in feat_capped.columns]
    feat_imputed = _apply_imputation(
        feat_capped[meta_cols + available_feat], medians
    )

    # ── 5. Figures ────────────────────────────────────────────────────────────
    print("[f2] Saving figures...")
    _fig_action_summary()
    _fig_endyear_evidence(feat_df)
    _fig_capping(feat_df, "runtimeMinutes",  "09_capping_runtimeMinutes.png")
    _fig_capping(feat_df, "numVotes_log1p",  "10_capping_numVotes_log1p.png")
    _fig_nan_audit(feat_df, feat_imputed, available_feat)

    # ── 6. Save outputs ───────────────────────────────────────────────────────
    feat_imputed.to_parquet(OUT_FEAT / "features_train_prepped.parquet", index=False)
    feat_imputed.to_csv(OUT_FEAT / "features_train_prepped.csv", index=False)

    nan_after = int(feat_imputed[available_feat].isna().sum().sum())
    print(
        f"[f2] Saved features_train_prepped.parquet + .csv  "
        f"shape={feat_imputed.shape}  remaining_NaN={nan_after}"
    )
    if nan_after > 0:
        print(f"[f2] WARNING: {nan_after} NaN values remain after imputation — check policy coverage.")

    state["features_train_prepped"] = feat_imputed
    state["final_feat_cols"]        = available_feat
    state["cap_bounds"]             = cap_bounds
    state["medians"]                = medians
    return state


if __name__ == "__main__":
    run({})
    print("[f2] Done.")
