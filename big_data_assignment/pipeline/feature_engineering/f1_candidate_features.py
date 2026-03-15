#!/usr/bin/env python3
"""
f1_candidate_features.py — Candidate Feature Generation
=========================================================
Pipeline step: feature_engineering / f1

Reads
-----
  data/processed/train_clean.parquet
    Columns available: tconst, primaryTitle, originalTitle, startYear,
    runtimeMinutes, numVotes_log1p, label, genres, titleType, isAdult,
    dir_count, dir_avg_birth_year, dir_min_birth_year, dir_avg_death_year,
    dir_professions, wri_count, wri_avg_birth_year, wri_min_birth_year,
    wri_avg_death_year, wri_professions.

    Optional columns (used when present): endYear, numVotes, dir_ids, wri_ids.

Writes
------
  data/processed/features_train.parquet   — full feature matrix (preferred)
  data/processed/features_train.csv       — CSV copy for compatibility
  data/processed/feature_figures/01_feature_missingness.png
  data/processed/feature_figures/02_base_distributions.png
  data/processed/feature_figures/03_binary_flags.png
  data/processed/feature_figures/04_aggregates.png
  data/processed/feature_figures/05_oof_diagram.png
  data/processed/feature_figures/06_oof_distributions.png

State keys consumed
-------------------
  train_df       (optional) — pre-loaded train_clean DataFrame

State keys produced
-------------------
  train_feat     — feature-engineered DataFrame
  dir_lookup     — {entity_id: smoothed_hit_rate} from full-data director OOF
  dir_gm         — director global mean hit rate
  wr_lookup      — {entity_id: smoothed_hit_rate} from full-data writer OOF
  wr_gm          — writer global mean hit rate
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold

# ── paths ──────────────────────────────────────────────────────────────────────
PIPELINE = Path(__file__).resolve().parent.parent
PROC     = PIPELINE / "outputs" / "cleaning"         # clean parquets (read-only)
RAW_CSV  = PIPELINE.parent / "data" / "raw" / "csv"
OUT_FEAT = PIPELINE / "outputs" / "features"
FIG_DIR  = OUT_FEAT
OUT_FEAT.mkdir(parents=True, exist_ok=True)

# ── rcParams ──────────────────────────────────────────────────────────────────
import matplotlib as mpl
mpl.rcParams.update({
    "figure.facecolor":  "#0a0a0a",
    "axes.facecolor":    "#111111",
    "axes.edgecolor":    "#252525",
    "axes.labelcolor":   "#e8e8e8",
    "text.color":        "#e8e8e8",
    "xtick.color":       "#666666",
    "ytick.color":       "#666666",
    "grid.color":        "#1e1e1e",
    "grid.linewidth":    0.6,
    "axes.grid":         True,
    "axes.grid.axis":    "y",
    "figure.dpi":        130,
    "savefig.dpi":       130,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "#0a0a0a",
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "legend.framealpha": 0.15,
    "legend.edgecolor":  "#252525",
})

# ── Colour palette ────────────────────────────────────────────────────────────
BG   = "#0a0a0a"
CRD  = "#111111"
BDR  = "#252525"
TXT  = "#e8e8e8"
MUT  = "#666666"
Y    = "#F5C518"
GRN  = "#2ecc71"
RED  = "#e74c3c"
ORG  = "#f39c12"
BLU  = "#1848f5"
PRP  = "#9b59b6"

# Legacy alias used by existing figure helpers
B = BLU

SEED = 42

# ── feature metadata ───────────────────────────────────────────────────────────

FEATURE_MOTIVATION = {
    "startYear":               "Release era signal; film markets and rating behavior change over time.",
    "endYear":                 "Series/end timing signal — retained as candidate; feature_selection drops it.",
    "runtimeMinutes":          "Runtime signal (raw, before capping).",
    "numVotes_log1p":          "Popularity proxy; log transform handles heavy-tailed vote counts.",
    "title_len":               "Simple lexical complexity proxy for title style.",
    "title_word_count":        "Title structure complexity signal.",
    "title_has_digit":         "Franchise/sequel/year marker signal in titles.",
    "title_has_colon":         "Subtitle/franchise formatting signal.",
    "title_has_question":      "Title style marker potentially linked to genre/tone.",
    "title_upper_ratio":       "Typography/style marker for naming conventions.",
    "has_original_title":      "Localization/remake/translation proxy.",
    "runtime_missing":         "Missingness can itself be informative.",
    "votes_missing":           "Missingness can itself be informative.",
    "start_missing":           "Missingness can itself be informative.",
    "end_missing":             "Missingness can itself be informative.",
    "year_span":               "Duration/lifecycle feature when both years are present.",
    "num_directors":           "Team size effect from many-to-many credits.",
    "num_unique_directors":    "Director diversity effect.",
    "num_writers":             "Writing team size effect.",
    "num_unique_writers":      "Writer diversity effect.",
    "is_auteur":               "Single-director/single-writer concentration proxy.",
    "director_hit_rate":       "Leak-safe OOF target encoding of director history.",
    "writer_hit_rate":         "Leak-safe OOF target encoding of writer history.",
    "canonical_title_hit_rate":"Leak-safe OOF prior success by normalized title.",
    "title_group_size_train":  "How often canonical title appears in training.",
    "title_unique_years_train":"Title ambiguity/remake proxy (same title across years).",
    "title_conflicting_years": "Binary conflict flag for canonical-title year mismatch.",
    "title_sim_to_hit":        "Cosine similarity of title TF-IDF to hit centroid.",
    "title_sim_to_non_hit":    "Cosine similarity of title TF-IDF to non-hit centroid.",
    "title_sim_margin":        "Net semantic tilt toward hit-like vs non-hit-like title language.",
}

FEATURE_GROUPS = {
    "base":       ["title_len", "title_word_count", "title_upper_ratio", "startYear", "year_span", "numVotes_log1p"],
    "binary":     ["title_has_digit", "title_has_colon", "title_has_question", "has_original_title",
                   "runtime_missing", "votes_missing", "start_missing", "end_missing"],
    "aggregates": ["num_directors", "num_unique_directors", "num_writers", "num_unique_writers", "is_auteur"],
    "encodings":  ["director_hit_rate", "writer_hit_rate", "canonical_title_hit_rate"],
}

GROUP_COLORS = {"base": Y, "binary": GRN, "aggregates": B, "encodings": ORG}


# ── feature computation ────────────────────────────────────────────────────────

def canonicalize_title(title: str) -> str:
    if title is None:
        return ""
    text = str(title).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return text.strip()


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute base lexical and numeric features from movie metadata."""
    out = df.copy()
    out["primaryTitle"]  = out["primaryTitle"].fillna("")
    out["originalTitle"] = out.get("originalTitle", pd.Series("", index=out.index)).fillna("")
    out["canonical_title"] = out["primaryTitle"].map(canonicalize_title)

    out["title_len"]        = out["primaryTitle"].astype(str).str.len().astype(float)
    out["title_word_count"] = out["primaryTitle"].astype(str).str.split().str.len().fillna(0).astype(float)
    out["title_has_digit"]  = out["primaryTitle"].astype(str).str.contains(r"\d", regex=True).astype(float)
    out["title_has_colon"]  = out["primaryTitle"].astype(str).str.contains(":", regex=False).astype(float)
    out["title_has_question"] = out["primaryTitle"].astype(str).str.contains(r"\?", regex=True).astype(float)

    title_len_safe = out["title_len"].replace(0, np.nan)
    out["title_upper_ratio"] = (
        out["primaryTitle"].astype(str).str.count(r"[A-Z]") / title_len_safe
    ).fillna(0.0)

    out["has_original_title"] = out["originalTitle"].astype(str).str.strip().ne("").astype(float)
    out["runtime_missing"]    = out["runtimeMinutes"].isna().astype(float)

    # votes_missing: derive from numVotes if available, else from numVotes_log1p being 0
    if "numVotes" in out.columns:
        out["votes_missing"] = out["numVotes"].isna().astype(float)
    else:
        # numVotes_log1p == 0 means log1p(0) = 0 → numVotes was 0 or missing
        out["votes_missing"] = (out["numVotes_log1p"].isna()).astype(float)

    out["start_missing"] = out["startYear"].isna().astype(float)
    out["end_missing"]   = out.get("endYear", pd.Series(np.nan, index=out.index)).isna().astype(float)

    out["year_span"] = (
        (out.get("endYear", pd.Series(np.nan, index=out.index)) - out["startYear"])
        .where(
            out["startYear"].notna()
            & out.get("endYear", pd.Series(np.nan, index=out.index)).notna(),
            0.0,
        )
        .clip(lower=0)
    )

    # numVotes_log1p: skip recompute if already present
    if "numVotes_log1p" not in out.columns:
        if "numVotes" in out.columns:
            out["numVotes_log1p"] = np.log1p(out["numVotes"].clip(lower=0))
        else:
            out["numVotes_log1p"] = 0.0

    return out


def add_aggregate_from_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive director/writer aggregate features directly from pre-aggregated
    columns that the data_cleaning pipeline already embedded in the parquet.

    Expected input columns
    ----------------------
    dir_count   : int — total director credits per title
    wri_count   : int — total writer credits per title
    dir_ids     : str, optional — comma-separated nconst strings for directors
    wri_ids     : str, optional — comma-separated nconst strings for writers
    """
    out = df.copy()

    # --- director counts ---
    out["num_directors"] = pd.to_numeric(out["dir_count"], errors="coerce").fillna(0).astype(float)

    if "dir_ids" in out.columns:
        def _unique_ids(s) -> int:
            if pd.isna(s) or str(s).strip() == "":
                return 0
            tokens = [t.strip() for t in str(s).split(",") if t.strip() not in ("", r"\N", "\\N")]
            return len(set(tokens))
        out["num_unique_directors"] = out["dir_ids"].map(_unique_ids).astype(float)
    else:
        # Fall back: unique directors = total count (best estimate without id list)
        out["num_unique_directors"] = out["num_directors"]

    # --- writer counts ---
    out["num_writers"] = pd.to_numeric(out["wri_count"], errors="coerce").fillna(0).astype(float)

    if "wri_ids" in out.columns:
        def _unique_wri(s) -> int:
            if pd.isna(s) or str(s).strip() == "":
                return 0
            tokens = [t.strip() for t in str(s).split(",") if t.strip() not in ("", r"\N", "\\N")]
            return len(set(tokens))
        out["num_unique_writers"] = out["wri_ids"].map(_unique_wri).astype(float)
    else:
        out["num_unique_writers"] = out["num_writers"]

    out["is_auteur"] = (
        (out["num_unique_directors"] == 1) & (out["num_unique_writers"] == 1)
    ).astype(float)

    return out


def _build_entity_index_from_ids(df: pd.DataFrame, id_col: str) -> Dict[str, List[str]]:
    """
    Build {tconst: [entity_id, ...]} from a comma-separated id column.
    Falls back to empty dict if the column is absent.
    """
    if id_col not in df.columns:
        return {}
    idx: Dict[str, List[str]] = {}
    for tconst, val in zip(df["tconst"].astype(str), df[id_col]):
        if pd.isna(val) or str(val).strip() == "":
            idx[tconst] = []
        else:
            tokens = [t.strip() for t in str(val).split(",")
                      if t.strip() not in ("", r"\N", "\\N")]
            idx[tconst] = sorted(set(tokens))
    return idx


def _load_entity_index_from_csv(path: Path, key_col: str, entity_col: str) -> Dict[str, List[str]]:
    """Load entity index from a raw edge CSV file.

    Returns {} if the file does not exist or cannot be read.
    """
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, usecols=[key_col, entity_col], dtype=str)
        _BAD = {"", r"\N", "\\N"}
        idx: Dict[str, List[str]] = defaultdict(list)
        for tc, eid in zip(df[key_col], df[entity_col]):
            if str(eid).strip() not in _BAD:
                idx[str(tc)].append(str(eid).strip())
        return {k: sorted(set(v)) for k, v in idx.items()}
    except Exception:
        return {}


def _build_entity_index_from_count(df: pd.DataFrame, count_col: str) -> Dict[str, List[str]]:
    """
    Fallback when individual id lists are not available.
    Creates synthetic entity ids per title using the count column so that OOF
    encoding degrades gracefully (each title gets its own synthetic group).
    """
    if count_col not in df.columns:
        return {}
    idx: Dict[str, List[str]] = {}
    for tconst, cnt in zip(df["tconst"].astype(str),
                           pd.to_numeric(df[count_col], errors="coerce").fillna(0)):
        # synthetic id = tconst itself so each title is its own group
        idx[tconst] = [f"_synthetic_{tconst}"] if cnt > 0 else []
    return idx


def add_title_group_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """Compute title-group stats within the training set (no label leakage)."""
    out = train_df.copy()
    grp = out.groupby("canonical_title", dropna=False)
    out["title_group_size_train"] = grp["canonical_title"].transform("size").astype(float)

    if "startYear" in out.columns:
        out["title_unique_years_train"] = grp["startYear"].transform(
            lambda s: s.dropna().nunique()
        ).astype(float)
    else:
        out["title_unique_years_train"] = 1.0

    out["title_conflicting_years"] = (out["title_unique_years_train"] > 1).astype(float)
    return out


def add_title_similarity_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """TF-IDF cosine similarity of each title to hit and non-hit centroids."""
    out = train_df.copy()
    title_series = out.get("primaryTitle", pd.Series("", index=out.index)).fillna("").astype(str)

    if "label" not in out.columns:
        out["title_sim_to_hit"]     = 0.0
        out["title_sim_to_non_hit"] = 0.0
        out["title_sim_margin"]     = 0.0
        return out

    y = pd.to_numeric(out["label"], errors="coerce")
    hit_mask = y.eq(1).to_numpy()
    non_mask = y.eq(0).to_numpy()

    if hit_mask.sum() == 0 or non_mask.sum() == 0:
        out["title_sim_to_hit"]     = 0.0
        out["title_sim_to_non_hit"] = 0.0
        out["title_sim_margin"]     = 0.0
        return out

    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=5000)
    X = vec.fit_transform(title_series)

    if X.shape[1] == 0:
        out["title_sim_to_hit"]     = 0.0
        out["title_sim_to_non_hit"] = 0.0
        out["title_sim_margin"]     = 0.0
        return out

    hit_centroid = np.asarray(X[hit_mask].mean(axis=0))
    non_centroid = np.asarray(X[non_mask].mean(axis=0))
    sim_hit = cosine_similarity(X, hit_centroid).ravel()
    sim_non = cosine_similarity(X, non_centroid).ravel()

    out["title_sim_to_hit"]     = sim_hit
    out["title_sim_to_non_hit"] = sim_non
    out["title_sim_margin"]     = sim_hit - sim_non
    return out


def compute_oof_encoding(
    train_df: pd.DataFrame,
    entity_index: Dict[str, List[str]],
    n_splits: int = 5,
    smoothing: float = 20.0,
) -> Tuple[np.ndarray, Dict[str, float], float]:
    """
    Leak-safe OOF target encoding for multi-entity (many-to-many) columns.

    Returns
    -------
    oof        : per-row smoothed hit rate (never sees own fold)
    full_lookup: {entity_id: smoothed_hit_rate} fitted on ALL training data
    global_mean: overall positive rate
    """
    y        = train_df["label"].astype(int).to_numpy()
    tconsts  = train_df["tconst"].astype(str).to_numpy()
    gm       = float(np.mean(y))
    skf      = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof      = np.full(len(train_df), gm, dtype=float)

    for fit_idx, holdout_idx in skf.split(tconsts, y):
        sums: Dict[str, float]  = defaultdict(float)
        cnts: Dict[str, int]    = defaultdict(int)
        for i in fit_idx:
            for ent in entity_index.get(tconsts[i], []):
                sums[ent] += float(y[i])
                cnts[ent] += 1
        lk = {e: (sums[e] + smoothing * gm) / (cnts[e] + smoothing) for e in cnts}
        for i in holdout_idx:
            ents = entity_index.get(tconsts[i], [])
            if ents:
                oof[i] = float(np.mean([lk.get(e, gm) for e in ents]))

    # Full-data lookup for inference on val/test
    sums2: Dict[str, float] = defaultdict(float)
    cnts2: Dict[str, int]   = defaultdict(int)
    for i in range(len(tconsts)):
        for ent in entity_index.get(tconsts[i], []):
            sums2[ent] += float(y[i])
            cnts2[ent] += 1
    full_lookup = {e: (sums2[e] + smoothing * gm) / (cnts2[e] + smoothing) for e in cnts2}

    return oof, full_lookup, gm


def compute_oof_group_rate(
    keys: pd.Series,
    labels: pd.Series,
    n_splits: int = 5,
    smoothing: float = 20.0,
) -> Tuple[np.ndarray, float]:
    """OOF target encoding for a single categorical key column (canonical_title)."""
    y  = labels.astype(int).to_numpy()
    k  = keys.fillna("").astype(str).to_numpy()
    gm = float(np.mean(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.full(len(y), gm, dtype=float)

    for fit_idx, holdout_idx in skf.split(k, y):
        sums: Dict[str, float] = defaultdict(float)
        cnts: Dict[str, int]   = defaultdict(int)
        for i in fit_idx:
            sums[k[i]] += float(y[i])
            cnts[k[i]] += 1
        lk = {ki: (sums[ki] + smoothing * gm) / (cnts[ki] + smoothing) for ki in cnts}
        for i in holdout_idx:
            oof[i] = lk.get(k[i], gm)

    return oof, gm


# ── figure helpers ─────────────────────────────────────────────────────────────

def _fig_missingness(df: pd.DataFrame, feat_cols: List[str]) -> None:
    miss = df[feat_cols].isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        print("[f1] No missingness in feature columns — skipping missingness figure.")
        return
    n = len(miss)
    fig, ax = plt.subplots(figsize=(14, max(5, n * 0.45)))
    colors = [RED if v > 0.5 else ORG if v > 0.1 else Y for v in miss.values]
    bars = ax.barh(miss.index.tolist(), miss.values, color=colors, alpha=0.88, edgecolor="none")
    ax.set_xlabel("Missing rate", fontsize=11, color=TXT, labelpad=8)
    ax.set_title("Feature missingness rates (before imputation)", fontsize=13,
                 fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    ax.tick_params(axis="y", labelsize=9, labelcolor=MUT)
    ax.axvline(0.5, color=RED, linestyle="--", linewidth=1)
    ax.axvline(0.1, color=ORG, linestyle="--", linewidth=1)
    x_max = ax.get_xlim()[1]
    for bar, v in zip(bars, miss.values):
        ax.text(v + 0.005 * x_max, bar.get_y() + bar.get_height() / 2,
                f"{v:.1%}", va="center", fontsize=8, color=TXT)
    fig.tight_layout(pad=2.5)
    fig.savefig(FIG_DIR / "01_feature_missingness.png", dpi=130,
                bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close(fig)


def _fig_distributions(
    df: pd.DataFrame,
    cols: List[str],
    title: str,
    color: str = Y,
    filename: str = "02_base_distributions.png",
) -> None:
    cols = [c for c in cols if c in df.columns and df[c].notna().sum() > 0]
    if not cols:
        return
    n     = len(cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    for i, col in enumerate(cols):
        ax = axes[i]
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.nunique() <= 4:
            vc = vals.value_counts().sort_index()
            bars = ax.bar(vc.index.astype(str), vc.values, color=color, alpha=0.88,
                          edgecolor="none")
            y_max = ax.get_ylim()[1]
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005 * y_max,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=8, color=TXT)
        else:
            ax.hist(vals, bins=30, color=color, alpha=0.35, edgecolor="none")
        ax.set_title(col, fontsize=13, fontweight="bold", color=TXT, pad=12)
        ax.title.set_position([0.5, 1.02])
        ax.tick_params(axis="x", labelsize=9, labelcolor=MUT)
        miss_rate = df[col].isna().mean()
        ax.set_xlabel(f"missing={miss_rate:.1%}", fontsize=7.5, color=MUT, labelpad=8)
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(title, color=TXT, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(pad=2.5)
    fig.savefig(FIG_DIR / filename, dpi=130, bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close(fig)


def _fig_oof_diagram() -> None:
    """Illustrate the 5-fold OOF encoding procedure as a diagram."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("OOF Target Encoding — How It Works (5-Fold)", fontsize=13,
                 fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])

    fold_colors = [Y, GRN, BLU, ORG, PRP]
    for i, fc in enumerate(fold_colors):
        x = 0.3 + i * 1.85
        ax.add_patch(plt.Rectangle((x, 2.5), 1.5, 0.8, color=fc, alpha=0.3))
        ax.text(x + 0.75, 2.9, f"Fold {i+1}", ha="center", va="center",
                color=fc, fontsize=8, fontweight="bold")

    for i, fc in enumerate(fold_colors):
        for j, fc2 in enumerate(fold_colors):
            if j != i:
                x = 0.3 + j * 1.85
                ax.add_patch(plt.Rectangle((x, 1.3), 1.5, 0.7, color=fc2, alpha=0.15))
        x_hold = 0.3 + i * 1.85
        ax.add_patch(plt.Rectangle((x_hold, 1.3), 1.5, 0.7, color=fold_colors[i], alpha=0.8))
        ax.text(x_hold + 0.75, 1.65, "holdout", ha="center", va="center",
                color=BG, fontsize=7, fontweight="bold")
        ax.annotate("", xy=(x_hold + 0.75, 1.3), xytext=(x_hold + 0.75, 2.5),
                    arrowprops=dict(arrowstyle="-|>", color=TXT, lw=1.2))
        ax.text(x_hold + 0.75, 0.95, "fit rate\non train\nfolds",
                ha="center", va="center", color=TXT, fontsize=7, style="italic")
        ax.annotate("", xy=(x_hold + 0.75, 0.7), xytext=(x_hold + 0.75, 0.45),
                    arrowprops=dict(arrowstyle="-|>", color=GRN, lw=1.2))

    ax.text(5, 0.2,
            "OOF scores assembled: director_hit_rate / writer_hit_rate / canonical_title_hit_rate",
            ha="center", va="center", color=GRN, fontsize=9,
            bbox=dict(fc="#0a1a0a", ec=GRN, boxstyle="round,pad=0.4"))
    ax.text(5, 3.65, "Training set (n rows)", ha="center", color=TXT, fontsize=9)
    ax.text(5, 1.15, "score applied to holdout fold only (no leakage)",
            ha="center", color=MUT, fontsize=7.5)

    fig.tight_layout(pad=2.5)
    fig.savefig(FIG_DIR / "05_oof_diagram.png", dpi=130,
                bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close(fig)


def _fig_oof_distributions(train_df: pd.DataFrame) -> None:
    enc_cols = [c for c in ["director_hit_rate", "writer_hit_rate", "canonical_title_hit_rate"]
                if c in train_df.columns]
    if not enc_cols:
        print("[f1] No OOF encoding columns found — skipping OOF distribution figure.")
        return

    n = len(enc_cols)
    width = 18 if n >= 3 else (16 if n == 2 else 14)
    fig, axes = plt.subplots(1, n, figsize=(width, 5))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, enc_cols):
        vals = train_df[col].dropna()
        if "label" in train_df.columns:
            lbl = pd.to_numeric(train_df.loc[vals.index, "label"], errors="coerce")
            hit = vals[lbl == 1]
            non = vals[lbl == 0]
        else:
            hit = vals
            non = pd.Series(dtype=float)

        if len(non) > 0:
            ax.hist(non, bins=30, color=RED, alpha=0.35, edgecolor="none",
                    label="label=0", density=True)
        if len(hit) > 0:
            ax.hist(hit, bins=30, color=GRN, alpha=0.35, edgecolor="none",
                    label="label=1", density=True)

        ax.set_title(col, fontsize=13, fontweight="bold", color=TXT, pad=12)
        ax.title.set_position([0.5, 1.02])
        ax.tick_params(axis="x", labelsize=9, labelcolor=MUT)
        ax.legend(fontsize=9)

    fig.suptitle("OOF encoding distributions by label (train set)", color=TXT,
                 fontsize=13, fontweight="bold")
    fig.tight_layout(pad=2.5)
    fig.savefig(FIG_DIR / "06_oof_distributions.png", dpi=130,
                bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close(fig)


def _fig_auteur_derivation(df: pd.DataFrame) -> None:
    """Show is_auteur derivation: num_unique_directors==1 AND num_unique_writers==1."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, col, color in zip(
        axes,
        ["num_unique_directors", "num_unique_writers", "is_auteur"],
        [Y, BLU, GRN],
    ):
        if col in df.columns:
            vals = df[col].dropna()
            if vals.nunique() <= 8:
                vc = vals.value_counts().sort_index()
                bars = ax.bar(vc.index.astype(str), vc.values, color=color, alpha=0.88,
                              edgecolor="none")
                y_max = ax.get_ylim()[1]
                for bar in bars:
                    h = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005 * y_max,
                            f"{h:.2f}", ha="center", va="bottom", fontsize=8, color=TXT)
            else:
                ax.hist(vals, bins=20, color=color, alpha=0.35, edgecolor="none")
            ax.set_title(col, fontsize=13, fontweight="bold", color=TXT, pad=12)
            ax.title.set_position([0.5, 1.02])
        ax.tick_params(axis="x", labelsize=9, labelcolor=MUT)

    fig.suptitle(
        "Auteur flag: num_unique_directors==1 AND num_unique_writers==1",
        color=TXT, fontsize=13, fontweight="bold",
    )
    fig.tight_layout(pad=2.5)
    fig.savefig(FIG_DIR / "04_aggregates.png", dpi=130,
                bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close(fig)


# ── main run ───────────────────────────────────────────────────────────────────

def run(state: dict) -> dict:
    """
    Run candidate feature generation.

    Consumes state keys: train_df
    Produces state keys: train_feat, dir_lookup, dir_gm, wr_lookup, wr_gm
    """
    # ── 1. Load data ──────────────────────────────────────────────────────────
    train_df = state.get("train_df")
    if train_df is None:
        fp = PROC / "train_clean.parquet"
        if fp.exists():
            train_df = pd.read_parquet(fp)
            print(f"[f1] Loaded train_clean.parquet  shape={train_df.shape}")
        else:
            raise FileNotFoundError(
                f"train_clean.parquet not found at {fp}. Run data_cleaning pipeline first."
            )

    # ── 2. Coerce numeric columns ─────────────────────────────────────────────
    for col in ["startYear", "endYear", "runtimeMinutes", "numVotes", "numVotes_log1p"]:
        if col in train_df.columns:
            train_df[col] = pd.to_numeric(train_df[col], errors="coerce")

    # label must be integer 0/1
    train_df["label"] = pd.to_numeric(train_df["label"], errors="coerce").astype(float)

    # ── 3. Aggregate features from parquet columns ────────────────────────────
    train_feat = add_aggregate_from_parquet(train_df)

    # ── 4. Base features ──────────────────────────────────────────────────────
    train_feat = add_base_features(train_feat)

    # ── 5. Title group features (train-only stats) ────────────────────────────
    if "label" in train_feat.columns:
        train_feat = add_title_group_features(train_feat)

    # ── 6. OOF encodings ──────────────────────────────────────────────────────
    if "label" in train_feat.columns:
        # Director entity index — prefer raw CSV for real nconst IDs
        if "dir_ids" in train_feat.columns:
            dir_idx = _build_entity_index_from_ids(train_feat, "dir_ids")
        else:
            dir_idx = _load_entity_index_from_csv(
                RAW_CSV / "movie_directors.csv", "tconst", "director"
            )
            if dir_idx:
                print("[f1] Loaded director entity index from movie_directors.csv.")
            else:
                dir_idx = _build_entity_index_from_count(train_feat, "dir_count")
                print("[f1] WARNING: movie_directors.csv not found — director OOF degraded to global mean.")

        oof_dir, dir_lookup, dir_gm = compute_oof_encoding(train_feat, dir_idx)
        train_feat["director_hit_rate"] = oof_dir

        # Writer entity index — prefer raw CSV for real nconst IDs
        if "wri_ids" in train_feat.columns:
            wr_idx = _build_entity_index_from_ids(train_feat, "wri_ids")
        else:
            wr_idx = _load_entity_index_from_csv(
                RAW_CSV / "movie_writers.csv", "tconst", "writer"
            )
            if wr_idx:
                print("[f1] Loaded writer entity index from movie_writers.csv.")
            else:
                wr_idx = _build_entity_index_from_count(train_feat, "wri_count")
                print("[f1] WARNING: movie_writers.csv not found — writer OOF degraded to global mean.")

        oof_wr, wr_lookup, wr_gm = compute_oof_encoding(train_feat, wr_idx)
        train_feat["writer_hit_rate"] = oof_wr

        # Canonical title OOF
        oof_ct, _ct_gm = compute_oof_group_rate(
            train_feat["canonical_title"], train_feat["label"]
        )
        train_feat["canonical_title_hit_rate"] = oof_ct

        # Title similarity
        train_feat = add_title_similarity_features(train_feat)

        state["train_feat"]  = train_feat
        state["dir_lookup"]  = dir_lookup
        state["dir_gm"]      = dir_gm
        state["wr_lookup"]   = wr_lookup
        state["wr_gm"]       = wr_gm
    else:
        print("[f1] No label column — skipping OOF encodings.")
        dir_lookup, dir_gm, wr_lookup, wr_gm = {}, 0.0, {}, 0.0

    # ── 7. Build final feature column list ────────────────────────────────────
    all_feat_cols = (
        [
            "title_len", "title_word_count", "title_has_digit", "title_has_colon",
            "title_has_question", "title_upper_ratio", "has_original_title",
            "runtime_missing", "votes_missing", "start_missing", "end_missing",
            "startYear", "endYear", "year_span", "numVotes_log1p",
            "num_directors", "num_unique_directors", "num_writers", "num_unique_writers",
            "is_auteur",
        ]
        + (["director_hit_rate", "writer_hit_rate", "canonical_title_hit_rate"]
           if "director_hit_rate" in train_feat.columns else [])
        + (["title_group_size_train", "title_unique_years_train", "title_conflicting_years",
            "title_sim_to_hit", "title_sim_to_non_hit", "title_sim_margin"]
           if "title_group_size_train" in train_feat.columns else [])
    )
    all_feat_cols = [c for c in all_feat_cols if c in train_feat.columns]

    # ── 8. Figures ────────────────────────────────────────────────────────────
    print("[f1] Saving figures...")
    _fig_missingness(train_feat, all_feat_cols)

    base_num = ["title_len", "title_word_count", "title_upper_ratio",
                "startYear", "year_span", "numVotes_log1p"]
    _fig_distributions(train_feat, base_num,
                       "Base numeric features", color=Y,
                       filename="02_base_distributions.png")

    binary_cols = ["title_has_digit", "title_has_colon", "title_has_question",
                   "has_original_title", "runtime_missing", "votes_missing",
                   "start_missing", "end_missing"]
    _fig_distributions(train_feat, binary_cols,
                       "Binary feature flags", color=GRN,
                       filename="03_binary_flags.png")

    _fig_auteur_derivation(train_feat)   # saves 04_aggregates.png
    _fig_oof_diagram()                   # saves 05_oof_diagram.png
    _fig_oof_distributions(train_feat)   # saves 06_oof_distributions.png

    # ── 9. Save outputs ───────────────────────────────────────────────────────
    save_cols_meta = ["tconst"] + (["label"] if "label" in train_feat.columns else [])
    save_feat_cols = [c for c in all_feat_cols if c in train_feat.columns]

    out_df = train_feat[save_cols_meta + save_feat_cols].copy()

    out_df.to_parquet(OUT_FEAT / "features_train.parquet", index=False)
    out_df.to_csv(OUT_FEAT / "features_train.csv", index=False)
    print(
        f"[f1] Saved features_train.parquet + .csv  "
        f"({len(out_df)} rows x {len(save_feat_cols)} features)"
    )

    # train_feat holds the full working DataFrame (includes parquet metadata cols).
    # features_train is the trimmed output that f2 expects — matches what was saved
    # to disk so the state path and the file path are identical in content.
    state["train_feat"]      = train_feat
    state["features_train"]  = out_df
    return state


if __name__ == "__main__":
    run({})
    print("[f1] Done.")
