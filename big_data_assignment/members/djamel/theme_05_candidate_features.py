#!/usr/bin/env python3
"""
Theme 05 — Candidate Feature Generation
========================================
Shows exactly:
  • How each feature group is computed (base / aggregate / title)
  • Feature motivation per column
  • Distribution histograms for every numeric feature
  • Missingness rates before imputation/capping
  • How OOF target encoding works (diagram + distribution)
  • The is_auteur flag derivation
  • Outputs a self-contained HTML report

Reads  : outputs_restart/movies_clean.csv
         outputs_restart/directors_clean.csv
         outputs_restart/writers_clean.csv
Writes : outputs_restart/theme_05_candidate_features.html
         outputs_restart/features_train.csv  (feature matrix, no label)
"""
from __future__ import annotations

import base64
import re
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold

# ── paths ──────────────────────────────────────────────────────────────────────
MEMBER = Path(__file__).resolve().parent
OUT    = MEMBER / "outputs_restart"
OUT.mkdir(exist_ok=True)

# ── style ──────────────────────────────────────────────────────────────────────
Y   = "#F5C518"
B   = "#1848f5"
BG  = "#0a0a0a"
CRD = "#141414"
TXT = "#ffffff"
MUT = "#888888"
GRN = "#2ecc71"
RED = "#e74c3c"
ORG = "#f39c12"
PRP = "#9b59b6"

SEED = 42

FEATURE_MOTIVATION = {
    "startYear":               "Release era signal; film markets and rating behavior change over time.",
    "endYear":                 "Series/end timing signal — retained as candidate; feature_selection drops it.",
    "runtimeMinutes_capped":   "Runtime effect with outlier control to avoid extreme-value distortion.",
    "numVotes_log1p":          "Popularity proxy; log transform handles heavy-tailed vote counts.",
    "numVotes_log1p_capped":   "Popularity proxy; log+capping handles heavy-tailed vote counts.",
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

CSS = f"""
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:{BG};color:{TXT};font-family:'Segoe UI',sans-serif;padding:24px;line-height:1.6}}
h1{{color:{Y};font-size:2rem;margin-bottom:6px}}
h2{{color:{Y};font-size:1.3rem;margin:0 0 12px}}
h3{{color:{ORG};font-size:1rem;margin:12px 0 6px}}
.subtitle{{color:{MUT};margin-bottom:28px;font-size:.95rem}}
.card{{background:{CRD};border:1px solid #222;border-radius:12px;padding:20px;margin-bottom:24px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start}}
.grid3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;align-items:start}}
.kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px}}
.kpi{{background:{CRD};border:1px solid #2a2a2a;border-radius:10px;padding:16px;text-align:center}}
.kpi .val{{font-size:1.8rem;font-weight:700;color:{Y}}}
.kpi .lbl{{color:{MUT};font-size:.8rem;margin-top:4px}}
table{{width:100%;border-collapse:collapse;font-size:.82rem}}
th{{background:{Y};color:#111;padding:8px 10px;text-align:left}}
td{{padding:7px 10px;border-bottom:1px solid #222;vertical-align:top}}
tr:hover td{{background:#1e1e1e}}
img{{max-width:100%;border-radius:8px;margin-top:8px}}
.note{{background:#1a1a2e;border-left:4px solid {Y};padding:10px 14px;
       border-radius:4px;font-size:.85rem;margin:10px 0;color:#ccc}}
.green{{color:{GRN};font-weight:600}} .red{{color:{RED};font-weight:600}}
.tag{{display:inline-block;padding:2px 8px;border-radius:12px;font-size:.75rem;font-weight:600;margin:1px}}
.tag-base{{background:#1a3a1a;color:{GRN}}}
.tag-agg{{background:#1a1a3a;color:{B}}}
.tag-enc{{background:#3a1a1a;color:{ORG}}}
.tag-title{{background:#2a1a2a;color:{PRP}}}
"""


def _b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def _img(fig): return f'<img src="data:image/png;base64,{_b64(fig)}">'
def _card(title, body, color=Y): return f'<div class="card"><h2 style="color:{color}">{title}</h2>{body}</div>'
def _kpi(val, lbl): return f'<div class="kpi"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>'
def _page(title, subtitle, kpis, sections):
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{title}</title><style>{CSS}</style></head><body>
<h1>{title}</h1><p class="subtitle">{subtitle}</p>
<div class="kpi-grid">{kpis}</div>{"".join(sections)}</body></html>"""


# ── feature engineering (inlined, no src/ imports) ─────────────────────────────

def canonicalize_title(title: str) -> str:
    if title is None:
        return ""
    text = str(title).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return text.strip()


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["primaryTitle"] = out["primaryTitle"].fillna("")
    out["originalTitle"] = out.get("originalTitle", pd.Series("", index=out.index)).fillna("")
    out["canonical_title"] = out["primaryTitle"].map(canonicalize_title)
    out["title_len"] = out["primaryTitle"].astype(str).str.len().astype(float)
    out["title_word_count"] = out["primaryTitle"].astype(str).str.split().str.len().fillna(0).astype(float)
    out["title_has_digit"] = out["primaryTitle"].astype(str).str.contains(r"\d", regex=True).astype(float)
    out["title_has_colon"] = out["primaryTitle"].astype(str).str.contains(":", regex=False).astype(float)
    out["title_has_question"] = out["primaryTitle"].astype(str).str.contains(r"\?", regex=True).astype(float)
    title_len_safe = out["title_len"].replace(0, np.nan)
    out["title_upper_ratio"] = (out["primaryTitle"].astype(str).str.count(r"[A-Z]") / title_len_safe).fillna(0.0)
    out["has_original_title"] = out["originalTitle"].astype(str).str.strip().ne("").astype(float)
    out["runtime_missing"] = out["runtimeMinutes"].isna().astype(float)
    out["votes_missing"] = out["numVotes"].isna().astype(float)
    out["start_missing"] = out["startYear"].isna().astype(float)
    out["end_missing"] = out["endYear"].isna().astype(float)
    out["year_span"] = (
        (out["endYear"] - out["startYear"])
        .where(out["startYear"].notna() & out["endYear"].notna(), 0.0)
        .clip(lower=0)
    )
    out["numVotes_log1p"] = np.log1p(out["numVotes"].clip(lower=0))
    return out


def add_aggregate_features(movies: pd.DataFrame, directors: pd.DataFrame, writers: pd.DataFrame) -> pd.DataFrame:
    d_agg = directors.groupby("tconst").agg(
        num_directors=("director_id", "size"),
        num_unique_directors=("director_id", "nunique"),
    ).reset_index()
    w_agg = writers.groupby("tconst").agg(
        num_writers=("writer_id", "size"),
        num_unique_writers=("writer_id", "nunique"),
    ).reset_index()
    feat = movies.merge(d_agg, on="tconst", how="left").merge(w_agg, on="tconst", how="left")
    for col in ["num_directors", "num_unique_directors", "num_writers", "num_unique_writers"]:
        feat[col] = feat[col].fillna(0).astype(float)
    feat["is_auteur"] = ((feat["num_unique_directors"] == 1) & (feat["num_unique_writers"] == 1)).astype(float)
    return feat


def add_title_group_features(train_df: pd.DataFrame) -> pd.DataFrame:
    out = train_df.copy()
    grp = out.groupby("canonical_title", dropna=False)
    out["title_group_size_train"] = grp["canonical_title"].transform("size").astype(float)
    if "startYear" in out.columns:
        out["title_unique_years_train"] = grp["startYear"].transform(lambda s: s.dropna().nunique()).astype(float)
    else:
        out["title_unique_years_train"] = 1.0
    out["title_conflicting_years"] = (out["title_unique_years_train"] > 1).astype(float)
    return out


def add_title_similarity_features(train_df: pd.DataFrame) -> pd.DataFrame:
    out = train_df.copy()
    title_series = out.get("primaryTitle", pd.Series("", index=out.index)).fillna("").astype(str)
    if "label" not in out.columns:
        out["title_sim_to_hit"] = 0.0
        out["title_sim_to_non_hit"] = 0.0
        out["title_sim_margin"] = 0.0
        return out

    y = pd.to_numeric(out["label"], errors="coerce")
    hit_mask = y.eq(1).to_numpy()
    non_mask = y.eq(0).to_numpy()
    if hit_mask.sum() == 0 or non_mask.sum() == 0:
        out["title_sim_to_hit"] = 0.0
        out["title_sim_to_non_hit"] = 0.0
        out["title_sim_margin"] = 0.0
        return out

    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=5000,
    )
    X = vec.fit_transform(title_series)
    if X.shape[1] == 0:
        out["title_sim_to_hit"] = 0.0
        out["title_sim_to_non_hit"] = 0.0
        out["title_sim_margin"] = 0.0
        return out

    hit_centroid = np.asarray(X[hit_mask].mean(axis=0))
    non_centroid = np.asarray(X[non_mask].mean(axis=0))
    sim_hit = cosine_similarity(X, hit_centroid).ravel()
    sim_non = cosine_similarity(X, non_centroid).ravel()
    out["title_sim_to_hit"] = sim_hit
    out["title_sim_to_non_hit"] = sim_non
    out["title_sim_margin"] = sim_hit - sim_non
    return out


def compute_oof_encoding(train_df, entity_index, n_splits=5, smoothing=20.0):
    y = train_df["label"].astype(int).to_numpy()
    tconsts = train_df["tconst"].astype(str).to_numpy()
    global_mean = float(np.mean(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.full(len(train_df), global_mean, dtype=float)
    for fit_idx, holdout_idx in skf.split(tconsts, y):
        sums: Dict[str, float] = defaultdict(float)
        counts: Dict[str, int] = defaultdict(int)
        for i in fit_idx:
            for ent in entity_index.get(tconsts[i], []):
                sums[ent] += float(y[i])
                counts[ent] += 1
        lookup = {e: (sums[e] + smoothing * global_mean) / (counts[e] + smoothing) for e in counts}
        for i in holdout_idx:
            ents = entity_index.get(tconsts[i], [])
            if ents:
                oof[i] = float(np.mean([lookup.get(e, global_mean) for e in ents]))
    sums2: Dict[str, float] = defaultdict(float)
    counts2: Dict[str, int] = defaultdict(int)
    for i in range(len(tconsts)):
        for ent in entity_index.get(tconsts[i], []):
            sums2[ent] += float(y[i])
            counts2[ent] += 1
    full_lookup = {e: (sums2[e] + smoothing * global_mean) / (counts2[e] + smoothing) for e in counts2}
    return oof, full_lookup, global_mean


def compute_oof_group_rate(keys, labels, n_splits=5, smoothing=20.0):
    y = labels.astype(int).to_numpy()
    k = keys.fillna("").astype(str).to_numpy()
    gm = float(np.mean(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.full(len(y), gm, dtype=float)
    for fit_idx, holdout_idx in skf.split(k, y):
        sums: Dict[str, float] = defaultdict(float)
        counts: Dict[str, int] = defaultdict(int)
        for i in fit_idx:
            sums[k[i]] += float(y[i])
            counts[k[i]] += 1
        lk = {ki: (sums[ki] + smoothing * gm) / (counts[ki] + smoothing) for ki in counts}
        for i in holdout_idx:
            oof[i] = lk.get(k[i], gm)
    return oof, gm


# ── figures ────────────────────────────────────────────────────────────────────

FEATURE_GROUPS = {
    "base":       ["title_len", "title_word_count", "title_upper_ratio", "startYear", "year_span", "numVotes_log1p"],
    "binary":     ["title_has_digit", "title_has_colon", "title_has_question", "has_original_title",
                   "runtime_missing", "votes_missing", "start_missing", "end_missing"],
    "aggregates": ["num_directors", "num_unique_directors", "num_writers", "num_unique_writers", "is_auteur"],
    "encodings":  ["director_hit_rate", "writer_hit_rate", "canonical_title_hit_rate"],
}

GROUP_COLORS = {"base": Y, "binary": GRN, "aggregates": B, "encodings": ORG}
GROUP_TAGS   = {"base": "tag-base", "binary": "tag-base", "aggregates": "tag-agg", "encodings": "tag-enc"}


def _fig_missingness(df: pd.DataFrame, feat_cols: List[str]) -> str:
    miss = df[feat_cols].isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        return "<p style='color:#888'>No missingness in feature columns.</p>"
    fig, ax = plt.subplots(figsize=(10, max(3, len(miss) * 0.4)), facecolor=BG)
    ax.set_facecolor(CRD)
    colors = [RED if v > 0.5 else ORG if v > 0.1 else Y for v in miss.values]
    bars = ax.barh(miss.index.tolist(), miss.values, color=colors)
    ax.set_xlabel("Missing rate", color=TXT)
    ax.set_title("Feature missingness rates (before imputation)", color=TXT, fontsize=12)
    ax.tick_params(colors=TXT, labelsize=8)
    for spine in ax.spines.values(): spine.set_color(MUT)
    ax.axvline(0.5, color=RED, linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(0.1, color=ORG, linestyle="--", linewidth=1, alpha=0.5)
    for bar, v in zip(bars, miss.values):
        ax.text(v + 0.005, bar.get_y() + bar.get_height()/2, f"{v:.1%}",
                va="center", color=TXT, fontsize=7)
    fig.tight_layout()
    return _img(fig)


def _fig_distributions(df: pd.DataFrame, cols: List[str], title: str, color=Y) -> str:
    cols = [c for c in cols if c in df.columns and df[c].notna().sum() > 0]
    if not cols:
        return ""
    n = len(cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 2.8), facecolor=BG)
    axes = np.array(axes).flatten() if n > 1 else [axes]
    for i, col in enumerate(cols):
        ax = axes[i]
        ax.set_facecolor(CRD)
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.nunique() <= 4:
            vc = vals.value_counts().sort_index()
            ax.bar(vc.index.astype(str), vc.values, color=color, alpha=0.85)
        else:
            ax.hist(vals, bins=30, color=color, alpha=0.85, edgecolor="none")
        ax.set_title(col, color=TXT, fontsize=9)
        ax.tick_params(colors=TXT, labelsize=7)
        for sp in ax.spines.values(): sp.set_color(MUT)
        ax.set_facecolor(CRD)
        miss_rate = df[col].isna().mean()
        ax.set_xlabel(f"missing={miss_rate:.1%}", color=MUT, fontsize=7)
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(title, color=TXT, fontsize=11, y=1.01)
    fig.tight_layout()
    return _img(fig)


def _fig_oof_diagram() -> str:
    """Illustrate the OOF encoding procedure."""
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    ax.set_title("OOF Target Encoding — How It Works (5-Fold)", color=TXT, fontsize=12, pad=10)

    fold_colors = [Y, GRN, B, ORG, PRP]
    for i, fc in enumerate(fold_colors):
        x = 0.3 + i * 1.85
        ax.add_patch(plt.Rectangle((x, 2.5), 1.5, 0.8, color=fc, alpha=0.3))
        ax.text(x + 0.75, 2.9, f"Fold {i+1}", ha="center", va="center", color=fc, fontsize=8, fontweight="bold")

    for i, fc in enumerate(fold_colors):
        # train folds = all others
        for j, fc2 in enumerate(fold_colors):
            if j != i:
                x = 0.3 + j * 1.85
                ax.add_patch(plt.Rectangle((x, 1.3), 1.5, 0.7, color=fc2, alpha=0.15))
        x_hold = 0.3 + i * 1.85
        ax.add_patch(plt.Rectangle((x_hold, 1.3), 1.5, 0.7, color=fold_colors[i], alpha=0.8))
        ax.text(x_hold + 0.75, 1.65, "holdout", ha="center", va="center", color=BG, fontsize=7, fontweight="bold")
        ax.annotate("", xy=(x_hold + 0.75, 1.3), xytext=(x_hold + 0.75, 2.5),
                    arrowprops=dict(arrowstyle="-|>", color=TXT, lw=1.2))
        ax.text(x_hold + 0.75, 0.95, "fit rate\non train\nfolds", ha="center", va="center",
                color=TXT, fontsize=7, style="italic")
        ax.annotate("", xy=(x_hold + 0.75, 0.7), xytext=(x_hold + 0.75, 0.45),
                    arrowprops=dict(arrowstyle="-|>", color=GRN, lw=1.2))

    ax.text(5, 0.2, "OOF scores assembled → director_hit_rate / writer_hit_rate / canonical_title_hit_rate",
            ha="center", va="center", color=GRN, fontsize=9,
            bbox=dict(fc="#0a1a0a", ec=GRN, boxstyle="round,pad=0.4"))
    ax.text(5, 3.65, "Training set (n rows)", ha="center", color=TXT, fontsize=9)
    ax.text(5, 1.15, "← score applied to holdout fold only (no leakage) →", ha="center", color=MUT, fontsize=8)
    fig.tight_layout()
    return _img(fig)


def _fig_auteur_derivation(df: pd.DataFrame) -> str:
    """Show is_auteur derivation: num_unique_directors=1 AND num_unique_writers=1."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), facecolor=BG)
    for ax, col, color in zip(axes,
                               ["num_unique_directors", "num_unique_writers", "is_auteur"],
                               [Y, B, GRN]):
        ax.set_facecolor(CRD)
        if col in df.columns:
            vals = df[col].dropna()
            if vals.nunique() <= 8:
                vc = vals.value_counts().sort_index()
                ax.bar(vc.index.astype(str), vc.values, color=color, alpha=0.85)
            else:
                ax.hist(vals, bins=20, color=color, alpha=0.85, edgecolor="none")
            ax.set_title(col, color=TXT, fontsize=10)
        ax.tick_params(colors=TXT, labelsize=8)
        for sp in ax.spines.values(): sp.set_color(MUT)
    fig.suptitle("Auteur flag derivation: num_unique_directors==1 AND num_unique_writers==1", color=TXT, fontsize=11)
    fig.tight_layout()
    return _img(fig)


def _motivation_table(feat_cols: List[str]) -> str:
    rows = ""
    group_map = {}
    for grp, cols in FEATURE_GROUPS.items():
        for c in cols:
            group_map[c] = grp
    for feat in feat_cols:
        grp = group_map.get(feat, "other")
        tag_cls = GROUP_TAGS.get(grp, "tag-agg")
        tag_html = f'<span class="tag {tag_cls}">{grp}</span>'
        mot = FEATURE_MOTIVATION.get(feat, "—")
        rows += f"<tr><td><code>{feat}</code></td><td>{tag_html}</td><td>{mot}</td></tr>"
    return f"""<table>
<tr><th>Feature</th><th>Group</th><th>Motivation</th></tr>
{rows}</table>"""


def _fig_encoding_distributions(train_df: pd.DataFrame) -> str:
    enc_cols = [c for c in ["director_hit_rate", "writer_hit_rate", "canonical_title_hit_rate"] if c in train_df.columns]
    if not enc_cols:
        return "<p style='color:#888'>No encoding columns computed.</p>"
    fig, axes = plt.subplots(1, len(enc_cols), figsize=(len(enc_cols)*4, 3.5), facecolor=BG)
    if len(enc_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, enc_cols):
        ax.set_facecolor(CRD)
        vals = train_df[col].dropna()
        hit = vals[train_df.loc[vals.index, "label"] == 1] if "label" in train_df.columns else vals
        non = vals[train_df.loc[vals.index, "label"] == 0] if "label" in train_df.columns else pd.Series(dtype=float)
        ax.hist(non, bins=30, color=RED, alpha=0.6, label="label=0", density=True)
        ax.hist(hit, bins=30, color=GRN, alpha=0.6, label="label=1", density=True)
        ax.set_title(col, color=TXT, fontsize=9)
        ax.tick_params(colors=TXT, labelsize=7)
        for sp in ax.spines.values(): sp.set_color(MUT)
        ax.legend(facecolor=CRD, edgecolor=MUT, labelcolor=TXT, fontsize=7)
    fig.suptitle("OOF encoding distributions by label (train set)", color=TXT, fontsize=11)
    fig.tight_layout()
    return _img(fig)


# ── main run ───────────────────────────────────────────────────────────────────

def run(state: dict) -> dict:
    # Load cleaned data
    movies_clean = state.get("movies_clean")
    directors_clean = state.get("directors_clean")
    writers_clean = state.get("writers_clean")
    train_df = state.get("train_df")

    # Fall back to CSVs
    for key, fname in [("movies_clean", "movies_clean.csv"),
                       ("directors_clean", "directors_clean.csv"),
                       ("writers_clean", "writers_clean.csv"),
                       ("train_df", "train_clean.csv")]:
        if locals()[key] is None:
            fp = OUT / fname
            if fp.exists():
                locals()[key]  # silence lint
                if key == "movies_clean":   movies_clean  = pd.read_csv(fp)
                elif key == "directors_clean": directors_clean = pd.read_csv(fp)
                elif key == "writers_clean":   writers_clean   = pd.read_csv(fp)
                elif key == "train_df":        train_df        = pd.read_csv(fp)

    if movies_clean is None:
        raise FileNotFoundError("movies_clean not found — run theme_04 first.")
    if directors_clean is None:
        raise FileNotFoundError("directors_clean not found — run theme_04 first.")
    if writers_clean is None:
        raise FileNotFoundError("writers_clean not found — run theme_04 first.")

    # Fix numeric cols that may have been read as strings
    for col in ["startYear", "endYear", "runtimeMinutes", "numVotes"]:
        if col in movies_clean.columns:
            movies_clean[col] = pd.to_numeric(movies_clean[col], errors="coerce")
    if train_df is not None:
        for col in ["startYear", "endYear", "runtimeMinutes", "numVotes"]:
            if col in train_df.columns:
                train_df[col] = pd.to_numeric(train_df[col], errors="coerce")

    # Step 1: aggregate features
    movies_feat = add_aggregate_features(movies_clean, directors_clean, writers_clean)

    # Step 2: base features
    movies_feat = add_base_features(movies_feat)

    # Step 3: OOF encodings (only if train_df available)
    if train_df is None and "label" in movies_clean.columns:
        # Notebook runs themes independently; recover train subset from labeled rows.
        train_df = movies_clean[movies_clean["label"].notna()].copy()
        print(f"[theme_05] train_df fallback from movies_clean labels: {len(train_df)} rows")

    if train_df is not None:
        for col in ["startYear", "endYear", "runtimeMinutes", "numVotes"]:
            if col in train_df.columns:
                train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        train_feat = add_aggregate_features(train_df, directors_clean, writers_clean)
        train_feat = add_base_features(train_feat)
        train_feat = add_title_group_features(train_feat)

        if "label" in train_feat.columns:
            # Director OOF
            dir_idx = directors_clean.groupby("tconst")["director_id"].apply(
                lambda s: sorted(set(s.dropna().astype(str)))
            ).to_dict()
            oof_dir, dir_lookup, dir_gm = compute_oof_encoding(train_feat, dir_idx)
            train_feat["director_hit_rate"] = oof_dir

            # Writer OOF
            wr_idx = writers_clean.groupby("tconst")["writer_id"].apply(
                lambda s: sorted(set(s.dropna().astype(str)))
            ).to_dict()
            oof_wr, wr_lookup, wr_gm = compute_oof_encoding(train_feat, wr_idx)
            train_feat["writer_hit_rate"] = oof_wr

            # Canonical title OOF
            oof_ct, ct_gm = compute_oof_group_rate(train_feat["canonical_title"], train_feat["label"])
            train_feat["canonical_title_hit_rate"] = oof_ct
            train_feat = add_title_similarity_features(train_feat)

            state["train_feat"] = train_feat
            state["dir_lookup"] = dir_lookup
            state["dir_gm"] = dir_gm
            state["wr_lookup"] = wr_lookup
            state["wr_gm"] = wr_gm
    else:
        train_feat = movies_feat  # fallback for distributions

    # Collect all feature columns
    all_feat_cols = (
        ["title_len", "title_word_count", "title_has_digit", "title_has_colon",
         "title_has_question", "title_upper_ratio", "has_original_title",
         "runtime_missing", "votes_missing", "start_missing", "end_missing",
         "startYear", "endYear", "year_span", "numVotes_log1p",
         "num_directors", "num_unique_directors", "num_writers", "num_unique_writers", "is_auteur"] +
        (["director_hit_rate", "writer_hit_rate", "canonical_title_hit_rate"]
         if "director_hit_rate" in train_feat.columns else []) +
        (["title_group_size_train", "title_unique_years_train", "title_conflicting_years",
          "title_sim_to_hit", "title_sim_to_non_hit", "title_sim_margin"]
         if "title_group_size_train" in train_feat.columns else [])
    )
    all_feat_cols = [c for c in all_feat_cols if c in train_feat.columns]

    # KPIs
    kpis = (
        _kpi(str(len(all_feat_cols)),       "Total candidate<br>features")
      + _kpi("5",                            "Feature groups:<br>base/bin/agg/enc/title")
      + _kpi(str(len([c for c in ["director_hit_rate","writer_hit_rate","canonical_title_hit_rate"] if c in train_feat.columns])),
             "OOF-encoded<br>features")
      + _kpi(f"{train_feat[all_feat_cols].isna().mean().mean():.1%}", "Mean missingness<br>across features")
    )

    sections = []

    # 1. Motivation table
    sections.append(_card("1. Feature registry — motivation per feature", _motivation_table(all_feat_cols)))

    # 2. Base feature distributions
    base_num = ["title_len", "title_word_count", "title_upper_ratio", "startYear", "year_span", "numVotes_log1p"]
    sections.append(_card("2. Base numeric features — distributions", f"""
<div class="note">These are computed directly from movie metadata — no edges, no labels, no leakage.</div>
{_fig_distributions(train_feat, base_num, "Base numeric features", color=Y)}
"""))

    # 3. Binary flag distributions
    binary_cols = ["title_has_digit", "title_has_colon", "title_has_question", "has_original_title",
                   "runtime_missing", "votes_missing", "start_missing", "end_missing"]
    sections.append(_card("3. Binary flags — value counts", f"""
<div class="note">Binary (0/1) features. Missingness flags encode whether a column was absent — <strong>they carry signal themselves</strong>.</div>
{_fig_distributions(train_feat, binary_cols, "Binary feature flags", color=GRN)}
"""))

    # 4. Aggregate feature distributions + auteur
    agg_cols = ["num_directors", "num_unique_directors", "num_writers", "num_unique_writers"]
    sections.append(_card("4. Aggregate features (director/writer counts) + is_auteur", f"""
<div class="note">Computed by grouping the many-to-many edge tables by tconst. Movies with no edge data get 0.</div>
{_fig_distributions(train_feat, agg_cols, "Director/writer count features", color=B)}
{_fig_auteur_derivation(train_feat)}
"""))

    # 5. OOF encoding diagram + distributions
    oof_section_body = f"""
<div class="note">
<strong>Why OOF?</strong> If we computed director_hit_rate = fraction of a director's movies that are hits
using the whole training set and then trained on the same set, the model would see a "perfect" hit rate
for a director who appears only once (because that rate is computed on their own label).
OOF splits the training set into 5 folds: for each fold, the hit rate is computed on the OTHER 4 folds
and applied to the holdout. This ensures that the encoding never sees the fold's own labels.
</div>
{_fig_oof_diagram()}
{_fig_encoding_distributions(train_feat)}
"""
    sections.append(_card("5. OOF target encodings — leakage-safe construction", oof_section_body))

    # 6. Missingness map
    sections.append(_card("6. Missingness rates before imputation/capping", f"""
<div class="note">
<strong>endYear</strong> has structural missingness (~90%) because most movies have no defined end date.
The feature_selection stage will explicitly drop the numeric endYear and retain <code>end_missing</code> + <code>year_span</code>.
</div>
{_fig_missingness(train_feat, all_feat_cols)}
"""))

    # Save features CSV
    if "label" in train_feat.columns:
        save_cols = [c for c in all_feat_cols if c in train_feat.columns]
        train_feat[["tconst"] + save_cols + ["label"]].to_csv(OUT / "features_train.csv", index=False)
        state["features_train"] = train_feat
        print(f"[theme_05] Saved features_train.csv ({len(train_feat)} rows × {len(save_cols)} features)")

    html = _page(
        "Theme 05 — Candidate Feature Generation",
        "base · binary flags · aggregate counts · OOF encodings · motivation registry",
        kpis, sections,
    )
    (OUT / "theme_05_candidate_features.html").write_text(html, encoding="utf-8")
    print(f"[theme_05] Wrote {OUT}/theme_05_candidate_features.html")
    return state


if __name__ == "__main__":
    run({})
    print("[theme_05] Done.")
