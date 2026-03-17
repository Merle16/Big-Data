#!/usr/bin/env python3
"""
g2_validate.py — PageRank Feature Validation & Visualisation
=============================================================

*** STANDALONE SCRIPT — run manually to validate g1_pagerank outputs ***

Produces 6 figures in pipeline/outputs/graph/:
  V1_top20_pagerank.png       — Top-20 people by PageRank (bar, colored by role)
  V2_pagerank_vs_hitrate.png  — Scatter: PageRank vs personal hit rate per person
  V3_network_subgraph.png     — Network subgraph of top-50 nodes (poster-ready)
  V4_auc_comparison.png       — Univariate AUC: PageRank features vs hit_rate baseline
  V5_distribution_shift.png   — Train vs val distributions (PSI check, percentile features)
  V6_psi_before_after.png     — Before/after KS: raw PageRank vs temporal within-role percentile

Also prints a summary table with Spearman correlations and univariate AUCs.

How to run
----------
  python pipeline/graph/g2_validate.py

Requirements
------------
  networkx  matplotlib  pandas  numpy  scikit-learn
  (all already in the project environment)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
PROC      = ROOT / "data" / "processed"
RAW       = ROOT / "data" / "raw" / "IMDB_external_csv"
PIPELINE  = ROOT / "pipeline"
OUT_FEAT  = PIPELINE / "outputs" / "features"
OUT_GRAPH = PIPELINE / "outputs" / "graph"
OUT_GRAPH.mkdir(parents=True, exist_ok=True)

# ── dark theme ─────────────────────────────────────────────────────────────────
BG  = "#0a0a0a"
CRD = "#111111"
BDR = "#252525"
TXT = "#e8e8e8"
MUT = "#666666"
Y   = "#F5C518"
GRN = "#47F3FF"
RED = "#e74c3c"
ORG = "#f39c12"
BLU = "#1848f5"
PRP = "#9b59b6"

ROLE_COLOR = {"director": Y, "writer": BLU, "actor": GRN, "actress": ORG}

mpl.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": CRD, "axes.edgecolor": BDR,
    "axes.labelcolor": TXT, "text.color": TXT, "xtick.color": MUT,
    "ytick.color": MUT, "grid.color": BDR, "grid.linewidth": 0.4,
    "legend.facecolor": CRD, "legend.edgecolor": BDR, "font.family": "sans-serif",
})


def _save(fig, fname: str) -> None:
    fig.savefig(OUT_GRAPH / fname, dpi=130, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  Saved {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_data() -> tuple:
    pr      = pd.read_csv(OUT_GRAPH / "pagerank_scores.csv")
    feat_tr = pd.read_parquet(OUT_GRAPH / "features_pagerank.parquet")
    feat_vl = pd.read_parquet(OUT_GRAPH / "features_pagerank_val.parquet")

    train   = pd.read_parquet(PROC / "train_clean.parquet")[["tconst", "label"]]
    val     = pd.read_parquet(PROC / "validation_hidden_clean.parquet")[["tconst"]]

    # Merge labels
    feat_tr = feat_tr.merge(train, on="tconst", how="left")

    # Merge existing hit-rate baseline if available
    baseline_cols = ["tconst", "director_hit_rate", "writer_hit_rate", "label"]
    try:
        base = pd.read_parquet(OUT_FEAT / "features_train_prepped.parquet")
        avail = [c for c in baseline_cols if c in base.columns]
        feat_tr = feat_tr.merge(base[avail], on="tconst", how="left", suffixes=("", "_base"))
    except Exception:
        pass

    # Principal data for network graph
    prin = pd.read_csv(RAW / "title_principals.csv")
    prin = prin[prin["category"].isin({"director", "writer", "actor", "actress"})]

    return pr, feat_tr, feat_vl, prin


# ══════════════════════════════════════════════════════════════════════════════
# Figure V1 — Top-20 people by PageRank
# ══════════════════════════════════════════════════════════════════════════════

def _fig_top20(pr: pd.DataFrame) -> None:
    top = (
        pr.dropna(subset=["primaryName"])
        .sort_values("pagerank", ascending=False)
        .head(20)
        .sort_values("pagerank", ascending=True)
        .reset_index(drop=True)
    )
    colors = [ROLE_COLOR.get(r, MUT) for r in top["category"]]

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(top["primaryName"], top["pagerank"], color=colors, alpha=0.88, edgecolor="none")
    x_max = ax.get_xlim()[1]
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002 * x_max, bar.get_y() + bar.get_height() / 2,
                f"{w:.5f}", va="center", ha="left", fontsize=8, color=TXT)

    patches = [mpatches.Patch(color=c, label=r) for r, c in ROLE_COLOR.items()]
    ax.legend(handles=patches, loc="lower right")
    ax.set_xlabel("PageRank score", fontsize=11, labelpad=8)
    ax.set_title("Top 20 people by collaboration-network PageRank",
                 fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout(pad=2.5)
    _save(fig, "V1_top20_pagerank.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure V2 — PageRank vs personal hit rate (per person)
# ══════════════════════════════════════════════════════════════════════════════

def _fig_scatter_vs_hitrate(pr: pd.DataFrame, feat_tr: pd.DataFrame,
                             prin: pd.DataFrame, train: pd.DataFrame) -> None:
    # Compute personal hit rate per nconst from training data
    prin_train = prin.merge(train[["tconst", "label"]], on="tconst", how="inner")
    person_stats = (
        prin_train.groupby("nconst")
        .agg(films=("tconst", "nunique"), hits=("label", "sum"), category=("category", "first"))
        .reset_index()
    )
    person_stats["hit_rate"] = person_stats["hits"] / person_stats["films"].clip(lower=1)
    person_stats = person_stats[person_stats["films"] >= 3]  # min 3 films for stability

    merged = person_stats.merge(pr[["nconst", "pagerank"]], on="nconst", how="inner")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    roles = [("director", axes[0]), ("actor", axes[1])]

    for role, ax in roles:
        sub = merged[merged["category"] == role]
        if sub.empty:
            continue
        ax.scatter(sub["hit_rate"], sub["pagerank"],
                   c=ROLE_COLOR.get(role, MUT), alpha=0.5, s=18, edgecolors="none")

        # Spearman r
        r, p = stats.spearmanr(sub["hit_rate"], sub["pagerank"])
        ax.set_title(f"{role.capitalize()}s — Spearman r={r:.3f} (p={p:.2e})",
                     fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Personal hit rate (train)", fontsize=10)
        ax.set_ylabel("PageRank score", fontsize=10)

        # Annotate top names
        top_pr = sub.nlargest(5, "pagerank").merge(pr[["nconst", "primaryName"]], on="nconst", how="left")
        for _, row in top_pr.iterrows():
            if pd.notna(row.get("primaryName")):
                ax.annotate(row["primaryName"],
                            xy=(row["hit_rate"], row["pagerank"]),
                            xytext=(5, 3), textcoords="offset points",
                            fontsize=7, color=TXT, alpha=0.85)

    fig.suptitle("PageRank vs personal hit rate — are they independent signals?",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(pad=2.5)
    _save(fig, "V2_pagerank_vs_hitrate.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure V3 — Network subgraph (top-50 nodes, poster-ready)
# ══════════════════════════════════════════════════════════════════════════════

def _fig_network(pr: pd.DataFrame, prin: pd.DataFrame, train: pd.DataFrame,
                 top_n: int = 40) -> None:
    """
    Poster-quality collaboration network.
    - Node colour = role (fixed, always readable from legend)
    - Node size   = PageRank score
    - Edge width & opacity scale with hit-collaboration weight (min visible)
    - Clear role legend, edge key annotation
    - No halo/glow — single crisp draw
    """
    import matplotlib.colors as mcolors

    # ── Select top-N named nodes ───────────────────────────────────────────────
    top_nodes = (
        pr.dropna(subset=["primaryName"])
        .sort_values("pagerank", ascending=False)
        .head(top_n)
    )
    node_set = set(top_nodes["nconst"])
    pr_max   = float(top_nodes["pagerank"].max())
    pr_min   = float(top_nodes["pagerank"].min())

    # ── Build edges ───────────────────────────────────────────────────────────
    prin_train = prin.merge(train[["tconst", "label"]], on="tconst", how="inner")
    G = nx.Graph()
    for nconst, row in top_nodes.set_index("nconst").iterrows():
        G.add_node(nconst, name=row["primaryName"], role=row["category"],
                   pr=row["pagerank"])

    for _, grp in prin_train.groupby("tconst"):
        people = list(dict.fromkeys(n for n in grp["nconst"] if n in node_set))
        w = float(grp["label"].iloc[0])
        for i, a in enumerate(people):
            for b in people[i + 1:]:
                if a == b:
                    continue
                if G.has_edge(a, b):
                    G[a][b]["weight"] += w
                else:
                    G.add_edge(a, b, weight=w)

    G.remove_edges_from(list(nx.selfloop_edges(G)))
    G.remove_nodes_from(list(nx.isolates(G)))
    if len(G.nodes) < 3:
        print("  [V3] Not enough connected top nodes — skipping network plot.")
        return

    # ── Layout ────────────────────────────────────────────────────────────────
    try:
        pos = nx.kamada_kawai_layout(G, weight=None)
    except Exception:
        pos = nx.spring_layout(G, seed=42, k=4.0 / np.sqrt(len(G.nodes)))

    pos = {n: (x * 2.4, y * 2.4) for n, (x, y) in pos.items()}

    # ── Node colours: fixed per role ──────────────────────────────────────────
    ROLE_COLORS = {
        "director": "#DBA506",
        "writer":   "#D0FEF5",
        "actor":    "#47F3FF",
        "actress":  "#C41E3D",
    }
    nodes        = list(G.nodes)
    node_pr_vals = [G.nodes[n].get("pr", pr_min) for n in nodes]
    node_roles   = [G.nodes[n].get("role", "actor") for n in nodes]
    node_colors  = [ROLE_COLORS.get(r, "#888888") for r in node_roles]
    # Size: 600 (lowest) → 5000 (highest)
    node_sizes   = [600 + 4400 * ((v - pr_min) / max(pr_max - pr_min, 1e-12)) ** 0.55
                    for v in node_pr_vals]

    # ── Edges: glow pass + crisp pass so even weak links are readable ──────────
    edge_list    = list(G.edges)
    edge_weights = [G[u][v].get("weight", 0) for u, v in edge_list]
    max_w        = max(edge_weights) if edge_weights else 1
    # Width: 1.0 min → 6.0 max (crisp line); glow is 3× wider, 15% alpha
    edge_widths  = [1.0 + 5.0 * (w / max_w) for w in edge_weights]
    edge_alphas  = [0.55 + 0.35 * (w / max_w) for w in edge_weights]  # 0.55 min
    def _edge_color(w: float) -> str:
        t = w / max_w
        # Low-weight: cool blue-grey; high-weight: warm IMDB yellow
        r = int(0x44 + t * (0xF5 - 0x44))
        g = int(0x66 + t * (0xC5 - 0x66))
        b = int(0x88 + t * (0x18 - 0x88))
        return f"#{r:02x}{g:02x}{b:02x}"
    edge_colors = [_edge_color(w) for w in edge_weights]

    # ── Draw ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(22, 18))
    ax.set_facecolor("#060810")
    fig.patch.set_facecolor("#060810")

    # Edges — glow pass first (wide, very transparent), then crisp line on top
    for (u, v), col, lw, a in zip(edge_list, edge_colors, edge_widths, edge_alphas):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        # Glow: 3× width, 15% alpha — makes even thin edges softly visible
        ax.plot([x0, x1], [y0, y1], color=col, linewidth=lw * 3.0,
                alpha=0.15, solid_capstyle="round", zorder=1)
        # Crisp line
        ax.plot([x0, x1], [y0, y1], color=col, linewidth=lw,
                alpha=a, solid_capstyle="round", zorder=2)

    # Nodes — glow pass (larger, transparent) + crisp draw on top
    for n, col, sz in zip(nodes, node_colors, node_sizes):
        x, y = pos[n]
        # Glow halo: 2.8× size, low alpha
        ax.scatter(x, y, s=sz * 2.8, color=col, alpha=0.18, zorder=3,
                   linewidths=0, edgecolors="none")
        # Crisp node
        ax.scatter(x, y, s=sz, color=col, alpha=0.93, zorder=4,
                   linewidths=1.2, edgecolors="#000000")

    # Labels — white, font size scales with PageRank
    for n, sz, pr_v in zip(nodes, node_sizes, node_pr_vals):
        x, y = pos[n]
        t  = (pr_v - pr_min) / max(pr_max - pr_min, 1e-12)
        fs = 8 + 5 * t
        ax.text(x, y + 0.075 * (sz / 2000) ** 0.35, G.nodes[n].get("name", ""),
                ha="center", va="bottom", fontsize=fs,
                color="#ffffff", fontweight="bold", zorder=4,
                bbox=dict(boxstyle="round,pad=0.2", fc="#060810",
                          ec="none", alpha=0.6))

    # ── Role legend (bottom left) ──────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(facecolor=ROLE_COLORS["director"], edgecolor="#000",
                       linewidth=0.8, label="Director"),
        mpatches.Patch(facecolor=ROLE_COLORS["writer"],   edgecolor="#000",
                       linewidth=0.8, label="Writer"),
        mpatches.Patch(facecolor=ROLE_COLORS["actor"],    edgecolor="#000",
                       linewidth=0.8, label="Actor"),
        mpatches.Patch(facecolor=ROLE_COLORS["actress"],  edgecolor="#000",
                       linewidth=0.8, label="Actress"),
    ]
    legend = ax.legend(
        handles=legend_patches, loc="lower left",
        fontsize=12, framealpha=0.85,
        facecolor="#111111", edgecolor="#333333",
        title="Role", title_fontsize=12,
    )
    legend.get_title().set_color(TXT)
    for text in legend.get_texts():
        text.set_color(TXT)

    # Edge key (bottom right)
    ax.annotate(
        "Node size  ∝  PageRank score\n"
        "Edge width & brightness  ∝  hit-collaboration strength",
        xy=(0.99, 0.01), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=10, color="#aaaaaa",
        bbox=dict(boxstyle="round,pad=0.5", fc="#111111", ec="#333333", alpha=0.85),
    )

    ax.set_title(
        "Hit-making collaboration network  ·  "
        f"Top {len(G.nodes)} connected filmmakers\n"
        "Computed with Spark GraphFrames PageRank on IMDB collaboration graph",
        fontsize=14, fontweight="bold", color=TXT, pad=20,
    )
    ax.axis("off")
    fig.tight_layout(pad=1.0)
    _save(fig, "V3_network_subgraph.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure V4 — Univariate AUC comparison
# ══════════════════════════════════════════════════════════════════════════════

def _fig_auc_comparison(feat_tr: pd.DataFrame) -> None:
    feat_cols = [
        "director_pagerank", "writer_pagerank",
        "top_actor_pagerank", "avg_cast_pagerank",
    ]
    baseline_cols = ["director_hit_rate", "writer_hit_rate"]

    all_cols = feat_cols + [c for c in baseline_cols if c in feat_tr.columns]
    labeled  = feat_tr.dropna(subset=["label"])
    y        = labeled["label"].astype(int).values

    aucs  = []
    names = []
    for col in all_cols:
        if col not in labeled.columns:
            continue
        x = labeled[col].fillna(labeled[col].median()).values
        auc = roc_auc_score(y, x)
        aucs.append(max(auc, 1 - auc))
        names.append(col)

    # Sort descending
    order  = np.argsort(aucs)[::-1]
    aucs   = [aucs[i]  for i in order]
    names  = [names[i] for i in order]
    colors = [Y if n in feat_cols else BLU for n in names]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(names, aucs, color=colors, alpha=0.88, edgecolor="none")
    ax.axhline(0.5,  color=MUT, linestyle="--", linewidth=1, label="random")
    ax.axhline(0.55, color=ORG, linestyle="--", linewidth=0.8)
    ax.axhline(0.60, color=GRN, linestyle="--", linewidth=0.8)
    ax.set_ylim(0.45, max(aucs) * 1.08)
    for bar, v in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9, color=TXT)
    ax.set_ylabel("Univariate ROC-AUC", fontsize=11)
    ax.set_title("PageRank features vs hit-rate baseline — univariate AUC",
                 fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(axis="x", labelsize=9, rotation=15)
    patches = [
        mpatches.Patch(color=Y,   label="PageRank (novel)"),
        mpatches.Patch(color=BLU, label="Hit-rate (baseline)"),
    ]
    ax.legend(handles=patches)
    fig.tight_layout(pad=2.5)
    _save(fig, "V4_auc_comparison.png")

    return dict(zip(names, aucs))


# ══════════════════════════════════════════════════════════════════════════════
# Figure V5 — Distribution shift train vs val (PSI)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_psi(tr: pd.Series, vl: pd.Series, n_bins: int = 10) -> float:
    tr = tr.dropna(); vl = vl.dropna()
    if tr.empty or vl.empty:
        return float("nan")
    edges = np.unique(np.quantile(tr, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return float("nan")
    tr_d = pd.cut(tr, bins=edges, include_lowest=True).value_counts(normalize=True).sort_index()
    vl_d = pd.cut(vl, bins=edges, include_lowest=True).value_counts(normalize=True).sort_index()
    al = pd.concat([tr_d, vl_d], axis=1).fillna(1e-6).clip(lower=1e-6)
    al.columns = ["t", "v"]
    return float(((al["v"] - al["t"]) * np.log(al["v"] / al["t"])).sum())


def _compute_ks(tr: pd.Series, vl: pd.Series) -> float:
    """Kolmogorov-Smirnov statistic — correct metric for continuous features.

    PSI uses discrete bins and is dominated by fallback spikes (0.5) in high-
    cardinality percentile features.  KS measures the maximum CDF distance
    between train and validation, which is distribution-model-free and does
    not require binning.  KS < 0.10 = stable, > 0.20 = notable shift.
    """
    tr = tr.dropna(); vl = vl.dropna()
    if tr.empty or vl.empty:
        return float("nan")
    ks_stat, _ = stats.ks_2samp(tr.values, vl.values)
    return float(ks_stat)


def _fig_distribution_shift(feat_tr: pd.DataFrame, feat_vl: pd.DataFrame) -> dict:
    cols = ["director_pagerank", "writer_pagerank", "top_actor_pagerank", "avg_cast_pagerank"]
    ks_vals  = {c: _compute_ks( feat_tr[c], feat_vl[c]) for c in cols if c in feat_tr.columns}
    psi_vals = {c: _compute_psi(feat_tr[c], feat_vl[c]) for c in cols if c in feat_tr.columns}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        ax  = axes[i]
        ks  = ks_vals.get(col, float("nan"))
        psi = psi_vals.get(col, float("nan"))
        tr  = feat_tr[col].dropna()
        vl  = feat_vl[col].dropna()

        bins = np.linspace(min(tr.min(), vl.min()), max(tr.max(), vl.max()), 30)
        ax.hist(tr, bins=bins, alpha=0.6, color=BLU, label="train", density=True)
        ax.hist(vl, bins=bins, alpha=0.6, color=ORG, label="val",   density=True)

        # KS is the primary metric for continuous features; PSI shown for reference
        ks_color = RED if ks > 0.20 else ORG if ks > 0.10 else GRN
        ax.set_title(f"{col}\nKS = {ks:.4f}  (PSI = {psi:.4f})",
                     fontsize=10, fontweight="bold", color=ks_color)
        ax.set_xlabel("Temporal percentile score", fontsize=9)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Temporal PageRank feature distribution: train vs val\n"
        "KS statistic — correct metric for continuous features  "
        "(KS < 0.10 = stable | > 0.20 = notable shift)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout(pad=2.5)
    _save(fig, "V5_distribution_shift.png")

    return ks_vals


# ══════════════════════════════════════════════════════════════════════════════
# Figure V6 — Before / after PSI: raw PageRank vs within-role percentile
# ══════════════════════════════════════════════════════════════════════════════

def _build_raw_features(
    pr: pd.DataFrame,
    prin: pd.DataFrame,
    split_tconsts: pd.Series,
) -> pd.DataFrame:
    """
    Reconstruct raw PageRank feature values (pre-percentile) from
    pagerank_scores.csv, so we can compare PSI before vs after the
    percentile transformation.  This mirrors _build_features() in g1_pagerank.py
    but uses the raw 'pagerank' column.
    """
    pr_lookup = pr.dropna(subset=["pagerank"]).set_index("nconst")["pagerank"].to_dict()
    global_median = float(pr["pagerank"].median())
    prin_by_film  = prin.groupby("tconst")

    rows = []
    for tconst in split_tconsts:
        if tconst not in prin_by_film.groups:
            rows.append({"tconst": tconst,
                         "director_pagerank_raw": global_median,
                         "writer_pagerank_raw":   global_median,
                         "top_actor_pagerank_raw": global_median,
                         "avg_cast_pagerank_raw": global_median})
            continue
        grp = prin_by_film.get_group(tconst)

        def _mean(cat):
            sub = grp[grp["category"] == cat]["nconst"]
            vals = [pr_lookup.get(n, global_median) for n in sub]
            return float(np.mean(vals)) if vals else global_median

        def _top_actor():
            actors = grp[grp["category"].isin({"actor","actress"})].sort_values("ordering")
            if actors.empty: return global_median
            return float(pr_lookup.get(actors.iloc[0]["nconst"], global_median))

        def _avg_cast():
            actors = (grp[grp["category"].isin({"actor","actress"})]
                      .sort_values("ordering").head(3))
            if actors.empty: return global_median
            return float(np.mean([pr_lookup.get(n, global_median) for n in actors["nconst"]]))

        rows.append({"tconst": tconst,
                     "director_pagerank_raw":  _mean("director"),
                     "writer_pagerank_raw":    _mean("writer"),
                     "top_actor_pagerank_raw": _top_actor(),
                     "avg_cast_pagerank_raw":  _avg_cast()})
    return pd.DataFrame(rows)


def _fig_psi_before_after(
    pr: pd.DataFrame,
    prin: pd.DataFrame,
    feat_tr: pd.DataFrame,
    feat_vl: pd.DataFrame,
) -> None:
    """
    Side-by-side KS statistic: raw PageRank score vs temporal within-role percentile.

    Justification
    -------------
    PSI relies on discrete binning and is not appropriate for continuous high-
    cardinality features: on percentile-transformed features the 0.5 fallback
    spike dominates the bin counts, producing artificially high PSI values that
    do not reflect genuine distribution shift.

    The Kolmogorov-Smirnov (KS) statistic measures the maximum distance between
    two empirical CDFs — it is binning-free and the standard choice for continuous
    features.  KS < 0.10 = stable, > 0.20 = notable shift.

    Raw PageRank scores have high KS because absolute network centrality grows
    over time (more collaborations = larger scores for recent films).  The temporal
    year-band construction fixes this: each film is scored relative to the network
    known at the time of release, giving comparable scales across eras.
    """
    cols   = ["director_pagerank", "writer_pagerank", "top_actor_pagerank", "avg_cast_pagerank"]
    labels = ["director", "writer", "top actor", "avg cast"]

    # ── Build raw features for train and val ──────────────────────────────────
    raw_tr = _build_raw_features(pr, prin, feat_tr["tconst"])
    raw_vl = _build_raw_features(pr, prin, feat_vl["tconst"])

    # ── KS: raw vs temporal percentile ───────────────────────────────────────
    ks_raw = [_compute_ks(raw_tr[f"{c}_raw"], raw_vl[f"{c}_raw"]) for c in cols]
    ks_pct = [_compute_ks(feat_tr[c],          feat_vl[c])         for c in cols]

    x     = np.arange(len(cols))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_raw = ax.bar(x - width/2, ks_raw, width, color=RED, alpha=0.85,
                      label="Raw PageRank score")
    bars_pct = ax.bar(x + width/2, ks_pct, width, color=GRN, alpha=0.85,
                      label="Temporal within-role percentile")

    ax.axhline(0.10, color=ORG, linewidth=1.2, linestyle="--", alpha=0.7,
               label="KS 0.10 — monitor")
    ax.axhline(0.20, color=RED, linewidth=1.2, linestyle="--", alpha=0.7,
               label="KS 0.20 — notable shift")

    for bar in bars_raw:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.003, f"{h:.3f}",
                ha="center", va="bottom", fontsize=9, color=TXT)
    for bar in bars_pct:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.003, f"{h:.3f}",
                ha="center", va="bottom", fontsize=9, color=TXT)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("KS statistic (train vs validation)", fontsize=11)
    ax.set_title(
        "Distribution stability before vs after temporal year-band construction\n"
        "KS statistic — correct metric for continuous features  (lower = more stable)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _save(fig, "V6_psi_before_after.png")


# ══════════════════════════════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary(feat_tr: pd.DataFrame, feat_vl: pd.DataFrame,
                   psi_vals: dict, auc_vals: dict) -> None:
    cols = ["director_pagerank", "writer_pagerank", "top_actor_pagerank", "avg_cast_pagerank"]
    labeled = feat_tr.dropna(subset=["label"])
    y = labeled["label"].astype(int).values

    print("\n" + "=" * 80)
    print(f"{'Feature':<25} {'AUC':>8} {'Spearman r':>12} {'KS':>8} {'Status':>10}")
    print("-" * 80)
    for col in cols:
        if col not in labeled.columns:
            continue
        x    = labeled[col].fillna(labeled[col].median()).values
        r, _ = stats.spearmanr(y, x)
        auc  = auc_vals.get(col, float("nan"))
        ks   = psi_vals.get(col, float("nan"))   # psi_vals now holds KS values
        status = "KEEP" if (auc > 0.55 and ks < 0.20) else "REVIEW" if auc > 0.52 else "WEAK"
        print(f"{col:<25} {auc:>8.4f} {r:>12.4f} {ks:>8.4f} {status:>10}")
    print("=" * 72)

    # Baseline comparison
    for bc in ["director_hit_rate", "writer_hit_rate"]:
        if bc in labeled.columns:
            x    = labeled[bc].fillna(labeled[bc].median()).values
            auc  = roc_auc_score(y, x)
            auc  = max(auc, 1 - auc)
            r, _ = stats.spearmanr(y, x)
            print(f"  Baseline {bc:<20} AUC={auc:.4f}  Spearman={r:.4f}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run() -> None:
    print("=" * 60)
    print("Graph Pipeline — g2: PageRank Validation & Visualisation")
    print("=" * 60)

    # Check g1 has been run
    if not (OUT_GRAPH / "pagerank_scores.csv").exists():
        print("[g2_validate] ERROR: pagerank_scores.csv not found.")
        print("  Run g1_pagerank.py first:  python pipeline/graph/g1_pagerank.py --networkx")
        return

    print("[g2_validate] Loading data...")
    pr, feat_tr, feat_vl, prin = _load_data()
    train = pd.read_parquet(PROC / "train_clean.parquet")[["tconst", "label"]]

    print("[g2_validate] Figure V1 — Top-20 PageRank bar chart...")
    _fig_top20(pr)

    print("[g2_validate] Figure V2 — PageRank vs personal hit rate scatter...")
    _fig_scatter_vs_hitrate(pr, feat_tr, prin, train)

    print("[g2_validate] Figure V3 — Network subgraph...")
    _fig_network(pr, prin, train, top_n=50)

    print("[g2_validate] Figure V4 — Univariate AUC comparison...")
    auc_vals = _fig_auc_comparison(feat_tr)

    print("[g2_validate] Figure V5 — Distribution shift (PSI)...")
    psi_vals = _fig_distribution_shift(feat_tr, feat_vl)

    print("[g2_validate] Figure V6 — PSI before/after percentile transformation...")
    _fig_psi_before_after(pr, prin, feat_tr, feat_vl)

    _print_summary(feat_tr, feat_vl, psi_vals, auc_vals)

    print(f"[g2_validate] All figures saved to {OUT_GRAPH}")


if __name__ == "__main__":
    run()
    print("[g2_validate] Done.")
