#!/usr/bin/env python3
"""
g2_validate.py — PageRank Feature Validation & Visualisation
=============================================================

*** STANDALONE SCRIPT — run manually to validate g1_pagerank outputs ***

Produces 5 figures in pipeline/outputs/graph/:
  V1_top20_pagerank.png       — Top-20 people by PageRank (bar, colored by role)
  V2_pagerank_vs_hitrate.png  — Scatter: PageRank vs personal hit rate per person
  V3_network_subgraph.png     — Network subgraph of top-50 nodes (poster-ready)
  V4_auc_comparison.png       — Univariate AUC: PageRank features vs hit_rate baseline
  V5_distribution_shift.png   — Train vs val distributions (PSI check)

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
GRN = "#2ecc71"
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
    - Nodes graded from dark grey (low PR) → IMDB yellow (highest PR)
    - Edge colour and alpha scale with hit-collaboration weight
    - kamada_kawai layout for clean, even spacing
    - Glow effect: draw each node twice (large+dim then small+bright)
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

    # Spread positions wider for breathing room
    pos = {n: (x * 2.2, y * 2.2) for n, (x, y) in pos.items()}

    # ── Node colours: dark grey → IMDB yellow gradient by PageRank ────────────
    # Role base colours — blended toward IMDB yellow as PageRank rises
    ROLE_BASE = {
        "director": np.array(mcolors.to_rgb("#DBA506")),
        "writer":   np.array(mcolors.to_rgb("#D0FEF5")),
        "actor":    np.array(mcolors.to_rgb("#007991")),
        "actress":  np.array(mcolors.to_rgb("#C41E3D")),
    }
    IMDB_Y = np.array(mcolors.to_rgb("#F5C518"))
    DIM    = np.array(mcolors.to_rgb("#1a1a1a"))

    def _node_color(node_pr: float, role: str) -> tuple:
        t    = (node_pr - pr_min) / max(pr_max - pr_min, 1e-12)
        t    = t ** 0.5
        base = ROLE_BASE.get(role, DIM)
        # Low PR → role colour; high PR → IMDB yellow
        return tuple(base + t * (IMDB_Y - base))

    nodes        = list(G.nodes)
    node_pr_vals = [G.nodes[n].get("pr", pr_min) for n in nodes]
    node_roles   = [G.nodes[n].get("role", "actor") for n in nodes]
    node_colors  = [_node_color(v, r) for v, r in zip(node_pr_vals, node_roles)]
    # Size: 400 (lowest) → 4000 (highest)
    node_sizes   = [400 + 3600 * ((v - pr_min) / max(pr_max - pr_min, 1e-12)) ** 0.6
                    for v in node_pr_vals]

    # ── Edge colours: weight → alpha + brightness ─────────────────────────────
    edge_list    = list(G.edges)
    edge_weights = [G[u][v].get("weight", 0) for u, v in edge_list]
    max_w        = max(edge_weights) if edge_weights else 1
    edge_alphas  = [0.15 + 0.75 * (w / max_w) for w in edge_weights]
    edge_widths  = [0.5  + 4.5  * (w / max_w) for w in edge_weights]
    # Colour: dim silver → bright IMDB yellow
    def _edge_color(w: float) -> str:
        t = w / max_w
        r = int(0x33 + t * (0xF5 - 0x33))
        g = int(0x33 + t * (0xC5 - 0x33))
        b = int(0x33 + t * (0x18 - 0x33))
        return f"#{r:02x}{g:02x}{b:02x}"
    edge_colors = [_edge_color(w) for w in edge_weights]

    # ── Draw ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(22, 18))
    ax.set_facecolor("#050505")
    fig.patch.set_facecolor("#050505")

    # Edges — draw dim wide first (glow), then crisp on top
    for (u, v), col, w, a in zip(edge_list, edge_colors, edge_widths, edge_alphas):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], color=col, linewidth=w * 2.2,
                alpha=a * 0.25, solid_capstyle="round", zorder=1)
        ax.plot([x0, x1], [y0, y1], color=col, linewidth=w,
                alpha=a, solid_capstyle="round", zorder=2)

    # Nodes — glow halo then crisp fill
    for n, col, sz in zip(nodes, node_colors, node_sizes):
        x, y = pos[n]
        ax.scatter(x, y, s=sz * 2.5, color=col, alpha=0.18, zorder=3,
                   linewidths=0)
        ax.scatter(x, y, s=sz, color=col, alpha=0.95, zorder=4,
                   linewidths=0.8, edgecolors="#000000")

    # Labels — offset slightly above node, font size scales with PR
    for n, sz, pr_v in zip(nodes, node_sizes, node_pr_vals):
        x, y = pos[n]
        t    = (pr_v - pr_min) / max(pr_max - pr_min, 1e-12)
        fs   = 7 + 5 * t    # 7pt → 12pt
        name = G.nodes[n].get("name", "")
        ax.text(x, y + 0.07 * (sz / 1500) ** 0.4, name,
                ha="center", va="bottom", fontsize=fs,
                color=TXT, fontweight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="#050505",
                          ec="none", alpha=0.55))

    # Legend — role colours removed; use gradient annotation instead
    ax.annotate("Node colour & size = PageRank score\n"
                "Edge brightness = hit-collaboration strength",
                xy=(0.01, 0.01), xycoords="axes fraction",
                fontsize=9, color=MUT,
                bbox=dict(boxstyle="round,pad=0.4", fc="#111111", ec=BDR, alpha=0.8))

    ax.set_title("Hit-making collaboration network\n"
                 "IMDB yellow = highest PageRank  ·  "
                 f"Top {len(G.nodes)} connected filmmakers",
                 fontsize=15, fontweight="bold", color=TXT, pad=18)
    ax.axis("off")
    fig.tight_layout(pad=0.5)
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


def _fig_distribution_shift(feat_tr: pd.DataFrame, feat_vl: pd.DataFrame) -> dict:
    cols   = ["director_pagerank", "writer_pagerank", "top_actor_pagerank", "avg_cast_pagerank"]
    psi_vals = {c: _compute_psi(feat_tr[c], feat_vl[c]) for c in cols if c in feat_tr.columns}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        ax  = axes[i]
        psi = psi_vals.get(col, float("nan"))
        tr  = feat_tr[col].dropna()
        vl  = feat_vl[col].dropna()

        bins = np.linspace(min(tr.min(), vl.min()), max(tr.max(), vl.max()), 30)
        ax.hist(tr, bins=bins, alpha=0.6, color=BLU, label="train", density=True)
        ax.hist(vl, bins=bins, alpha=0.6, color=ORG, label="val",   density=True)

        psi_color = RED if psi > 0.25 else ORG if psi > 0.10 else GRN
        ax.set_title(f"{col}\nPSI = {psi:.4f}", fontsize=10, fontweight="bold", color=psi_color)
        ax.set_xlabel("PageRank score", fontsize=9)
        ax.legend(fontsize=8)

    fig.suptitle("PageRank feature distribution: train vs val\n"
                 "PSI < 0.10 = stable  |  0.10–0.25 = monitor  |  > 0.25 = unstable",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout(pad=2.5)
    _save(fig, "V5_distribution_shift.png")

    return psi_vals


# ══════════════════════════════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary(feat_tr: pd.DataFrame, feat_vl: pd.DataFrame,
                   psi_vals: dict, auc_vals: dict) -> None:
    cols = ["director_pagerank", "writer_pagerank", "top_actor_pagerank", "avg_cast_pagerank"]
    labeled = feat_tr.dropna(subset=["label"])
    y = labeled["label"].astype(int).values

    print("\n" + "=" * 72)
    print(f"{'Feature':<25} {'AUC':>8} {'Spearman r':>12} {'PSI':>8} {'Status':>10}")
    print("-" * 72)
    for col in cols:
        if col not in labeled.columns:
            continue
        x   = labeled[col].fillna(labeled[col].median()).values
        r, p = stats.spearmanr(y, x)
        auc  = auc_vals.get(col, float("nan"))
        psi  = psi_vals.get(col, float("nan"))
        status = "KEEP" if (auc > 0.55 and psi < 0.25) else "REVIEW" if auc > 0.52 else "WEAK"
        print(f"{col:<25} {auc:>8.4f} {r:>12.4f} {psi:>8.4f} {status:>10}")
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

    _print_summary(feat_tr, feat_vl, psi_vals, auc_vals)

    print(f"[g2_validate] All figures saved to {OUT_GRAPH}")


if __name__ == "__main__":
    run()
    print("[g2_validate] Done.")
