#!/usr/bin/env python3
"""
g1_pagerank.py — Collaboration Network PageRank via Spark GraphFrames
======================================================================

*** STANDALONE SCRIPT — not part of the automatic pipeline ***
*** Review and validate with your team before integrating.  ***

Purpose
-------
Build a success-weighted collaboration graph from IMDB data and compute
PageRank for every director, writer, and actor. The intuition: people
who frequently collaborate on *hit* films occupy central positions in
the network and are more likely to be involved in future hits.

Graph structure
---------------
  Vertices : every unique person (nconst) across directors, writers,
             and actors/actresses in title_principals.csv
  Edges    : person A ↔ person B for every film they share, with
             weight = label of that film (1.0 = hit, 0.0 = flop).
             Self-loops are excluded.

Features produced
-----------------
  director_pagerank    — PageRank of the film's director(s), averaged
  writer_pagerank      — PageRank of the film's writer(s), averaged
  top_actor_pagerank   — PageRank of the top-billed actor (ordering=1)
  avg_cast_pagerank    — Mean PageRank of top-3 billed actors

Outputs (written to pipeline/outputs/graph/)
--------------------------------------------
  pagerank_scores.csv        — nconst, name, role, pagerank
  features_pagerank.parquet  — tconst + 4 PageRank features (train split)
  features_pagerank_val.parquet
  features_pagerank_test.parquet

How to run
----------
  # From big_data_assignment/
  python pipeline/graph/g1_pagerank.py

  # Optional flags
  python pipeline/graph/g1_pagerank.py --reset-prob 0.15 --tol 0.01
  python pipeline/graph/g1_pagerank.py --max-iter 20 --spark-cores 4

Requirements
------------
  pyspark>=3.3  graphframes   (pip install pyspark graphframes)
  The graphframes jar must be on the Spark classpath — the script
  handles this automatically via the packages argument.

Notes for the team
------------------
- This script uses only the TRAINING split labels to build the graph.
  Val/test splits receive PageRank scores via a left-join — no leakage.
- PageRank is computed on the undirected graph (edges go both ways).
  Edge weights represent hit-collaboration strength.
- If pyspark/graphframes is not installed, the script falls back to a
  NetworkX implementation so you can still validate the feature values
  locally without a Spark cluster.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]   # big_data_assignment/
RAW        = ROOT / "data" / "raw" / "IMDB_external_csv"
PROC       = ROOT / "data" / "processed"
PIPELINE   = ROOT / "pipeline"
OUT_GRAPH  = PIPELINE / "outputs" / "graph"
OUT_GRAPH.mkdir(parents=True, exist_ok=True)

# ── config ─────────────────────────────────────────────────────────────────────
try:
    from pipeline.config import CFG as _CFG
except ImportError:
    import yaml as _yaml
    _cfg_path = ROOT / "config.yaml"
    _CFG: dict = _yaml.safe_load(_cfg_path.read_text(encoding="utf-8")) if _cfg_path.exists() else {}

SEED = _CFG.get("global", {}).get("seed", 42)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df, test_df) with tconst + label + startYear."""
    train = pd.read_parquet(PROC / "train_clean.parquet")[["tconst", "label", "startYear"]]
    val   = pd.read_parquet(PROC / "validation_hidden_clean.parquet")[["tconst", "startYear"]]
    test  = pd.read_parquet(PROC / "test_hidden_clean.parquet")[["tconst", "startYear"]]
    # Coerce startYear to int, fill missing with 0 (will get fallback)
    for df in (train, val, test):
        df["startYear"] = pd.to_numeric(df["startYear"], errors="coerce").fillna(0).astype(int)
    return train, val, test


def _load_people() -> pd.DataFrame:
    """Load title_principals filtered to directors, writers, actors/actresses."""
    prin = pd.read_csv(RAW / "title_principals.csv")
    keep_cats = {"director", "writer", "actor", "actress"}
    prin = prin[prin["category"].isin(keep_cats)][
        ["tconst", "nconst", "ordering", "category"]
    ].copy()
    prin["ordering"] = pd.to_numeric(prin["ordering"], errors="coerce").fillna(99).astype(int)
    return prin


def _load_names() -> pd.DataFrame:
    nb = pd.read_csv(RAW / "name_basics.csv")[["nconst", "primaryName"]]
    return nb


def _build_edges(principals: pd.DataFrame, train_labels: pd.DataFrame) -> pd.DataFrame:
    """
    For every film in the training set, create pairwise edges between all
    people who worked on it. Edge weight = film label (1=hit, 0=flop).
    Returns DataFrame with columns: src, dst, weight.
    """
    # Attach label to principals (train only — no leakage)
    prin_train = principals.merge(train_labels, on="tconst", how="inner")

    edges = []
    for tconst, grp in prin_train.groupby("tconst"):
        people  = grp["nconst"].unique().tolist()   # unique() already deduplicates
        weight  = float(grp["label"].iloc[0])
        for i, a in enumerate(people):
            for b in people[i + 1:]:
                if a == b:   # safety guard against self-loops
                    continue
                edges.append({"src": a, "dst": b, "weight": weight})
                edges.append({"src": b, "dst": a, "weight": weight})  # undirected

    return pd.DataFrame(edges)


# ══════════════════════════════════════════════════════════════════════════════
# PageRank implementations
# ══════════════════════════════════════════════════════════════════════════════

def _pagerank_spark(
    edges_df: pd.DataFrame,
    vertices: list[str],
    reset_prob: float = 0.15,
    tol: float = 0.01,
    max_iter: int = 20,
    spark_cores: int = 4,
) -> pd.DataFrame:
    """
    Compute PageRank using Spark GraphFrames.
    Returns DataFrame with columns: nconst, pagerank.
    """
    from pyspark.sql import SparkSession

    print("[g1_pagerank] Starting Spark session...")
    spark = (
        SparkSession.builder
        .appName("CollaborationGraphPageRank")
        .master(f"local[{spark_cores}]")
        .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.2-s_2.12")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    from graphframes import GraphFrame

    # Vertices
    v_df = spark.createDataFrame(
        pd.DataFrame({"id": vertices})
    )

    # Edges — GraphFrames requires 'src' and 'dst' columns
    e_df = spark.createDataFrame(edges_df[["src", "dst", "weight"]])

    g = GraphFrame(v_df, e_df)

    print(f"[g1_pagerank] Running PageRank "
          f"(resetProbability={reset_prob}, tol={tol}, maxIter={max_iter})...")
    # GraphFrames requires exactly one of maxIter or tol — use maxIter for reproducibility
    results = g.pageRank(resetProbability=reset_prob, maxIter=max_iter)

    pr = (
        results.vertices
        .select("id", "pagerank")
        .toPandas()
        .rename(columns={"id": "nconst"})
    )

    spark.stop()
    print(f"[g1_pagerank] Spark PageRank complete — {len(pr)} nodes scored.")
    return pr


def _pagerank_networkx(
    edges_df: pd.DataFrame,
    vertices: list[str],
    reset_prob: float = 0.15,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> pd.DataFrame:
    """
    NetworkX fallback — identical logic, no Spark required.
    Used for local validation without a Spark installation.
    """
    import networkx as nx

    print("[g1_pagerank] Using NetworkX fallback (no Spark)...")
    G = nx.DiGraph()
    G.add_nodes_from(vertices)
    for _, row in edges_df.iterrows():
        if G.has_edge(row["src"], row["dst"]):
            G[row["src"]][row["dst"]]["weight"] += row["weight"]
        else:
            G.add_edge(row["src"], row["dst"], weight=row["weight"])

    pr_dict = nx.pagerank(
        G,
        alpha=1 - reset_prob,
        tol=tol,
        max_iter=max_iter,
        weight="weight",
    )
    pr = pd.DataFrame(list(pr_dict.items()), columns=["nconst", "pagerank"])
    print(f"[g1_pagerank] NetworkX PageRank complete — {len(pr)} nodes scored.")
    return pr


def _compute_pagerank(
    edges_df: pd.DataFrame,
    vertices: list[str],
    reset_prob: float,
    tol: float,
    max_iter: int,
    spark_cores: int,
    force_networkx: bool = False,
) -> pd.DataFrame:
    if not force_networkx:
        try:
            return _pagerank_spark(edges_df, vertices, reset_prob, tol, max_iter, spark_cores)
        except ImportError:
            print("[g1_pagerank] pyspark/graphframes not found — falling back to NetworkX.")
    return _pagerank_networkx(edges_df, vertices, reset_prob, tol=1e-6, max_iter=max_iter)


# ══════════════════════════════════════════════════════════════════════════════
# Feature assembly
# ══════════════════════════════════════════════════════════════════════════════

def _compute_role_percentiles(
    pr: pd.DataFrame,
    principals: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """
    Compute within-role percentile rank (0–1) for every person's PageRank.

    Why percentile instead of raw score
    ------------------------------------
    Raw PageRank is confounded by two things:
      1. Genre network density — studio-system genres (action, blockbuster)
         have denser co-credit networks, so everyone in those genres scores
         higher regardless of quality.
      2. Distribution shift — the absolute PageRank scale shifts between
         the training and validation splits (PSI > 0.25 on raw features),
         hurting generalisation.

    A within-role percentile answers "how central is this person relative
    to their peers?" rather than giving an absolute network score.
    This makes the feature:
      • Stable across splits (percentiles are always 0–1)
      • Genre-neutral (a drama director at p90 is comparable to an action
        director at p90)
      • Robust to graph size changes (new people enter at p50 fallback)

    Pooling
    -------
    actor and actress share one percentile pool — they compete for the
    same roles and their network positions are directly comparable.

    Returns: {role: {nconst: percentile_rank (0–1)}}
    """
    role_pct: dict[str, dict[str, float]] = {}

    for role in ("director", "writer"):
        nconsts = set(principals[principals["category"] == role]["nconst"].unique())
        sub = pr[pr["nconst"].isin(nconsts)].copy()
        if sub.empty:
            role_pct[role] = {}
        else:
            sub["pct"] = sub["pagerank"].rank(pct=True)
            role_pct[role] = sub.set_index("nconst")["pct"].to_dict()

    # actors + actresses share one pool
    act_nconsts = set(
        principals[principals["category"].isin({"actor", "actress"})]["nconst"].unique()
    )
    sub = pr[pr["nconst"].isin(act_nconsts)].copy()
    if sub.empty:
        act_pct: dict[str, float] = {}
    else:
        sub["pct"] = sub["pagerank"].rank(pct=True)
        act_pct = sub.set_index("nconst")["pct"].to_dict()
    role_pct["actor"]   = act_pct
    role_pct["actress"] = act_pct

    return role_pct


def _build_features(
    split_df: pd.DataFrame,
    principals: pd.DataFrame,
    pr: pd.DataFrame,
    role_pct: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """
    For each film in split_df, compute:
      director_pagerank   — mean within-director percentile of director(s)
      writer_pagerank     — mean within-writer percentile of writer(s)
      top_actor_pagerank  — within-actor percentile of ordering=1 actor/actress
      avg_cast_pagerank   — mean within-actor percentile of top-3 actors/actresses

    Values are percentile ranks (0–1), not raw PageRank scores.
    Fallback = 0.5 (median percentile) for crew absent from the training graph.
    """
    FALLBACK = 0.5  # median percentile — neutral for debut / unknown crew

    def _mean_pct(group: pd.DataFrame, cat: str) -> float:
        sub  = group[group["category"] == cat]
        pool = role_pct.get(cat, {})
        vals = [pool.get(n, FALLBACK) for n in sub["nconst"]]
        return float(np.mean(vals)) if vals else FALLBACK

    def _top_actor_pct(group: pd.DataFrame) -> float:
        actors = group[group["category"].isin({"actor", "actress"})].sort_values("ordering")
        if actors.empty:
            return FALLBACK
        pool = role_pct.get("actor", {})
        return float(pool.get(actors.iloc[0]["nconst"], FALLBACK))

    def _avg_cast_pct(group: pd.DataFrame, top_n: int = 3) -> float:
        actors = (
            group[group["category"].isin({"actor", "actress"})]
            .sort_values("ordering")
            .head(top_n)
        )
        if actors.empty:
            return FALLBACK
        pool = role_pct.get("actor", {})
        vals = [pool.get(n, FALLBACK) for n in actors["nconst"]]
        return float(np.mean(vals))

    rows = []
    prin_by_film = principals.groupby("tconst")

    for tconst in split_df["tconst"]:
        if tconst not in prin_by_film.groups:
            rows.append({
                "tconst":             tconst,
                "director_pagerank":  FALLBACK,
                "writer_pagerank":    FALLBACK,
                "top_actor_pagerank": FALLBACK,
                "avg_cast_pagerank":  FALLBACK,
            })
            continue
        grp = prin_by_film.get_group(tconst)
        rows.append({
            "tconst":             tconst,
            "director_pagerank":  _mean_pct(grp, "director"),
            "writer_pagerank":    _mean_pct(grp, "writer"),
            "top_actor_pagerank": _top_actor_pct(grp),
            "avg_cast_pagerank":  _avg_cast_pct(grp),
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Temporal feature assembly
# ══════════════════════════════════════════════════════════════════════════════

def _build_temporal_features(
    split_df: pd.DataFrame,
    principals: pd.DataFrame,
    train_df: pd.DataFrame,
    cutoffs: list[int],
    reset_prob: float,
    tol: float,
    max_iter: int,
    spark_cores: int,
    force_networkx: bool,
) -> pd.DataFrame:
    """
    Build PageRank features without temporal leakage.

    Why this fixes the original problem
    ------------------------------------
    The original graph used ALL training films to compute PageRank, then
    assigned those scores to every film regardless of release year. This means
    a 1992 film got a PageRank score that included collaborations from 1995,
    2000, and 2010 — future information. The score was therefore almost
    identical to the lifetime hit_rate, making it collinear and adding no
    independent signal.

    The fix: for films released in year Y, build the graph using only
    training films released BEFORE year Y. This produces a genuine leading
    indicator — "how central was this person in the network at the time this
    film was made?"

    Why this should fix the PSI shift
    ----------------------------------
    The original PSI shift occurred because val films have a different
    temporal distribution than train films. With temporal graphs, a 2012
    train film and a 2012 val film both get scores from the same pre-2010
    graph — their feature distributions are structurally aligned.

    Implementation: year-band graphs
    ---------------------------------
    We build one graph per cutoff year (O(n_cutoffs) PageRank computations).
    Films in year band [cutoff_i, cutoff_{i+1}) get scores from the graph
    built at cutoff_i. This is a practical approximation — the alternative
    (one graph per unique year) would require ~100 PageRank runs.

    Films released before the first cutoff, or with no year data, get the
    global fallback of 0.5 (median percentile — neutral, not penalised).
    """
    PR_COLS  = ["director_pagerank", "writer_pagerank",
                "top_actor_pagerank", "avg_cast_pagerank"]
    FALLBACK = 0.5
    cutoffs  = sorted(cutoffs)

    # Attach year to principals (for filtering edges by year)
    year_map   = train_df[["tconst", "startYear"]].copy()
    prin_yr    = principals.merge(year_map, on="tconst", how="left")
    prin_yr["startYear"] = prin_yr["startYear"].fillna(0).astype(int)

    # Result store: tconst → feature dict
    feat_store: dict[str, dict] = {}

    for band_idx, cutoff in enumerate(cutoffs):
        # Films in this band: startYear in [cutoff, next_cutoff)
        next_cutoff = cutoffs[band_idx + 1] if band_idx + 1 < len(cutoffs) else 9999
        band_mask = (
            (split_df["startYear"] >= cutoff) &
            (split_df["startYear"] <  next_cutoff)
        )
        band_tconsts = split_df.loc[band_mask, "tconst"].tolist()
        if not band_tconsts:
            continue

        # Training films released STRICTLY BEFORE this cutoff
        train_before = train_df[train_df["startYear"] < cutoff][["tconst", "label"]]
        if len(train_before) < 50:
            # Too few films to build a meaningful graph — use fallback
            print(f"[g1_pagerank] Cutoff {cutoff}: only {len(train_before)} train films — using fallback")
            for tc in band_tconsts:
                feat_store[tc] = {c: FALLBACK for c in PR_COLS}
            continue

        print(f"[g1_pagerank] Cutoff {cutoff} (band {cutoff}–{next_cutoff}): "
              f"{len(train_before)} train films, {len(band_tconsts)} scored films")

        # Build edges from pre-cutoff training films only
        edges_df = _build_edges(principals, train_before)
        vertices = principals["nconst"].unique().tolist()

        pr = _compute_pagerank(
            edges_df, vertices, reset_prob, tol, max_iter, spark_cores, force_networkx
        )
        role_pct = _compute_role_percentiles(pr, principals)

        # Score band films
        band_df  = pd.DataFrame({"tconst": band_tconsts})
        feat_df  = _build_features(band_df, principals, pr, role_pct)

        for _, row in feat_df.iterrows():
            feat_store[row["tconst"]] = {c: row[c] for c in PR_COLS}

    # Assemble final DataFrame — any film not scored gets fallback
    n_fallback = sum(1 for tc in split_df["tconst"] if tc not in feat_store)
    if n_fallback:
        print(f"[g1_pagerank] {n_fallback} films before first cutoff → fallback=0.5")

    rows = []
    for tconst in split_df["tconst"]:
        row = feat_store.get(tconst, {c: FALLBACK for c in PR_COLS})
        rows.append({"tconst": tconst, **row})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run(
    reset_prob: float = 0.15,
    tol: float = 0.01,
    max_iter: int = 20,
    spark_cores: int = 4,
    force_networkx: bool = False,
    temporal_cutoffs: list[int] | None = None,
) -> pd.DataFrame:
    print("=" * 60)
    print("Graph Pipeline — g1: Collaboration Network PageRank")
    print("=" * 60)

    _gr = _CFG.get("graph", {}).get("pagerank", {})
    if temporal_cutoffs is None:
        temporal_cutoffs = _gr.get(
            "temporal_cutoffs",
            [1970, 1980, 1990, 1995, 2000, 2005, 2010, 2015, 2020],
        )

    print(f"[g1_pagerank] Temporal cutoffs: {temporal_cutoffs}")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("[g1_pagerank] Loading splits and principals...")
    train_df, val_df, test_df = _load_splits()
    principals = _load_people()
    names      = _load_names()

    print(f"[g1_pagerank] Train: {len(train_df)} films | "
          f"Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"[g1_pagerank] Principals: {len(principals)} rows, "
          f"{principals['nconst'].nunique():,} unique people")

    # ── 2. Build temporal features for each split ─────────────────────────────
    # The temporal approach computes a separate PageRank graph per year-band,
    # using only training films released BEFORE each cutoff year.
    # This eliminates temporal leakage and should fix the PSI shift between
    # train and val splits.
    #
    # A global lifetime graph is also computed once and saved to
    # pagerank_scores.csv for visualisation/validation purposes (V1–V3).
    print("[g1_pagerank] Building global lifetime graph for visualisation...")
    edges_global = _build_edges(principals, train_df[["tconst", "label"]])
    vertices     = principals["nconst"].unique().tolist()
    print(f"[g1_pagerank] Global graph: {len(vertices):,} vertices, "
          f"{len(edges_global):,} directed edges")

    pr_global  = _compute_pagerank(
        edges_global, vertices, reset_prob, tol, max_iter, spark_cores, force_networkx
    )
    role_pct_global = _compute_role_percentiles(pr_global, principals)

    # Save global scores for g2_validate.py visualisations
    pr_named = pr_global.merge(names, on="nconst", how="left")
    pr_named = pr_named.merge(
        principals[["nconst", "category"]].drop_duplicates("nconst"),
        on="nconst", how="left",
    )
    pr_named["pagerank_pct"] = pr_named.apply(
        lambda row: role_pct_global.get(str(row["category"]), {}).get(row["nconst"], 0.5),
        axis=1,
    )
    pr_named = pr_named.sort_values("pagerank", ascending=False).reset_index(drop=True)
    pr_named.to_csv(OUT_GRAPH / "pagerank_scores.csv", index=False)
    print(f"[g1_pagerank] Saved pagerank_scores.csv ({len(pr_named)} rows)")

    print("\nTop 10 by PageRank (global lifetime graph):")
    print(
        pr_named[["primaryName", "category", "pagerank", "pagerank_pct"]]
        .head(10).to_string(index=False)
    )
    print()

    # ── 3. Build TEMPORAL features for each split ─────────────────────────────
    print("[g1_pagerank] Building temporal features (year-band graphs)...")
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n[g1_pagerank] ── {split_name.upper()} ──")
        feat_df = _build_temporal_features(
            split_df=split_df,
            principals=principals,
            train_df=train_df[["tconst", "label", "startYear"]],
            cutoffs=temporal_cutoffs,
            reset_prob=reset_prob,
            tol=tol,
            max_iter=max_iter,
            spark_cores=spark_cores,
            force_networkx=force_networkx,
        )
        out_name = f"features_pagerank{'_' + split_name if split_name != 'train' else ''}"
        feat_df.to_parquet(OUT_GRAPH / f"{out_name}.parquet", index=False)
        feat_df.to_csv(OUT_GRAPH / f"{out_name}.csv", index=False)
        print(f"[g1_pagerank] Saved {out_name}.parquet ({len(feat_df)} rows)")
        if split_name == "train":
            print("  Feature ranges (all should be 0–1):")
            for col in ["director_pagerank", "writer_pagerank",
                        "top_actor_pagerank", "avg_cast_pagerank"]:
                print(f"    {col}: [{feat_df[col].min():.3f}, {feat_df[col].max():.3f}]"
                      f"  mean={feat_df[col].mean():.3f}")

    print("\n" + "=" * 60)
    print("Graph Pipeline — Complete")
    print(f"Outputs in: {OUT_GRAPH}")
    print("=" * 60)

    return pr_named


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute PageRank on the IMDB collaboration graph."
    )
    parser.add_argument("--reset-prob",    type=float, default=0.15,
                        help="PageRank teleportation probability (default: 0.15)")
    parser.add_argument("--tol",           type=float, default=0.01,
                        help="Convergence tolerance (default: 0.01)")
    parser.add_argument("--max-iter",      type=int,   default=20,
                        help="Max PageRank iterations (default: 20)")
    parser.add_argument("--spark-cores",   type=int,   default=4,
                        help="Spark local cores (default: 4)")
    parser.add_argument("--networkx",      action="store_true",
                        help="Force NetworkX fallback instead of Spark")
    args = parser.parse_args()

    run(
        reset_prob=args.reset_prob,
        tol=args.tol,
        max_iter=args.max_iter,
        spark_cores=args.spark_cores,
        force_networkx=args.networkx,
    )
    print("[g1_pagerank] Done.")
