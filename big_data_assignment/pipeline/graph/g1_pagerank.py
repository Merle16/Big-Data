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
    """Return (train_df, val_df, test_df) with tconst + label columns."""
    train = pd.read_parquet(PROC / "train_clean.parquet")[["tconst", "label"]]
    val   = pd.read_parquet(PROC / "validation_hidden_clean.parquet")[["tconst"]]
    test  = pd.read_parquet(PROC / "test_hidden_clean.parquet")[["tconst"]]
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
    results = g.pageRank(resetProbability=reset_prob, tol=tol, maxIter=max_iter)

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

def _build_features(
    split_df: pd.DataFrame,
    principals: pd.DataFrame,
    pr: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each film in split_df, compute:
      director_pagerank   — mean PageRank of directors
      writer_pagerank     — mean PageRank of writers
      top_actor_pagerank  — PageRank of ordering=1 actor/actress
      avg_cast_pagerank   — mean PageRank of top-3 actors/actresses
    """
    global_median = float(pr["pagerank"].median())

    pr_lookup = pr.set_index("nconst")["pagerank"].to_dict()

    def _mean_pr(group: pd.DataFrame, cat: str | list) -> float:
        cats = [cat] if isinstance(cat, str) else cat
        sub  = group[group["category"].isin(cats)]
        vals = [pr_lookup.get(n, global_median) for n in sub["nconst"]]
        return float(np.mean(vals)) if vals else global_median

    def _top_actor_pr(group: pd.DataFrame) -> float:
        actors = group[group["category"].isin({"actor", "actress"})].sort_values("ordering")
        if actors.empty:
            return global_median
        return float(pr_lookup.get(actors.iloc[0]["nconst"], global_median))

    def _avg_cast_pr(group: pd.DataFrame, top_n: int = 3) -> float:
        actors = (
            group[group["category"].isin({"actor", "actress"})]
            .sort_values("ordering")
            .head(top_n)
        )
        if actors.empty:
            return global_median
        vals = [pr_lookup.get(n, global_median) for n in actors["nconst"]]
        return float(np.mean(vals))

    rows = []
    prin_by_film = principals.groupby("tconst")

    for tconst in split_df["tconst"]:
        if tconst not in prin_by_film.groups:
            rows.append({
                "tconst":             tconst,
                "director_pagerank":  global_median,
                "writer_pagerank":    global_median,
                "top_actor_pagerank": global_median,
                "avg_cast_pagerank":  global_median,
            })
            continue
        grp = prin_by_film.get_group(tconst)
        rows.append({
            "tconst":             tconst,
            "director_pagerank":  _mean_pr(grp, "director"),
            "writer_pagerank":    _mean_pr(grp, "writer"),
            "top_actor_pagerank": _top_actor_pr(grp),
            "avg_cast_pagerank":  _avg_cast_pr(grp),
        })

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
) -> pd.DataFrame:
    print("=" * 60)
    print("Graph Pipeline — g1: Collaboration Network PageRank")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("[g1_pagerank] Loading splits and principals...")
    train_df, val_df, test_df = _load_splits()
    principals = _load_people()
    names      = _load_names()

    print(f"[g1_pagerank] Train: {len(train_df)} films | "
          f"Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"[g1_pagerank] Principals: {len(principals)} rows, "
          f"{principals['nconst'].nunique():,} unique people")

    # ── 2. Build edges (train labels only) ───────────────────────────────────
    print("[g1_pagerank] Building collaboration edges...")
    edges_df = _build_edges(principals, train_df)
    vertices = principals["nconst"].unique().tolist()
    print(f"[g1_pagerank] Graph: {len(vertices):,} vertices, {len(edges_df):,} directed edges")

    # ── 3. PageRank ───────────────────────────────────────────────────────────
    pr = _compute_pagerank(
        edges_df, vertices, reset_prob, tol, max_iter, spark_cores, force_networkx
    )

    # ── 4. Attach names and save scores ──────────────────────────────────────
    pr_named = pr.merge(names, on="nconst", how="left")
    pr_named = pr_named.merge(
        principals[["nconst", "category"]].drop_duplicates("nconst"),
        on="nconst", how="left",
    )
    pr_named = pr_named.sort_values("pagerank", ascending=False).reset_index(drop=True)
    pr_named.to_csv(OUT_GRAPH / "pagerank_scores.csv", index=False)
    print(f"[g1_pagerank] Saved pagerank_scores.csv ({len(pr_named)} rows)")

    # Top 10 for quick sanity check
    print("\nTop 10 by PageRank:")
    print(pr_named[["primaryName", "category", "pagerank"]].head(10).to_string(index=False))
    print()

    # ── 5. Build features for each split ─────────────────────────────────────
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"[g1_pagerank] Building features for {split_name}...")
        feat_df = _build_features(split_df, principals, pr)
        out_name = f"features_pagerank{'_' + split_name if split_name != 'train' else ''}"
        feat_df.to_parquet(OUT_GRAPH / f"{out_name}.parquet", index=False)
        feat_df.to_csv(OUT_GRAPH / f"{out_name}.csv", index=False)
        print(f"[g1_pagerank] Saved {out_name}.parquet ({len(feat_df)} rows)")

    print("=" * 60)
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
