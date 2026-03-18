#!/usr/bin/env python3
"""
temporal_holdout_eval.py — Temporal Train/Holdout Evaluation + Final Submission
================================================================================

Performs a principled temporal split of the labeled IMDB data to get an honest
out-of-time AUC estimate, selects a Youden's-J optimal threshold, then retrains
on all labeled data and writes final submission files.

Design rationale
----------------
A random 80/20 internal split inflates AUC because directors and writers seen
in the internal validation set are also present in training (same-pool OOF
effect). Using a temporal cutoff (pre-2013 train, 2013+ holdout) gives a true
out-of-distribution estimate that better predicts leaderboard performance.

Youden's J threshold (argmax TPR − FPR) is calibrated on the temporal holdout
rather than on the training distribution. Because the training set has a higher
hit rate (~53%) than the hidden validation set (~41%), the optimal threshold is
well below 0.5.

Run from the project root:
    python pipeline/temporal_holdout_eval.py                         # standard
    python pipeline/temporal_holdout_eval.py --tune                  # + param search
    python pipeline/temporal_holdout_eval.py --tune --n-trials 40    # 40 trials
    python pipeline/temporal_holdout_eval.py --threshold 0.50        # force threshold
    python pipeline/temporal_holdout_eval.py --global-fallback       # ablation control

Flags
-----
  --tune              Run random hyperparameter search before final training.
                      Evaluates each candidate on the temporal holdout and prints
                      a live leaderboard. Best params are used for submission.
  --n-trials N        Number of random parameter combinations to try (default: 30).
  --threshold T       Override Youden's J with a fixed threshold T for the
                      submission file. Useful for ladder-testing different
                      operating points without re-running the full search.
                      Holdout AUC and Youden threshold are still printed.
  --global-fallback   Use the global-median PageRank fallback instead of the
                      era-stratified (decade) fallback. Use this as the control
                      condition when ablating the era-stratified improvement so
                      that only one variable changes per leaderboard submission.

Self-contained: does NOT import from the pipeline package.
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent   # project root

TRAIN_CLEAN   = ROOT / "data" / "processed" / "train_clean.parquet"
VAL_CLEAN     = ROOT / "data" / "processed" / "validation_hidden_clean.parquet"
TEST_CLEAN    = ROOT / "data" / "processed" / "test_hidden_clean.parquet"

TITLE_CREW    = ROOT / "data" / "raw" / "IMDB_external_csv" / "title_crew.csv"
NAME_BASICS   = ROOT / "data" / "raw" / "IMDB_external_csv" / "name_basics.csv"

PAGERANK_FILE = ROOT / "pipeline" / "outputs" / "graph" / "features_pagerank.parquet"

VAL_FEAT_FILE  = ROOT / "pipeline" / "outputs" / "features" / "features_val.parquet"
TEST_FEAT_FILE = ROOT / "pipeline" / "outputs" / "features" / "features_test.parquet"

VAL_RAW_CSV   = ROOT / "data" / "raw" / "csv" / "validation_hidden.csv"
TEST_RAW_CSV  = ROOT / "data" / "raw" / "csv" / "test_hidden.csv"

SUBMISSIONS   = ROOT / "submissions"
SUBMISSIONS.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
TEMPORAL_CUTOFF = 2013      # TRAIN: year <= cutoff, HOLDOUT: year > cutoff
SMOOTHING_C     = 5         # Bayesian smoothing count for hit-rate encoding
TFIDF_MAX_FEAT  = 3000
TFIDF_NGRAM     = (1, 2)
SEED            = 42

GENRE_LIST = [
    "action", "adventure", "animation", "biography", "comedy", "crime",
    "documentary", "drama", "family", "fantasy", "film_noir", "history",
    "horror", "music", "musical", "mystery", "romance", "sci_fi", "short",
    "sport", "thriller", "war", "western",
]

FEATURE_COLS = [
    # numerical base
    "startYear", "numVotes_log1p", "runtimeMinutes",
    # title lexical
    "title_len", "title_word_count", "title_has_digit", "title_has_colon",
    "has_original_title",
    # crew counts
    "num_directors", "num_writers", "is_auteur",
    # birth years (filled with median)
    "dir_avg_birth_year", "wri_avg_birth_year",
    # OOF hit-rate encodings
    "director_hit_rate", "writer_hit_rate",
    # title-similarity features
    "title_sim_to_hit", "title_sim_to_non_hit", "title_sim_margin",
    # pagerank
    "director_pagerank", "writer_pagerank", "top_actor_pagerank", "avg_cast_pagerank",
] + [f"genre_{g}" for g in GENRE_LIST]


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_crew_index(path: Path) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Return (dir_idx, wri_idx) where each is {tconst: [nconst, ...]}."""
    _BAD = {"", r"\N", "\\N"}
    dir_idx: Dict[str, List[str]] = defaultdict(list)
    wri_idx: Dict[str, List[str]] = defaultdict(list)
    print("  Loading title_crew.csv …")
    crew = pd.read_csv(path, usecols=["tconst", "directors", "writers"], dtype=str)
    for _, row in crew.iterrows():
        tc = str(row["tconst"]).strip()
        for raw, idx in [(row["directors"], dir_idx), (row["writers"], wri_idx)]:
            if pd.isna(raw) or str(raw).strip() in _BAD:
                continue
            for tok in str(raw).split(","):
                tok = tok.strip()
                if tok not in _BAD:
                    idx[tc].append(tok)
    # deduplicate
    dir_idx = {k: sorted(set(v)) for k, v in dir_idx.items()}
    wri_idx = {k: sorted(set(v)) for k, v in wri_idx.items()}
    return dir_idx, wri_idx


def load_birth_years(path: Path) -> Dict[str, float]:
    """Return {nconst: birthYear (float)} from name_basics."""
    print("  Loading name_basics.csv …")
    nb = pd.read_csv(path, usecols=["nconst", "birthYear"], dtype=str)
    bmap: Dict[str, float] = {}
    for _, row in nb.iterrows():
        nc = str(row["nconst"]).strip()
        by = row["birthYear"]
        if pd.notna(by) and str(by).strip() not in ("", r"\N", "\\N"):
            try:
                bmap[nc] = float(by)
            except ValueError:
                pass
    return bmap


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# ── 2a. Director/writer LOO smoothed hit-rate ──────────────────────────────────

def compute_loo_hit_rate(
    df: pd.DataFrame,
    entity_idx: Dict[str, List[str]],
    global_mean: float,
    C: float = SMOOTHING_C,
) -> np.ndarray:
    """
    Leave-one-out (LOO) smoothed hit-rate encoding for a training set.

    For each film i, the feature is the mean over its entities e of:
        rate_e_LOO(i) = (sum_labels_other_films_with_e + C * gm) / (count_other + C)

    Returns an array aligned to df.index.
    """
    tconsts = df["tconst"].astype(str).tolist()
    labels  = df["label"].astype(float).values

    # Pre-build entity → {tconst_idx: label} map
    ent_tconst_list: Dict[str, List[int]] = defaultdict(list)
    for i, tc in enumerate(tconsts):
        for eid in entity_idx.get(tc, []):
            ent_tconst_list[eid].append(i)

    # For each entity: total sum and count across ALL train films
    ent_sum:   Dict[str, float] = {}
    ent_count: Dict[str, int]   = {}
    for eid, idxs in ent_tconst_list.items():
        ent_sum[eid]   = float(sum(labels[j] for j in idxs))
        ent_count[eid] = len(idxs)

    result = np.full(len(df), global_mean)
    for i, tc in enumerate(tconsts):
        entities = entity_idx.get(tc, [])
        if not entities:
            result[i] = global_mean
            continue
        rates = []
        for eid in entities:
            total_sum   = ent_sum.get(eid, 0.0)
            total_count = ent_count.get(eid, 0)
            # Subtract this film's contribution
            own_label  = labels[i]
            other_sum  = total_sum   - own_label
            other_cnt  = total_count - 1
            rate = (other_sum + C * global_mean) / (other_cnt + C)
            rates.append(rate)
        result[i] = float(np.mean(rates))
    return result


def build_entity_lookup(
    df: pd.DataFrame,
    entity_idx: Dict[str, List[str]],
    global_mean: float,
    C: float = SMOOTHING_C,
) -> Dict[str, float]:
    """
    Build a lookup {entity_id: smoothed_hit_rate} fitted on ALL of df.
    Used for applying to holdout / hidden sets.
    """
    tconsts = df["tconst"].astype(str).tolist()
    labels  = df["label"].astype(float).values

    ent_sum:   Dict[str, float] = defaultdict(float)
    ent_count: Dict[str, int]   = defaultdict(int)
    for i, tc in enumerate(tconsts):
        for eid in entity_idx.get(tc, []):
            ent_sum[eid]   += labels[i]
            ent_count[eid] += 1

    lookup: Dict[str, float] = {}
    for eid in ent_sum:
        lookup[eid] = (ent_sum[eid] + C * global_mean) / (ent_count[eid] + C)
    return lookup


def apply_entity_lookup(
    df: pd.DataFrame,
    entity_idx: Dict[str, List[str]],
    lookup: Dict[str, float],
    global_mean: float,
) -> np.ndarray:
    """Apply pre-fitted lookup to a (hold-out or hidden) DataFrame."""
    result = np.full(len(df), global_mean)
    for i, tc in enumerate(df["tconst"].astype(str)):
        entities = entity_idx.get(tc, [])
        if not entities:
            result[i] = global_mean
            continue
        rates = [lookup.get(eid, global_mean) for eid in entities]
        result[i] = float(np.mean(rates))
    return result


# ── 2b. Title similarity (TF-IDF cosine to class centroids) ────────────────────

def fit_title_sim(train_df: pd.DataFrame):
    """
    Fit TF-IDF on train titles and compute hit/non-hit centroids.
    Returns (vectorizer, hit_centroid, nonhit_centroid).
    """
    vec = TfidfVectorizer(max_features=TFIDF_MAX_FEAT, ngram_range=TFIDF_NGRAM)
    X = vec.fit_transform(train_df["primaryTitle"].fillna("").astype(str))
    labels = train_df["label"].astype(float).values

    hit_mask    = labels == 1
    nonhit_mask = labels == 0

    hit_centroid    = np.asarray(X[hit_mask].mean(axis=0))       # (1, n_feat)
    nonhit_centroid = np.asarray(X[nonhit_mask].mean(axis=0))
    return vec, hit_centroid, nonhit_centroid


def compute_title_sim_features(titles: pd.Series, vec, hit_centroid, nonhit_centroid) -> pd.DataFrame:
    """Compute title_sim_to_hit, title_sim_to_non_hit, title_sim_margin."""
    X = vec.transform(titles.fillna("").astype(str))
    sim_hit    = cosine_similarity(X, hit_centroid).ravel()
    sim_nonhit = cosine_similarity(X, nonhit_centroid).ravel()
    return pd.DataFrame({
        "title_sim_to_hit":     sim_hit,
        "title_sim_to_non_hit": sim_nonhit,
        "title_sim_margin":     sim_hit - sim_nonhit,
    })


# ── 2c. Genre binary features ─────────────────────────────────────────────────

def compute_genre_features(genres_series: pd.Series) -> pd.DataFrame:
    """Parse space-separated genre strings into binary columns."""
    result = {}
    gs = genres_series.fillna("").astype(str)
    for g in GENRE_LIST:
        if g == "sci_fi":
            pattern = "Sci-Fi|Sci Fi|sci.fi"
        elif g == "film_noir":
            pattern = "Film.Noir"
        else:
            pattern = g
        result[f"genre_{g}"] = gs.str.contains(pattern, case=False, regex=True).astype(float)
    return pd.DataFrame(result, index=genres_series.index)


# ── 2d. Crew count features from entity_idx ────────────────────────────────────

def compute_crew_counts(
    df: pd.DataFrame,
    dir_idx: Dict[str, List[str]],
    wri_idx: Dict[str, List[str]],
) -> pd.DataFrame:
    """Compute num_directors, num_writers, is_auteur."""
    num_dirs = []
    num_wris = []
    is_auteur = []
    for tc in df["tconst"].astype(str):
        dirs = set(dir_idx.get(tc, []))
        wris = set(wri_idx.get(tc, []))
        num_dirs.append(float(len(dirs)))
        num_wris.append(float(len(wris)))
        overlap = dirs & wris
        is_auteur.append(1.0 if (len(dirs) == 1 and len(wris) == 1 and len(overlap) == 1) else 0.0)
    return pd.DataFrame({
        "num_directors": num_dirs,
        "num_writers":   num_wris,
        "is_auteur":     is_auteur,
    }, index=df.index)


# ── 2e. Avg birth year features ────────────────────────────────────────────────

def compute_birth_year_features(
    df: pd.DataFrame,
    dir_idx: Dict[str, List[str]],
    wri_idx: Dict[str, List[str]],
    birth_map: Dict[str, float],
    dir_median: float,
    wri_median: float,
) -> pd.DataFrame:
    """Compute dir_avg_birth_year and wri_avg_birth_year with imputation."""
    dir_by = []
    wri_by = []
    for tc in df["tconst"].astype(str):
        dirs = dir_idx.get(tc, [])
        wris = wri_idx.get(tc, [])
        d_years = [birth_map[d] for d in dirs if d in birth_map]
        w_years = [birth_map[w] for w in wris if w in birth_map]
        dir_by.append(float(np.mean(d_years)) if d_years else dir_median)
        wri_by.append(float(np.mean(w_years)) if w_years else wri_median)
    return pd.DataFrame({
        "dir_avg_birth_year": dir_by,
        "wri_avg_birth_year": wri_by,
    }, index=df.index)


# ── 2f. PageRank merge (era-stratified fallback) ───────────────────────────────

PR_COLS = ["director_pagerank", "writer_pagerank", "top_actor_pagerank", "avg_cast_pagerank"]


def build_global_pr_medians(
    train_df: pd.DataFrame,
    pr_df: pd.DataFrame,
) -> Dict[str, Dict]:
    """
    Build global-only PageRank medians (no era stratification).
    Returns the same dict structure as build_era_pr_medians so merge_pagerank
    works identically — useful as the control condition in ablation studies.
    """
    merged = train_df[["tconst"]].merge(pr_df[["tconst"] + PR_COLS], on="tconst", how="left")
    result: Dict[str, Dict] = {}
    for col in PR_COLS:
        global_med = float(merged[col].median())
        result[col] = {"global": global_med}   # no decade keys → always falls back to global
    return result


def build_era_pr_medians(
    train_df: pd.DataFrame,
    pr_df: pd.DataFrame,
) -> Dict[str, Dict[int, float]]:
    """
    Build era-stratified (decade) PageRank medians from the training set.

    For a film with no crew record the global median is a poor fallback: a
    1940s film and a 2015 film have very different collaboration-network
    densities. Stratifying by decade gives a much better prior.

    Returns {pr_col: {decade: median_value, 'global': global_median}}.
    """
    merged = train_df[["tconst", "startYear"]].merge(
        pr_df[["tconst"] + PR_COLS], on="tconst", how="left"
    )
    era_medians: Dict[str, Dict[int, float]] = {}
    for col in PR_COLS:
        col_meds: Dict[int, float] = {}
        global_med = float(merged[col].median())
        merged["_decade"] = (merged["startYear"] // 10 * 10).astype("Int64")
        for decade, grp in merged.groupby("_decade"):
            vals = grp[col].dropna()
            col_meds[int(decade)] = float(vals.median()) if len(vals) >= 5 else global_med
        col_meds["global"] = global_med          # type: ignore[assignment]
        era_medians[col] = col_meds
    return era_medians


def merge_pagerank(
    df: pd.DataFrame,
    pr_df: pd.DataFrame,
    era_pr_medians: Dict[str, Dict],
) -> pd.DataFrame:
    """
    Left-merge pagerank features with era-stratified fallback.

    Films with no crew record receive the median PageRank of other films
    from the same decade rather than the global median, correcting for the
    fact that collaboration-network density grows over time.
    """
    merged = df.merge(pr_df[["tconst"] + PR_COLS], on="tconst", how="left")
    decades = (pd.to_numeric(merged["startYear"], errors="coerce") // 10 * 10).astype("Int64")
    for col in PR_COLS:
        col_meds = era_pr_medians[col]
        global_med = col_meds["global"]
        fallback = decades.map(lambda d: col_meds.get(int(d), global_med)  # type: ignore[arg-type]
                               if pd.notna(d) else global_med)
        merged[col] = merged[col].fillna(fallback)
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# 3. FULL FEATURE MATRIX BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(
    df: pd.DataFrame,
    dir_hit_rates: np.ndarray,
    wri_hit_rates: np.ndarray,
    title_sim_df: pd.DataFrame,
    genre_df: pd.DataFrame,
    crew_df: pd.DataFrame,
    birth_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble all feature columns into one DataFrame."""
    base = df[["tconst"]].copy()

    # Numerical base
    base["startYear"]      = pd.to_numeric(df["startYear"], errors="coerce")
    base["numVotes_log1p"] = pd.to_numeric(df["numVotes_log1p"], errors="coerce").fillna(0.0)
    base["runtimeMinutes"] = pd.to_numeric(df["runtimeMinutes"], errors="coerce")

    # Title lexical
    titles = df["primaryTitle"].fillna("").astype(str)
    orig   = df["originalTitle"].fillna("").astype(str) if "originalTitle" in df.columns else pd.Series("", index=df.index)
    base["title_len"]        = titles.str.len().astype(float)
    base["title_word_count"] = titles.str.split().str.len().fillna(0).astype(float)
    base["title_has_digit"]  = titles.str.contains(r"\d", regex=True).astype(float)
    base["title_has_colon"]  = titles.str.contains(":", regex=False).astype(float)
    base["has_original_title"] = (orig.str.strip() != titles.str.strip()).astype(float)

    # OOF hit rates
    base["director_hit_rate"] = dir_hit_rates
    base["writer_hit_rate"]   = wri_hit_rates

    # Concatenate everything
    result = pd.concat(
        [base.reset_index(drop=True),
         title_sim_df.reset_index(drop=True),
         genre_df.reset_index(drop=True),
         crew_df.reset_index(drop=True),
         birth_df.reset_index(drop=True)],
        axis=1,
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 4. MAIN SCRIPT
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Temporal holdout evaluation + final submission generator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--tune", action="store_true", default=False,
        help="Run random hyperparameter search before final training.",
    )
    p.add_argument(
        "--n-trials", dest="n_trials", type=int, default=30,
        help="Number of random parameter combinations to try (default: 30).",
    )
    p.add_argument(
        "--threshold", dest="threshold", type=float, default=None, metavar="T",
        help=(
            "Override Youden's J threshold with a fixed value T for the final submission. "
            "Useful for ladder-testing without re-running hyperparameter search. "
            "The holdout AUC and Youden threshold are still printed for reference."
        ),
    )
    p.add_argument(
        "--global-fallback", dest="global_fallback", action="store_true", default=False,
        help=(
            "Use global-median PageRank fallback instead of era-stratified (decade) fallback. "
            "Use this flag to isolate the effect of the era-stratified improvement."
        ),
    )
    return p.parse_args()


# ── Hyperparameter search grid ─────────────────────────────────────────────────

_PARAM_GRID = {
    "n_estimators":     [200, 300, 500, 700, 1000],
    "max_depth":        [3, 4, 5, 6, 7],
    "learning_rate":    [0.01, 0.03, 0.05, 0.08, 0.10],
    "subsample":        [0.70, 0.75, 0.80, 0.85, 0.90],
    "colsample_bytree": [0.60, 0.70, 0.80, 0.85, 0.90],
    "min_child_weight": [1, 2, 3, 5],
    "gamma":            [0.0, 0.1, 0.2, 0.5],
    "reg_alpha":        [0.0, 0.01, 0.05, 0.10],
    "reg_lambda":       [0.5, 1.0, 1.5, 2.0],
}


def _random_params(seed: int) -> dict:
    rng = random.Random(seed)
    return {k: rng.choice(v) for k, v in _PARAM_GRID.items()}


def run_hyperparameter_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_hld: np.ndarray,
    y_hld: np.ndarray,
    n_trials: int,
    base_seed: int = SEED,
) -> dict:
    """
    Random search over XGBoost hyperparameters using the temporal holdout as
    evaluation set. Prints a live leaderboard after each trial.

    Returns the best parameter dict.
    """
    print(f"\n{'─'*70}")
    print(f"  HYPERPARAMETER SEARCH  ({n_trials} random trials)")
    print(f"  Evaluation metric: holdout ROC-AUC")
    print(f"{'─'*70}")
    print(f"  {'Trial':>5}  {'AUC':>7}  {'n_est':>5}  {'depth':>5}  "
          f"{'lr':>6}  {'sub':>5}  {'col':>5}  {'mcw':>4}  {'gamma':>6}")
    print(f"  {'─'*5}  {'─'*7}  {'─'*5}  {'─'*5}  "
          f"{'─'*6}  {'─'*5}  {'─'*5}  {'─'*4}  {'─'*6}")

    best_auc    = -1.0
    best_params = {}
    results     = []

    for trial in range(1, n_trials + 1):
        params = _random_params(base_seed + trial * 17)
        model = XGBClassifier(
            **params,
            random_state=base_seed,
            eval_metric="logloss",
            verbosity=0,
        )
        model.fit(X_train, y_train, verbose=False)
        auc = roc_auc_score(y_hld, model.predict_proba(X_hld)[:, 1])
        results.append((auc, params))

        marker = " ◀ best" if auc > best_auc else ""
        if auc > best_auc:
            best_auc    = auc
            best_params = params

        print(f"  {trial:>5}  {auc:>7.4f}  "
              f"{params['n_estimators']:>5}  "
              f"{params['max_depth']:>5}  "
              f"{params['learning_rate']:>6.3f}  "
              f"{params['subsample']:>5.2f}  "
              f"{params['colsample_bytree']:>5.2f}  "
              f"{params['min_child_weight']:>4}  "
              f"{params['gamma']:>6.2f}"
              f"{marker}")

    print(f"\n  {'─'*70}")
    print(f"  Best AUC: {best_auc:.4f}")
    print(f"  Best params: {best_params}")
    print(f"  {'─'*70}\n")
    return best_params


def main():
    args = _parse_args()
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print("=" * 70)
    print("  TEMPORAL HOLDOUT EVALUATION + FINAL SUBMISSION GENERATOR")
    print(f"  Run timestamp: {ts}")
    mode_parts = []
    if args.tune:
        mode_parts.append(f"hyperparameter search ({args.n_trials} trials)")
    else:
        mode_parts.append("standard params")
    mode_parts.append("global-median fallback" if args.global_fallback else "era-stratified fallback")
    print(f"  Mode: {' + '.join(mode_parts)}")
    print("=" * 70)

    # ── Load external data ──────────────────────────────────────────────────
    print("\n[1] Loading external data …")
    dir_idx, wri_idx = load_crew_index(TITLE_CREW)
    birth_map = load_birth_years(NAME_BASICS)
    print(f"  crew index: {len(dir_idx)} tconsts with directors, "
          f"{len(wri_idx)} tconsts with writers")
    print(f"  birth_map: {len(birth_map)} people")

    print("  Loading pagerank features …")
    pr_df = pd.read_parquet(PAGERANK_FILE)

    # ── Load labeled data ───────────────────────────────────────────────────
    print("\n[2] Loading labeled data …")
    labeled = pd.read_parquet(TRAIN_CLEAN)
    print(f"  Labeled data: {len(labeled)} rows, label mean = {labeled['label'].mean():.4f}")

    # ── Temporal split ──────────────────────────────────────────────────────
    print(f"\n[3] Temporal split (cutoff year: {TEMPORAL_CUTOFF}) …")
    train_df   = labeled[labeled["startYear"] <= TEMPORAL_CUTOFF].copy().reset_index(drop=True)
    holdout_df = labeled[labeled["startYear"] >  TEMPORAL_CUTOFF].copy().reset_index(drop=True)
    print(f"  TRAIN   : {len(train_df)} rows, "
          f"hit rate = {train_df['label'].mean():.4f}")
    print(f"  HOLDOUT : {len(holdout_df)} rows, "
          f"hit rate = {holdout_df['label'].mean():.4f}")

    # ── Compute training-set statistics ────────────────────────────────────
    print("\n[4] Computing training-set statistics …")

    # Birth year medians (imputation values for unknown people)
    train_dir_years = []
    train_wri_years = []
    for tc in train_df["tconst"].astype(str):
        d_years = [birth_map[d] for d in dir_idx.get(tc, []) if d in birth_map]
        w_years = [birth_map[w] for w in wri_idx.get(tc, []) if w in birth_map]
        if d_years:
            train_dir_years.append(np.mean(d_years))
        if w_years:
            train_wri_years.append(np.mean(w_years))
    dir_birth_median = float(np.median(train_dir_years)) if train_dir_years else 1970.0
    wri_birth_median = float(np.median(train_wri_years)) if train_wri_years else 1970.0
    print(f"  dir birth year median (train): {dir_birth_median:.0f}")
    print(f"  wri birth year median (train): {wri_birth_median:.0f}")

    # PageRank medians on train
    if args.global_fallback:
        era_pr_medians = build_global_pr_medians(train_df, pr_df)
        print(f"  pagerank fallback: GLOBAL median  (--global-fallback)")
    else:
        era_pr_medians = build_era_pr_medians(train_df, pr_df)
        print(f"  pagerank fallback: ERA-STRATIFIED (decade buckets)")
    global_pr_str = {col: f"{era_pr_medians[col]['global']:.4f}" for col in PR_COLS}
    print(f"  pagerank global medians (train): {global_pr_str}")

    # Runtime median for filling NaN
    runtime_median_train = float(train_df["runtimeMinutes"].median())

    # ── TRAIN features ──────────────────────────────────────────────────────
    print("\n[5] Engineering TRAIN features (LOO encoding, TF-IDF fitted on train) …")

    global_mean_dir = float(train_df["label"].mean())
    global_mean_wri = global_mean_dir  # same base rate

    dir_loo = compute_loo_hit_rate(train_df, dir_idx, global_mean_dir)
    wri_loo = compute_loo_hit_rate(train_df, wri_idx, global_mean_wri)
    print(f"  director LOO: mean={dir_loo.mean():.4f}, std={dir_loo.std():.4f}")
    print(f"  writer  LOO: mean={wri_loo.mean():.4f}, std={wri_loo.std():.4f}")

    # TF-IDF fit on train
    tfidf_vec, hit_centroid, nonhit_centroid = fit_title_sim(train_df)
    print(f"  TF-IDF vocab size: {len(tfidf_vec.vocabulary_)}")

    train_title_sim  = compute_title_sim_features(train_df["primaryTitle"], tfidf_vec, hit_centroid, nonhit_centroid)
    train_genre      = compute_genre_features(train_df["genres"])
    train_crew       = compute_crew_counts(train_df, dir_idx, wri_idx)
    train_birth      = compute_birth_year_features(train_df, dir_idx, wri_idx, birth_map,
                                                    dir_birth_median, wri_birth_median)

    train_feat = build_feature_matrix(
        train_df, dir_loo, wri_loo, train_title_sim, train_genre, train_crew, train_birth
    )
    # Fill runtime NaN with training median
    train_feat["runtimeMinutes"] = train_feat["runtimeMinutes"].fillna(runtime_median_train)
    train_feat = merge_pagerank(train_feat, pr_df, era_pr_medians)
    # Ensure all feature cols exist
    for col in FEATURE_COLS:
        if col not in train_feat.columns:
            train_feat[col] = 0.0

    # ── HOLDOUT features ────────────────────────────────────────────────────
    print("\n[6] Engineering HOLDOUT features (applying train statistics) …")

    # Build lookups from FULL train set
    dir_lookup = build_entity_lookup(train_df, dir_idx, global_mean_dir)
    wri_lookup = build_entity_lookup(train_df, wri_idx, global_mean_wri)

    dir_hld = apply_entity_lookup(holdout_df, dir_idx, dir_lookup, global_mean_dir)
    wri_hld = apply_entity_lookup(holdout_df, wri_idx, wri_lookup, global_mean_wri)

    hld_title_sim = compute_title_sim_features(holdout_df["primaryTitle"], tfidf_vec, hit_centroid, nonhit_centroid)
    hld_genre     = compute_genre_features(holdout_df["genres"])
    hld_crew      = compute_crew_counts(holdout_df, dir_idx, wri_idx)
    hld_birth     = compute_birth_year_features(holdout_df, dir_idx, wri_idx, birth_map,
                                                 dir_birth_median, wri_birth_median)

    hld_feat = build_feature_matrix(
        holdout_df, dir_hld, wri_hld, hld_title_sim, hld_genre, hld_crew, hld_birth
    )
    hld_feat["runtimeMinutes"] = hld_feat["runtimeMinutes"].fillna(runtime_median_train)
    hld_feat = merge_pagerank(hld_feat, pr_df, era_pr_medians)
    for col in FEATURE_COLS:
        if col not in hld_feat.columns:
            hld_feat[col] = 0.0

    # ── Model training ──────────────────────────────────────────────────────
    print("\n[7] Training XGBClassifier on temporal TRAIN set …")
    X_train = train_feat[FEATURE_COLS].astype(float).values
    y_train = train_df["label"].astype(int).values
    X_hld   = hld_feat[FEATURE_COLS].astype(float).values
    y_hld   = holdout_df["label"].astype(int).values

    # Handle any remaining NaN/Inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_hld   = np.nan_to_num(X_hld,   nan=0.0, posinf=0.0, neginf=0.0)

    model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=SEED,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train, verbose=False)
    print("  Model training complete.")

    # ── Evaluation on holdout ───────────────────────────────────────────────
    print("\n[8] Evaluating on HOLDOUT …")
    proba_hld = model.predict_proba(X_hld)[:, 1]
    auc = roc_auc_score(y_hld, proba_hld)
    print(f"  HOLDOUT AUC: {auc:.4f}")

    # Youden's J threshold
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_hld, proba_hld)
    youden_j   = tpr - fpr
    best_idx   = int(np.argmax(youden_j))
    best_thresh = float(thresholds[best_idx])
    print(f"  Youden's J optimal threshold: {best_thresh:.4f}")
    print(f"    TPR={tpr[best_idx]:.4f}, FPR={fpr[best_idx]:.4f}, J={youden_j[best_idx]:.4f}")

    def evaluate_at_threshold(proba, y_true, thresh, label=""):
        """Print accuracy/F1/precision/recall for a given probability threshold."""
        preds = (proba >= thresh).astype(int)
        n_true  = int(preds.sum())
        n_false = int((preds == 0).sum())
        acc  = accuracy_score(y_true, preds)
        f1   = f1_score(y_true, preds, zero_division=0)
        prec = precision_score(y_true, preds, zero_division=0)
        rec  = recall_score(y_true, preds, zero_division=0)
        hit_rate = float(preds.sum() / len(preds))
        print(f"\n  {label} (threshold={thresh:.4f}):")
        print(f"    Predicted True={n_true}, False={n_false}  "
              f"(predicted hit rate={hit_rate:.4f}, actual hit rate={y_true.mean():.4f})")
        print(f"    Accuracy={acc:.4f}, F1={f1:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")

    evaluate_at_threshold(proba_hld, y_hld, best_thresh, "Youden threshold")
    evaluate_at_threshold(proba_hld, y_hld, 0.5,         "Threshold=0.5")

    # --threshold override: user can force a specific threshold for submission
    submission_thresh = best_thresh
    if args.threshold is not None:
        submission_thresh = args.threshold
        print(f"\n  ⚠  --threshold override: using {submission_thresh:.4f} for submission "
              f"(Youden was {best_thresh:.4f})")
        evaluate_at_threshold(proba_hld, y_hld, submission_thresh, f"Forced threshold")

    print(f"\n  --- SUMMARY ---")
    print(f"  Train hit rate      : {y_train.mean():.4f}")
    print(f"  Holdout hit rate    : {y_hld.mean():.4f}")
    print(f"  Holdout AUC         : {auc:.4f}")
    print(f"  Youden threshold    : {best_thresh:.4f}")
    print(f"  Submission threshold: {submission_thresh:.4f}")

    # ── Optional hyperparameter search ──────────────────────────────────────
    best_xgb_params: dict = dict(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        min_child_weight=1, gamma=0.0,
        reg_alpha=0.0, reg_lambda=1.0,
    )

    if args.tune:
        best_xgb_params = run_hyperparameter_search(
            X_train, y_train, X_hld, y_hld,
            n_trials=args.n_trials,
        )
        # Re-evaluate best params on holdout and recompute Youden threshold
        best_model = XGBClassifier(
            **best_xgb_params,
            random_state=SEED,
            eval_metric="logloss",
            verbosity=0,
        )
        best_model.fit(X_train, y_train, verbose=False)
        proba_hld_tuned = best_model.predict_proba(X_hld)[:, 1]
        auc_tuned = roc_auc_score(y_hld, proba_hld_tuned)
        from sklearn.metrics import roc_curve as _roc
        _fpr, _tpr, _ths = _roc(y_hld, proba_hld_tuned)
        _j_idx = int(np.argmax(_tpr - _fpr))
        tuned_youden = float(_ths[_j_idx])
        submission_thresh = tuned_youden if args.threshold is None else args.threshold
        print(f"\n  Tuned model holdout AUC  : {auc_tuned:.4f}  (default was {auc:.4f})")
        print(f"  Tuned Youden threshold   : {tuned_youden:.4f}")
        evaluate_at_threshold(proba_hld_tuned, y_hld, submission_thresh, "Tuned Youden threshold")
        best_thresh = tuned_youden   # keep for summary print

    # ── Retrain on ALL labeled data ─────────────────────────────────────────
    print("\n[9] Retraining on ALL labeled data …")
    print(f"  XGBoost params: {best_xgb_params}")
    all_labeled = labeled.copy().reset_index(drop=True)

    # OOF director/writer hit rates on full labeled set via 5-fold CV
    print("  Computing 5-fold OOF hit rates on all labeled data …")
    all_labels = all_labeled["label"].astype(float).values
    all_gm     = float(all_labels.mean())

    all_dir_oof = np.full(len(all_labeled), all_gm)
    all_wri_oof = np.full(len(all_labeled), all_gm)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold_idx, (fold_tr, fold_val) in enumerate(
            skf.split(np.zeros(len(all_labeled)), all_labeled["label"].astype(int))):
        fold_train_df = all_labeled.iloc[fold_tr].copy()
        fold_val_df   = all_labeled.iloc[fold_val].copy()
        fold_gm_d     = float(fold_train_df["label"].mean())
        fold_gm_w     = fold_gm_d
        d_lkp = build_entity_lookup(fold_train_df, dir_idx, fold_gm_d)
        w_lkp = build_entity_lookup(fold_train_df, wri_idx, fold_gm_w)
        all_dir_oof[fold_val] = apply_entity_lookup(fold_val_df, dir_idx, d_lkp, fold_gm_d)
        all_wri_oof[fold_val] = apply_entity_lookup(fold_val_df, wri_idx, w_lkp, fold_gm_w)
        print(f"    Fold {fold_idx + 1}/5 done.")

    # Build full-data dir/wri lookups for hidden inference
    dir_lookup_full = build_entity_lookup(all_labeled, dir_idx, all_gm)
    wri_lookup_full = build_entity_lookup(all_labeled, wri_idx, all_gm)

    # TF-IDF fit on all labeled data
    tfidf_full, hit_cent_full, nonhit_cent_full = fit_title_sim(all_labeled)
    print(f"  Full TF-IDF vocab: {len(tfidf_full.vocabulary_)}")

    # Birth year medians on all labeled
    all_dir_years, all_wri_years = [], []
    for tc in all_labeled["tconst"].astype(str):
        dy = [birth_map[d] for d in dir_idx.get(tc, []) if d in birth_map]
        wy = [birth_map[w] for w in wri_idx.get(tc, []) if w in birth_map]
        if dy: all_dir_years.append(np.mean(dy))
        if wy: all_wri_years.append(np.mean(wy))
    dir_birth_med_full = float(np.median(all_dir_years)) if all_dir_years else dir_birth_median
    wri_birth_med_full = float(np.median(all_wri_years)) if all_wri_years else wri_birth_median

    runtime_median_full = float(all_labeled["runtimeMinutes"].median())

    # PageRank medians on full labeled set (same fallback strategy as training)
    if args.global_fallback:
        era_pr_medians_full = build_global_pr_medians(all_labeled, pr_df)
    else:
        era_pr_medians_full = build_era_pr_medians(all_labeled, pr_df)

    # Build full labeled feature matrix
    all_title_sim = compute_title_sim_features(
        all_labeled["primaryTitle"], tfidf_full, hit_cent_full, nonhit_cent_full)
    all_genre     = compute_genre_features(all_labeled["genres"])
    all_crew      = compute_crew_counts(all_labeled, dir_idx, wri_idx)
    all_birth     = compute_birth_year_features(
        all_labeled, dir_idx, wri_idx, birth_map,
        dir_birth_med_full, wri_birth_med_full)

    all_feat = build_feature_matrix(
        all_labeled, all_dir_oof, all_wri_oof,
        all_title_sim, all_genre, all_crew, all_birth)
    all_feat["runtimeMinutes"] = all_feat["runtimeMinutes"].fillna(runtime_median_full)
    all_feat = merge_pagerank(all_feat, pr_df, era_pr_medians_full)
    for col in FEATURE_COLS:
        if col not in all_feat.columns:
            all_feat[col] = 0.0

    X_all = all_feat[FEATURE_COLS].astype(float).values
    y_all = all_labeled["label"].astype(int).values
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    model_full = XGBClassifier(
        **best_xgb_params,
        random_state=SEED,
        eval_metric="logloss",
        verbosity=0,
    )
    model_full.fit(X_all, y_all, verbose=False)
    print("  Full model training complete.")

    # ── Build hidden val and test feature matrices ──────────────────────────
    print("\n[10] Building features for hidden validation and test sets …")

    def build_hidden_features(clean_path: Path, label_name: str) -> pd.DataFrame:
        """
        Build the full feature matrix for a hidden (unlabeled) split.

        Uses statistics fitted on all labeled data (lookups, TF-IDF, birth medians,
        PageRank medians) so that no information from the hidden set leaks into
        the encodings.  Returns a DataFrame with columns '_proba' and '_tconst'
        appended for downstream alignment and submission writing.
        """
        print(f"  Processing {label_name} …")
        hid_df = pd.read_parquet(clean_path).reset_index(drop=True)

        dir_rates = apply_entity_lookup(hid_df, dir_idx, dir_lookup_full, all_gm)
        wri_rates = apply_entity_lookup(hid_df, wri_idx, wri_lookup_full, all_gm)
        title_sim = compute_title_sim_features(
            hid_df["primaryTitle"], tfidf_full, hit_cent_full, nonhit_cent_full)
        genre_f   = compute_genre_features(hid_df["genres"])
        crew_f    = compute_crew_counts(hid_df, dir_idx, wri_idx)
        birth_f   = compute_birth_year_features(
            hid_df, dir_idx, wri_idx, birth_map,
            dir_birth_med_full, wri_birth_med_full)

        feat = build_feature_matrix(
            hid_df, dir_rates, wri_rates, title_sim, genre_f, crew_f, birth_f)
        feat["runtimeMinutes"] = feat["runtimeMinutes"].fillna(runtime_median_full)
        feat = merge_pagerank(feat, pr_df, era_pr_medians_full)
        for col in FEATURE_COLS:
            if col not in feat.columns:
                feat[col] = 0.0

        X_hid = feat[FEATURE_COLS].astype(float).values
        X_hid = np.nan_to_num(X_hid, nan=0.0, posinf=0.0, neginf=0.0)
        proba_hid = model_full.predict_proba(X_hid)[:, 1]
        feat["_proba"] = proba_hid
        feat["_tconst"] = hid_df["tconst"].values
        return feat

    val_feat_full  = build_hidden_features(VAL_CLEAN,  "hidden validation")
    test_feat_full = build_hidden_features(TEST_CLEAN, "hidden test")

    # ── Align to raw CSV order and write submissions ─────────────────────────
    print("\n[11] Writing submission files …")

    def write_submission(feat_df: pd.DataFrame, raw_csv: Path,
                         out_path: Path, label_name: str):
        """
        Write a submission file aligned to the raw hidden CSV row order.

        Merges model probabilities onto the raw tconst order (not the cleaned
        parquet order), then writes one 'True'/'False' per line using
        submission_thresh.  tconsts missing from the feature matrix fall back
        to the global labeled-set mean probability.
        """
        raw = pd.read_csv(raw_csv, usecols=["tconst"])
        prob_map = dict(zip(feat_df["_tconst"].astype(str),
                            feat_df["_proba"].values))
        n_missing = 0
        lines = []
        for tc in raw["tconst"].astype(str):
            p = prob_map.get(tc, None)
            if p is None:
                # Fallback to global mean
                p = all_gm
                n_missing += 1
            lines.append("True" if p >= submission_thresh else "False")
        if n_missing > 0:
            print(f"    WARNING: {n_missing} tconsts in raw CSV not found in "
                  f"hidden features — using global mean fallback.")
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        n_true = lines.count("True")
        print(f"  {label_name}: {len(lines)} predictions written → "
              f"True={n_true} ({n_true / len(lines):.4f})")
        print(f"  Saved to: {out_path}")

    val_out  = SUBMISSIONS / f"validation_final_holdout_{ts}.txt"
    test_out = SUBMISSIONS / f"test_final_holdout_{ts}.txt"

    write_submission(val_feat_full,  VAL_RAW_CSV,  val_out,  "Validation")
    write_submission(test_feat_full, TEST_RAW_CSV, test_out, "Test")

    print("\n" + "=" * 70)
    print("  DONE.")
    print(f"  Holdout AUC       : {auc:.4f}")
    print(f"  Youden threshold  : {best_thresh:.4f}")
    print(f"  Validation output : {val_out}")
    print(f"  Test output       : {test_out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
