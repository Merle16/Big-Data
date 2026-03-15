#!/usr/bin/env python3
"""
Genre enrichment — external Movies_by_Genre catalog.

Reads
-----
  data/Movies_by_Genre/*.csv          — external genre catalog (16 genre files)
  pipeline/outputs/cleaning/train_clean.parquet   — cleaned movie table

Writes (all to pipeline/outputs/enrichment/)
---------------------------------------------
  CSVs
    movies_with_genres.csv
    genre_file_summary.csv
    genre_conflict_summary.csv
    genre_join_strategy_audit.csv
    genre_field_alignment.csv
    genre_recovery_summary.csv
    genre_fill_candidates.csv
    genre_token_centers.csv
    genre_top_tokens.csv
    genre_top_tokens_all.csv
    genre_schema_profile.csv

  Figures
    01_join_strategy.png
    02_top_genres.png
    03_recovery.png

State keys produced
-------------------
  movies_with_genres, genre_file_summary, genre_conflict_summary,
  genre_join_strategy_audit, genre_field_alignment, genre_recovery_summary,
  genre_fill_candidates, genre_top_tokens, genre_top_tokens_all,
  genre_centers, genre_schema_profile
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("KMP_ENABLE_SHM", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── rcParams ──────────────────────────────────────────────────────────────────
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

_ROOT      = Path(__file__).resolve().parents[1]
_CLEAN_DIR = _ROOT / "pipeline" / "outputs" / "cleaning"
_GENRE_DIR = _ROOT / "data" / "Movies_by_Genre"
_OUT_DIR   = _ROOT / "pipeline" / "outputs" / "enrichment"

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

SOURCE_TAG_MAP = {
    "action": "action",
    "adventure": "adventure",
    "animation": "animation",
    "biography": "biography",
    "crime": "crime",
    "family": "family",
    "fantasy": "fantasy",
    "film-noir": "film noir",
    "history": "history",
    "horror": "horror",
    "mystery": "mystery",
    "romance": "romance",
    "scifi": "science fiction",
    "sports": "sport",
    "thriller": "thriller",
    "war": "war",
}

FIELD_SPECS = [
    {
        "field": "startYear",
        "indicator": "startYear_was_missing",
        "direct": "genre_year_direct",
        "center": "genre_center_year",
        "tolerance": 1.0,
        "tolerance_label": "within_1_year_pct",
    },
    {
        "field": "runtimeMinutes",
        "indicator": "runtimeMinutes_was_missing",
        "direct": "genre_runtime_direct",
        "center": "genre_center_runtime",
        "tolerance": 5.0,
        "tolerance_label": "within_5_min_pct",
    },
    {
        "field": "numVotes",
        "indicator": "numVotes_was_missing",
        "direct": "genre_votes_direct",
        "center": "genre_center_votes",
        "pct_tolerance": 0.10,
        "tolerance_label": "within_10pct_pct",
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm_title(title) -> str:
    if pd.isna(title):
        return ""
    text = str(title).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _split_genres(value) -> List[str]:
    if pd.isna(value):
        return []
    tokens = []
    for part in str(value).split(","):
        tok = part.strip().lower()
        tok = tok.replace("sci-fi", "science fiction")
        tok = tok.replace("film-noir", "film noir")
        tok = re.sub(r"\s+", " ", tok)
        if tok:
            tokens.append(tok)
    return sorted(set(tokens))


def _schema_profile(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        non_null = int(df[col].notna().sum())
        rows.append({
            "stage": label,
            "column": col,
            "dtype": str(df[col].dtype),
            "non_null": non_null,
            "missing_pct": round((1 - non_null / len(df)) * 100, 2) if len(df) else 0.0,
            "unique_non_null": int(df[col].dropna().nunique()),
        })
    return pd.DataFrame(rows)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_genre_catalog(genre_dir: Path) -> pd.DataFrame:
    if not genre_dir.exists():
        raise FileNotFoundError(f"Genre folder not found: {genre_dir}")

    frames: List[pd.DataFrame] = []
    for csv_path in sorted(genre_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        df["genre_source_file"] = csv_path.stem
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    clean = raw.rename(columns={
        "movie_id":    "tconst",
        "movie_name":  "genre_movie_title",
        "genre":       "genre_text",
        "certificate": "genre_certificate",
        "rating":      "genre_rating",
        "description": "genre_description",
        "director":    "genre_director",
        "director_id": "genre_director_id",
        "star":        "genre_star",
        "star_id":     "genre_star_id",
        "votes":       "genre_votes",
        "gross(in $)": "genre_gross_usd",
    }).copy()

    clean["tconst"] = clean["tconst"].astype("string").str.strip()
    clean["genre_source_tag"] = (
        clean["genre_source_file"].map(SOURCE_TAG_MAP).fillna(clean["genre_source_file"])
    )
    clean["genre_title_key"]      = clean["genre_movie_title"].map(_norm_title)
    clean["genre_year"]           = pd.to_numeric(clean["year"], errors="coerce")
    clean["genre_runtime_minutes"] = pd.to_numeric(
        clean["runtime"].astype("string").str.extract(r"(\d+)")[0], errors="coerce"
    )
    clean["genre_rating"]    = pd.to_numeric(clean["genre_rating"], errors="coerce")
    clean["genre_votes"]     = pd.to_numeric(clean["genre_votes"], errors="coerce")
    clean["genre_gross_usd"] = pd.to_numeric(
        clean["genre_gross_usd"].astype("string").str.replace(r"[$,]", "", regex=True),
        errors="coerce",
    )
    return clean


# ── Build steps ───────────────────────────────────────────────────────────────

def build_file_summary(catalog: pd.DataFrame) -> pd.DataFrame:
    return (
        catalog.groupby("genre_source_file", dropna=False)
        .agg(
            rows=("tconst", "size"),
            unique_tconst=("tconst", "nunique"),
            missing_year_pct=("genre_year", lambda s: round(s.isna().mean() * 100, 2)),
            missing_runtime_pct=("genre_runtime_minutes", lambda s: round(s.isna().mean() * 100, 2)),
            missing_rating_pct=("genre_rating", lambda s: round(s.isna().mean() * 100, 2)),
            missing_votes_pct=("genre_votes", lambda s: round(s.isna().mean() * 100, 2)),
            missing_gross_pct=("genre_gross_usd", lambda s: round(s.isna().mean() * 100, 2)),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
        .reset_index(drop=True)
    )


def build_token_long(catalog: pd.DataFrame) -> pd.DataFrame:
    parsed = catalog[["tconst", "genre_text"]].copy()
    parsed["genre_token"] = parsed["genre_text"].apply(_split_genres)
    parsed = parsed.explode("genre_token").drop(columns=["genre_text"])

    source_tags = catalog[["tconst", "genre_source_tag"]].rename(
        columns={"genre_source_tag": "genre_token"}
    )
    tokens = pd.concat([parsed, source_tags], ignore_index=True)
    tokens["genre_token"] = (
        tokens["genre_token"].astype("string").str.strip().str.lower()
    )
    tokens["genre_token"] = tokens["genre_token"].replace(
        {"<na>": pd.NA, "": pd.NA, "nan": pd.NA}
    )
    return tokens.dropna(subset=["genre_token"]).drop_duplicates()


def aggregate_genre_catalog(catalog: pd.DataFrame, tokens: pd.DataFrame) -> pd.DataFrame:
    numeric = (
        catalog.groupby("tconst", dropna=False)[
            ["genre_year", "genre_runtime_minutes", "genre_votes", "genre_rating", "genre_gross_usd"]
        ]
        .median()
        .rename(columns={
            "genre_year":             "genre_year_direct",
            "genre_runtime_minutes":  "genre_runtime_direct",
            "genre_votes":            "genre_votes_direct",
            "genre_rating":           "genre_rating_direct",
            "genre_gross_usd":        "genre_gross_direct",
        })
        .reset_index()
    )

    grp = catalog.groupby("tconst", dropna=False)
    counts = pd.DataFrame({
        "tconst":                    numeric["tconst"],
        "genre_external_rows":       grp.size().values,
        "genre_source_file_count":   grp["genre_source_file"].nunique().values,
        "genre_title_variant_count": grp["genre_title_key"].nunique(dropna=True).values,
        "genre_year_variant_count":  grp["genre_year"].nunique(dropna=True).values,
        "genre_runtime_variant_count": grp["genre_runtime_minutes"].nunique(dropna=True).values,
        "genre_votes_variant_count": grp["genre_votes"].nunique(dropna=True).values,
        "genre_rating_variant_count": grp["genre_rating"].nunique(dropna=True).values,
    })

    title_first = (
        catalog.sort_values(["tconst", "genre_movie_title", "genre_title_key"], kind="mergesort")
        .drop_duplicates("tconst")[["tconst", "genre_movie_title", "genre_title_key"]]
        .rename(columns={
            "genre_movie_title": "genre_title_external",
            "genre_title_key":   "genre_title_key_external",
        })
    )

    token_summary = (
        tokens.sort_values(["tconst", "genre_token"], kind="mergesort")
        .groupby("tconst", dropna=False)
        .agg(
            genre_token_count=("genre_token", "size"),
            genre_labels=("genre_token", "|".join),
        )
        .reset_index()
    )

    out = (
        numeric
        .merge(counts, on="tconst", how="left")
        .merge(title_first, on="tconst", how="left")
        .merge(token_summary, on="tconst", how="left")
    )
    for col in [
        "genre_title_variant_count", "genre_year_variant_count",
        "genre_runtime_variant_count", "genre_votes_variant_count",
        "genre_rating_variant_count", "genre_token_count",
    ]:
        out[col] = out[col].fillna(0).astype(int)
    return out


def compute_genre_centers(catalog: pd.DataFrame, tokens: pd.DataFrame) -> pd.DataFrame:
    movie_numeric = (
        catalog.groupby("tconst", dropna=False)[
            ["genre_year", "genre_runtime_minutes", "genre_votes", "genre_rating", "genre_gross_usd"]
        ]
        .median()
        .reset_index()
    )
    token_frame = tokens.merge(movie_numeric, on="tconst", how="left").drop_duplicates()
    token_frame["genre_votes_log1p"] = np.log1p(token_frame["genre_votes"].clip(lower=0))
    token_frame["genre_gross_log1p"] = np.log1p(token_frame["genre_gross_usd"].clip(lower=0))

    centers = (
        token_frame.groupby("genre_token", dropna=False)
        .agg(
            matched_movies=("tconst", "nunique"),
            median_year=("genre_year", "median"),
            median_runtime_minutes=("genre_runtime_minutes", "median"),
            median_rating=("genre_rating", "median"),
            median_votes_log1p=("genre_votes_log1p", "median"),
            median_gross_log1p=("genre_gross_log1p", "median"),
            votes_iqr=("genre_votes_log1p", lambda s: s.quantile(0.75) - s.quantile(0.25)),
            runtime_iqr=("genre_runtime_minutes", lambda s: s.quantile(0.75) - s.quantile(0.25)),
        )
        .reset_index()
    )
    centers["median_votes"]     = np.expm1(centers["median_votes_log1p"])
    centers["median_gross_usd"] = np.expm1(centers["median_gross_log1p"])
    return centers.sort_values(
        ["matched_movies", "genre_token"], ascending=[False, True]
    ).reset_index(drop=True)


def assess_join_strategies(movies: pd.DataFrame, genre_agg: pd.DataFrame) -> pd.DataFrame:
    direct_ids = set(genre_agg["tconst"].dropna().astype(str))
    unmatched  = movies.loc[~movies["tconst"].astype(str).isin(direct_ids)].copy()
    unmatched["title_key"]       = unmatched["primaryTitle"].map(_norm_title)
    unmatched["startYear_round"] = pd.to_numeric(unmatched["startYear"], errors="coerce").round().astype("Int64")
    unmatched["runtime_round"]   = pd.to_numeric(unmatched["runtimeMinutes"], errors="coerce").round().astype("Int64")

    key_table = genre_agg[["tconst", "genre_title_key_external", "genre_year_direct", "genre_runtime_direct"]].copy()
    key_table["genre_year_round"]    = pd.to_numeric(key_table["genre_year_direct"], errors="coerce").round().astype("Int64")
    key_table["genre_runtime_round"] = pd.to_numeric(key_table["genre_runtime_direct"], errors="coerce").round().astype("Int64")

    title_year = unmatched.merge(
        key_table[["tconst", "genre_title_key_external", "genre_year_round"]].drop_duplicates(),
        left_on=["title_key", "startYear_round"],
        right_on=["genre_title_key_external", "genre_year_round"],
        how="inner",
    )
    title_year_runtime = unmatched.merge(
        key_table[["tconst", "genre_title_key_external", "genre_year_round", "genre_runtime_round"]].drop_duplicates(),
        left_on=["title_key", "startYear_round", "runtime_round"],
        right_on=["genre_title_key_external", "genre_year_round", "genre_runtime_round"],
        how="inner",
    )

    def _ambiguity(frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0
        return int((frame.groupby("tconst_x")["tconst_y"].nunique() > 1).sum())

    total = movies["tconst"].nunique()
    return pd.DataFrame([
        {
            "strategy": "direct_tconst",
            "matched_movies": int(movies["tconst"].astype(str).isin(direct_ids).sum()),
            "coverage_pct": round(movies["tconst"].astype(str).isin(direct_ids).mean() * 100, 2),
            "ambiguous_movies": 0,
            "decision": "use",
            "reason": "External files already provide IMDb title IDs; this is the clean, auditable key.",
        },
        {
            "strategy": "title_year_fallback",
            "matched_movies": int(title_year["tconst_x"].nunique()) if not title_year.empty else 0,
            "coverage_pct": round((title_year["tconst_x"].nunique() / total) * 100, 2) if total else 0.0,
            "ambiguous_movies": _ambiguity(title_year),
            "decision": "reject",
            "reason": "Adds negligible coverage and introduces title collisions.",
        },
        {
            "strategy": "title_year_runtime_fallback",
            "matched_movies": int(title_year_runtime["tconst_x"].nunique()) if not title_year_runtime.empty else 0,
            "coverage_pct": round((title_year_runtime["tconst_x"].nunique() / total) * 100, 2) if total else 0.0,
            "ambiguous_movies": _ambiguity(title_year_runtime),
            "decision": "reject",
            "reason": "Too strict to add meaningful safe coverage after ID matching.",
        },
    ])


def attach_genre_centers(movies_with_genres: pd.DataFrame, centers: pd.DataFrame) -> pd.DataFrame:
    center_lookup = centers.set_index("genre_token").to_dict(orient="index")

    def _per_row_center(label_text, key):
        if pd.isna(label_text):
            return np.nan
        vals = [
            float(center_lookup[tok.strip().lower()][key])
            for tok in str(label_text).split("|")
            if tok.strip().lower() in center_lookup
            and pd.notna(center_lookup[tok.strip().lower()].get(key))
        ]
        return float(np.median(vals)) if vals else np.nan

    out = movies_with_genres.copy()
    out["genre_center_year"]      = out["genre_labels"].apply(lambda s: _per_row_center(s, "median_year"))
    out["genre_center_runtime"]   = out["genre_labels"].apply(lambda s: _per_row_center(s, "median_runtime_minutes"))
    out["genre_center_rating"]    = out["genre_labels"].apply(lambda s: _per_row_center(s, "median_rating"))
    out["genre_center_votes"]     = out["genre_labels"].apply(lambda s: _per_row_center(s, "median_votes"))
    out["genre_center_gross_usd"] = out["genre_labels"].apply(lambda s: _per_row_center(s, "median_gross_usd"))
    return out


def build_movies_with_genres(movies: pd.DataFrame, genre_agg: pd.DataFrame, centers: pd.DataFrame) -> pd.DataFrame:
    out = movies.merge(genre_agg, on="tconst", how="left")
    out["genre_match_flag"] = out["genre_external_rows"].notna().astype(int)
    out["genre_title_key"]  = out["primaryTitle"].map(_norm_title)

    out["genre_title_exact"] = (
        (out["genre_title_key"] != "")
        & out["genre_title_key"].eq(out["genre_title_key_external"].fillna(""))
    ).astype(int)
    out["genre_year_abs_diff"]    = (pd.to_numeric(out["startYear"], errors="coerce") - out["genre_year_direct"]).abs()
    out["genre_runtime_abs_diff"] = (pd.to_numeric(out["runtimeMinutes"], errors="coerce") - out["genre_runtime_direct"]).abs()

    out = attach_genre_centers(out, centers)

    out["genre_fill_startYear"]      = out["genre_year_direct"].where(out["genre_year_direct"].notna(), out["genre_center_year"])
    out["genre_fill_runtimeMinutes"] = out["genre_runtime_direct"].where(out["genre_runtime_direct"].notna(), out["genre_center_runtime"])
    out["genre_fill_numVotes"]       = out["genre_votes_direct"].where(out["genre_votes_direct"].notna(), out["genre_center_votes"])

    out["genre_fill_startYear_source"] = np.select(
        [out["genre_year_direct"].notna(), out["genre_center_year"].notna()],
        ["direct_tconst", "genre_center"], default="none",
    )
    out["genre_fill_runtime_source"] = np.select(
        [out["genre_runtime_direct"].notna(), out["genre_center_runtime"].notna()],
        ["direct_tconst", "genre_center"], default="none",
    )
    out["genre_fill_numVotes_source"] = np.select(
        [out["genre_votes_direct"].notna(), out["genre_center_votes"].notna()],
        ["direct_tconst", "genre_center"], default="none",
    )
    return out


def build_field_alignment(movies_with_genres: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in FIELD_SPECS:
        field    = spec["field"]
        indicator = spec["indicator"]
        direct   = spec["direct"]
        sub = movies_with_genres[
            (movies_with_genres[indicator] == 0)
            & movies_with_genres[direct].notna()
            & movies_with_genres[field].notna()
        ].copy()
        if sub.empty:
            rows.append({
                "field": field, "comparison_rows": 0,
                "exact_match_pct": np.nan,
                spec["tolerance_label"]: np.nan,
                "median_abs_diff": np.nan, "correlation": np.nan,
                "note": "No comparable matched rows.",
            })
            continue

        diff = (
            pd.to_numeric(sub[field], errors="coerce")
            - pd.to_numeric(sub[direct], errors="coerce")
        ).abs()
        row = {
            "field": field,
            "comparison_rows": int(len(sub)),
            "exact_match_pct": round((diff == 0).mean() * 100, 2),
            "median_abs_diff": round(float(diff.median()), 3),
            "correlation": round(float(sub[[field, direct]].corr().iloc[0, 1]), 4),
        }
        if "pct_tolerance" in spec:
            rel = diff / sub[[field, direct]].max(axis=1).replace(0, np.nan)
            row[spec["tolerance_label"]] = round((rel <= spec["pct_tolerance"]).mean() * 100, 2)
            row["note"] = "High rank agreement but snapshot drift is visible; direct fills need caution."
        else:
            row[spec["tolerance_label"]] = round((diff <= spec["tolerance"]).mean() * 100, 2)
            row["note"] = "Direct external values align closely enough to act as credible fill candidates."
        rows.append(row)
    return pd.DataFrame(rows)


def build_recovery_summary(movies_with_genres: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in FIELD_SPECS:
        indicator = spec["indicator"]
        direct    = spec["direct"]
        center    = spec["center"]
        sub = movies_with_genres[movies_with_genres[indicator] == 1].copy()
        total       = int(len(sub))
        direct_hits = int(sub[direct].notna().sum())
        center_hits = int(sub[center].notna().sum())
        any_hits    = int((sub[direct].notna() | sub[center].notna()).sum())
        rows.append({
            "field": spec["field"],
            "missing_rows": total,
            "direct_recovered_rows": direct_hits,
            "direct_recovered_pct": round((direct_hits / total) * 100, 2) if total else 0.0,
            "center_recovered_rows": center_hits,
            "center_recovered_pct": round((center_hits / total) * 100, 2) if total else 0.0,
            "combined_recovered_rows": any_hits,
            "combined_recovered_pct": round((any_hits / total) * 100, 2) if total else 0.0,
            "recommended_use": (
                "direct_tconst" if spec["field"] in {"startYear", "runtimeMinutes"}
                else "direct_tconst_with_snapshot_caution"
            ),
        })
    return pd.DataFrame(rows)


def build_fill_candidates(movies_with_genres: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "tconst", "primaryTitle", "split", "genre_labels", "genre_match_flag",
        "genre_source_file_count",
        "startYear_was_missing", "genre_year_direct", "genre_center_year",
        "genre_fill_startYear", "genre_fill_startYear_source",
        "runtimeMinutes_was_missing", "genre_runtime_direct", "genre_center_runtime",
        "genre_fill_runtimeMinutes", "genre_fill_runtime_source",
        "numVotes_was_missing", "genre_votes_direct", "genre_center_votes",
        "genre_fill_numVotes", "genre_fill_numVotes_source",
        "genre_rating_direct", "genre_gross_direct",
    ]
    keep = movies_with_genres[
        (movies_with_genres["startYear_was_missing"] == 1)
        | (movies_with_genres["runtimeMinutes_was_missing"] == 1)
        | (movies_with_genres["numVotes_was_missing"] == 1)
    ].copy()
    keep = keep[
        keep["genre_fill_startYear"].notna()
        | keep["genre_fill_runtimeMinutes"].notna()
        | keep["genre_fill_numVotes"].notna()
    ]
    return keep[cols].sort_values(["split", "tconst"]).reset_index(drop=True)


def build_top_genres(tokens: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    matched_ids = set(movies["tconst"].astype(str))
    return (
        tokens[tokens["tconst"].astype(str).isin(matched_ids)]
        .groupby("genre_token", dropna=False)["tconst"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index(name="matched_movies")
    )


def build_top_tokens(tokens: pd.DataFrame) -> pd.DataFrame:
    return (
        tokens.groupby("genre_token", dropna=False)["tconst"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index(name="catalog_movies")
    )


def build_conflict_summary(catalog: pd.DataFrame, genre_agg: pd.DataFrame, tokens: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {"metric": "raw_rows",           "value": int(len(catalog)),                                   "detail": "Rows across all genre CSV files."},
        {"metric": "unique_tconst",      "value": int(catalog["tconst"].nunique()),                    "detail": "Unique IMDb title IDs in the external catalog."},
        {"metric": "multi_source_movies","value": int((genre_agg["genre_source_file_count"] > 1).sum()),"detail": "Movies appearing in more than one genre source file."},
        {"metric": "title_conflict_movies","value": int((genre_agg["genre_title_variant_count"] > 1).sum()),"detail": "Same IMDb ID with multiple normalized titles."},
        {"metric": "year_conflict_movies","value": int((genre_agg["genre_year_variant_count"] > 1).sum()),  "detail": "Same IMDb ID with more than one external year value."},
        {"metric": "runtime_conflict_movies","value": int((genre_agg["genre_runtime_variant_count"] > 1).sum()),"detail": "Same IMDb ID with more than one runtime value."},
        {"metric": "votes_conflict_movies",  "value": int((genre_agg["genre_votes_variant_count"] > 1).sum()),  "detail": "Same IMDb ID with more than one vote snapshot."},
        {"metric": "rating_conflict_movies", "value": int((genre_agg["genre_rating_variant_count"] > 1).sum()), "detail": "Same IMDb ID with more than one rating snapshot."},
        {"metric": "multi_token_movies", "value": int(tokens.groupby("tconst")["genre_token"].nunique().gt(1).sum()), "detail": "Movies with more than one genre token after normalization."},
    ])


def build_subgroup_coverage(movies_with_genres: pd.DataFrame) -> pd.DataFrame:
    """Coverage of genre join broken down by decade and titleType subgroups.

    Overall coverage can look healthy while specific cohorts fail badly.
    """
    rows = []

    # By decade
    years = pd.to_numeric(movies_with_genres["startYear"], errors="coerce")
    decade = (years // 10 * 10).astype("Int64")
    for dec, grp in movies_with_genres.groupby(decade, observed=True):
        if pd.isna(dec):
            continue
        total   = len(grp)
        matched = int(grp["genre_match_flag"].sum())
        rows.append({
            "group_type": "decade",
            "group_value": str(int(dec)) + "s",
            "total": total,
            "matched": matched,
            "coverage_pct": round(matched / total * 100, 2) if total else 0.0,
        })

    # By titleType
    if "titleType" in movies_with_genres.columns:
        for tt, grp in movies_with_genres.groupby("titleType", observed=True):
            if pd.isna(tt):
                continue
            total   = len(grp)
            matched = int(grp["genre_match_flag"].sum())
            rows.append({
                "group_type": "titleType",
                "group_value": str(tt),
                "total": total,
                "matched": matched,
                "coverage_pct": round(matched / total * 100, 2) if total else 0.0,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["status"] = df["coverage_pct"].apply(
            lambda p: "ok" if p >= 80 else ("warn" if p >= 50 else "low")
        )
    return df


def build_join_precision(movies_with_genres: pd.DataFrame) -> pd.DataFrame:
    """Estimate precision of the direct-tconst join using field consistency as a proxy.

    Since we have no external ground truth, precision is estimated by checking whether
    matched external values are consistent with the cleaned pipeline values:
      - year consistent: |startYear - genre_year_direct| ≤ 2 (where both non-null)
      - runtime consistent: |runtimeMinutes - genre_runtime_direct| ≤ 15 (where both non-null)

    A match is "suspicious" when BOTH fields are present and BOTH diverge.
    """
    matched = movies_with_genres[movies_with_genres["genre_match_flag"] == 1].copy()
    if matched.empty:
        return pd.DataFrame()

    sy    = pd.to_numeric(matched["startYear"],         errors="coerce")
    gy    = pd.to_numeric(matched["genre_year_direct"], errors="coerce")
    rt    = pd.to_numeric(matched["runtimeMinutes"],         errors="coerce")
    grt   = pd.to_numeric(matched["genre_runtime_direct"], errors="coerce")

    both_year    = sy.notna() & gy.notna()
    both_runtime = rt.notna() & grt.notna()

    year_ok    = (sy - gy).abs() <= 2
    runtime_ok = (rt - grt).abs() <= 15

    suspicious = (both_year & ~year_ok & both_runtime & ~runtime_ok).astype(int)

    total   = len(matched)
    rows = [{
        "strategy":                 "direct_tconst",
        "matched_movies":           total,
        "year_comparable_rows":     int(both_year.sum()),
        "year_consistent_pct":      round(year_ok[both_year].mean() * 100, 2) if both_year.any() else np.nan,
        "runtime_comparable_rows":  int(both_runtime.sum()),
        "runtime_consistent_pct":   round(runtime_ok[both_runtime].mean() * 100, 2) if both_runtime.any() else np.nan,
        "suspicious_matches":       int(suspicious.sum()),
        "suspicious_pct":           round(suspicious.mean() * 100, 2),
        "estimated_precision_floor": round((1 - suspicious.mean()) * 100, 2),
        "note": (
            "Low suspicious rate → join is trustworthy."
            if suspicious.mean() < 0.05
            else "Elevated suspicious rate → review conflicting year/runtime matches."
        ),
    }]
    return pd.DataFrame(rows)


def build_consistency_checks(movies_with_genres: pd.DataFrame) -> pd.DataFrame:
    """Downstream consistency checks after enrichment.

    Flags impossible or implausible combinations between enriched genre labels
    and cleaned pipeline fields.
    """
    rows = []

    mwg = movies_with_genres.copy()
    gl  = mwg["genre_labels"].fillna("").astype(str).str.lower()

    # 1. isAdult=1 and animation genre
    if "isAdult" in mwg.columns:
        adult = pd.to_numeric(mwg["isAdult"], errors="coerce").eq(1)
        flag  = adult & gl.str.contains(r"\banimation\b", regex=True)
        rows.append({
            "check": "isAdult=1 AND animation_genre",
            "n_flagged": int(flag.sum()),
            "total": len(mwg),
            "flag_pct": round(flag.mean() * 100, 3),
            "severity": "warn" if flag.sum() > 0 else "ok",
            "note": "Adult-rated animation; may be legitimate but worth reviewing.",
        })

    # 2. Very short runtime but not tagged short/documentary
    if "runtimeMinutes" in mwg.columns:
        rt = pd.to_numeric(mwg["runtimeMinutes"], errors="coerce")
        short_rt = rt < 20
        not_short_type = ~gl.str.contains(r"\b(short|animation|documentary)\b", regex=True)
        flag2 = short_rt & mwg["genre_match_flag"].eq(1) & not_short_type
        rows.append({
            "check": "runtime<20min AND no short/animation/documentary genre",
            "n_flagged": int(flag2.sum()),
            "total": len(mwg),
            "flag_pct": round(flag2.mean() * 100, 3),
            "severity": "warn" if flag2.sum() > 5 else "ok",
            "note": "Short runtime without short/animation/doc genre tag may be miscategorised.",
        })

    # 3. Genre label overlap with IMDB genres field
    if "genres" in mwg.columns:
        imdb_genre_vals = mwg["genres"].fillna("").astype(str).str.lower()
        # Check if any token in genre_labels appears in IMDB genres string
        def _any_overlap(row_gl, row_imdb):
            if not row_gl or not row_imdb:
                return False
            tokens = set(t.strip() for t in row_gl.split("|") if t.strip())
            return any(t in row_imdb for t in tokens)

        matched_mwg = mwg[mwg["genre_match_flag"] == 1]
        if not matched_mwg.empty:
            gl_matched   = gl[matched_mwg.index]
            imdb_matched = imdb_genre_vals[matched_mwg.index]
            overlap = [_any_overlap(g, i) for g, i in zip(gl_matched, imdb_matched)]
            overlap_pct = sum(overlap) / len(overlap) * 100 if overlap else 0.0
            rows.append({
                "check": "genre_label_vs_imdb_genres_overlap",
                "n_flagged": len(overlap) - sum(overlap),
                "total": len(overlap),
                "flag_pct": round(100 - overlap_pct, 2),
                "severity": "warn" if overlap_pct < 60 else "ok",
                "note": f"{overlap_pct:.1f}% of matched movies share ≥1 genre token with IMDB genres field.",
            })

    # 4. External year impossible vs IMDB year (> 5 years apart where both present)
    if "startYear" in mwg.columns and "genre_year_direct" in mwg.columns:
        sy = pd.to_numeric(mwg["startYear"],         errors="coerce")
        gy = pd.to_numeric(mwg["genre_year_direct"], errors="coerce")
        both = sy.notna() & gy.notna()
        flag4 = both & ((sy - gy).abs() > 5)
        rows.append({
            "check": "startYear_vs_genre_year_delta>5",
            "n_flagged": int(flag4.sum()),
            "total": int(both.sum()),
            "flag_pct": round(flag4[both].mean() * 100, 3) if both.any() else 0.0,
            "severity": "warn" if flag4.sum() > 0 else "ok",
            "note": "Large year discrepancy between IMDB and external genre catalog.",
        })

    return pd.DataFrame(rows)


def build_delta_impact(movies_with_genres: pd.DataFrame) -> pd.DataFrame:
    """Quantifies how many records would materially change if genre-enriched fills
    were used instead of MICE-imputed values.

    Compares cleaned pipeline value vs genre_fill_* for each field.
    Rows without an available genre fill are excluded.
    """
    rows = []
    fill_specs = [
        ("startYear",      "genre_fill_startYear",      2.0,  "years"),
        ("runtimeMinutes", "genre_fill_runtimeMinutes",  10.0, "minutes"),
        ("numVotes_log1p", "genre_fill_numVotes",        0.5,  "log1p units"),
    ]

    mwg = movies_with_genres.copy()
    for field, fill_col, threshold, unit in fill_specs:
        if field not in mwg.columns or fill_col not in mwg.columns:
            continue

        pipeline_val = pd.to_numeric(mwg[field],    errors="coerce")
        fill_val     = pd.to_numeric(mwg[fill_col], errors="coerce")

        comparable = pipeline_val.notna() & fill_val.notna()
        if not comparable.any():
            continue

        delta = (pipeline_val[comparable] - fill_val[comparable]).abs()
        material = delta > threshold

        rows.append({
            "field":                  field,
            "fill_source":            fill_col,
            "fill_available_rows":    int(fill_val.notna().sum()),
            "fill_available_pct":     round(fill_val.notna().mean() * 100, 2),
            "comparable_rows":        int(comparable.sum()),
            "material_delta_rows":    int(material.sum()),
            "material_delta_pct":     round(material.mean() * 100, 2),
            "median_abs_delta":       round(float(delta.median()), 3),
            "p90_abs_delta":          round(float(delta.quantile(0.90)), 3),
            "threshold":              threshold,
            "unit":                   unit,
            "note": (
                f"Genre fill agrees within ±{threshold} {unit} for "
                f"{100 - round(material.mean() * 100, 1):.1f}% of comparable rows."
            ),
        })

    return pd.DataFrame(rows)


def analyze(movies: pd.DataFrame, genre_dir: Path) -> Dict[str, pd.DataFrame]:
    catalog       = load_genre_catalog(genre_dir)
    file_summary  = build_file_summary(catalog)
    tokens        = build_token_long(catalog)
    genre_agg     = aggregate_genre_catalog(catalog, tokens)
    conflict_summary = build_conflict_summary(catalog, genre_agg, tokens)
    centers       = compute_genre_centers(catalog, tokens)
    movies_with_genres = build_movies_with_genres(movies.copy(), genre_agg, centers)
    join_audit    = assess_join_strategies(movies.copy(), genre_agg)
    field_alignment   = build_field_alignment(movies_with_genres)
    recovery_summary  = build_recovery_summary(movies_with_genres)
    fill_candidates   = build_fill_candidates(movies_with_genres)
    top_genres        = build_top_genres(tokens, movies)
    top_tokens_all    = build_top_tokens(tokens)
    schema_profile    = _schema_profile(catalog, "clean")

    subgroup_coverage  = build_subgroup_coverage(movies_with_genres)
    join_precision     = build_join_precision(movies_with_genres)
    consistency_checks = build_consistency_checks(movies_with_genres)
    delta_impact       = build_delta_impact(movies_with_genres)

    return {
        "genre_catalog":            catalog,
        "genre_file_summary":       file_summary,
        "genre_tokens":             tokens,
        "genre_agg":                genre_agg,
        "genre_conflict_summary":   conflict_summary,
        "genre_centers":            centers,
        "movies_with_genres":       movies_with_genres,
        "genre_join_strategy_audit": join_audit,
        "genre_field_alignment":    field_alignment,
        "genre_recovery_summary":   recovery_summary,
        "genre_fill_candidates":    fill_candidates,
        "genre_top_tokens":         top_genres,
        "genre_top_tokens_all":     top_tokens_all,
        "genre_schema_profile":     schema_profile,
        "genre_subgroup_coverage":  subgroup_coverage,
        "genre_join_precision":     join_precision,
        "genre_consistency_checks": consistency_checks,
        "genre_delta_impact":       delta_impact,
    }


# ── Figures ───────────────────────────────────────────────────────────────────

def _fig_join_strategy(join_audit: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = [GRN if d == "use" else RED for d in join_audit["decision"]]
    bars = ax.bar(join_audit["strategy"], join_audit["matched_movies"], color=colors,
                  alpha=0.88, edgecolor="none")
    ax.set_title("Join strategy comparison", fontsize=13, fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    ax.set_ylabel("Matched movies", fontsize=11, color=TXT, labelpad=8)
    labels_str = [s[:18] + "…" if len(s) > 18 else s for s in join_audit["strategy"]]
    ax.set_xticklabels(labels_str, rotation=35, ha="right", fontsize=9, color=MUT)

    y_max = ax.get_ylim()[1]
    for bar, pct in zip(bars, join_audit["coverage_pct"]):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005 * y_max,
                f"{pct:.2f}%", ha="center", va="bottom", fontsize=8, color=TXT)

    fig.tight_layout(pad=2.5)
    fig.savefig(out_dir / "01_join_strategy.png", dpi=130, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)


def _fig_top_genres(top_genres: pd.DataFrame, out_dir: Path) -> None:
    top = top_genres.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.barh(top["genre_token"], top["matched_movies"], color=Y, alpha=0.88, edgecolor="none")
    ax.set_title("Most common matched genre tokens", fontsize=13, fontweight="bold",
                 color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    ax.set_xlabel("Matched movies", fontsize=11, color=TXT, labelpad=8)
    ax.tick_params(axis="y", labelsize=9, labelcolor=MUT)

    x_max = ax.get_xlim()[1]
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005 * x_max, bar.get_y() + bar.get_height() / 2,
                f"{w:.0f}", va="center", fontsize=8, color=TXT)

    fig.tight_layout(pad=2.5)
    fig.savefig(out_dir / "02_top_genres.png", dpi=130, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)


def _fig_recovery(recovery_summary: pd.DataFrame, out_dir: Path) -> None:
    x     = np.arange(len(recovery_summary))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    bars_dir = ax.bar(x - width / 2, recovery_summary["direct_recovered_pct"], width=width,
                      color=GRN, alpha=0.88, edgecolor="none", label="direct key")
    bars_ctr = ax.bar(x + width / 2, recovery_summary["center_recovered_pct"], width=width,
                      color=ORG, alpha=0.88, edgecolor="none", label="genre center")
    ax.set_xticks(x)
    ax.set_xticklabels(recovery_summary["field"], fontsize=9, color=MUT)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Recovered missing rows (%)", fontsize=11, color=TXT, labelpad=8)
    ax.set_title("Recovery potential for originally missing fields", fontsize=13,
                 fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    ax.legend(fontsize=9)

    y_max = ax.get_ylim()[1]
    for bar in list(bars_dir) + list(bars_ctr):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005 * y_max,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8, color=TXT)

    fig.tight_layout(pad=2.5)
    fig.savefig(out_dir / "03_recovery.png", dpi=130, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)


def _fig_subgroup_coverage(subgroup_coverage: pd.DataFrame, out_dir: Path) -> None:
    if subgroup_coverage.empty:
        return
    for gtype in subgroup_coverage["group_type"].unique():
        sub = subgroup_coverage[subgroup_coverage["group_type"] == gtype].copy()
        sub = sub.sort_values("group_value")
        colors = [GRN if s == "ok" else (ORG if s == "warn" else RED) for s in sub["status"]]
        fig, ax = plt.subplots(figsize=(14, 5))
        bars = ax.bar(sub["group_value"], sub["coverage_pct"], color=colors, alpha=0.88,
                      edgecolor="none")
        ax.axhline(80, color=MUT, linestyle="--", linewidth=1, label="80% threshold")
        ax.set_title(f"Genre join coverage by {gtype}", fontsize=13, fontweight="bold",
                     color=TXT, pad=12)
        ax.title.set_position([0.5, 1.02])
        ax.set_ylabel("Coverage (%)", fontsize=11, color=TXT, labelpad=8)
        labels_str = [str(v)[:18] + "…" if len(str(v)) > 18 else str(v) for v in sub["group_value"]]
        ax.set_xticklabels(labels_str, rotation=35, ha="right", fontsize=9, color=MUT)
        ax.set_ylim(0, 110)
        ax.legend(fontsize=9)

        y_max = ax.get_ylim()[1]
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005 * y_max,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8, color=TXT)

        fig.tight_layout(pad=2.5)
        fname = f"04_subgroup_coverage_{gtype}.png"
        fig.savefig(out_dir / fname, dpi=130, bbox_inches="tight", facecolor=BG, edgecolor="none")
        plt.close(fig)


def _fig_join_precision(join_precision: pd.DataFrame, out_dir: Path) -> None:
    if join_precision.empty:
        return
    metrics = ["year_consistent_pct", "runtime_consistent_pct", "estimated_precision_floor"]
    labels  = ["Year consistent (%)", "Runtime consistent (%)", "Est. precision floor (%)"]
    vals    = [join_precision[m].iloc[0] if m in join_precision.columns else float("nan") for m in metrics]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(labels, vals, color=[GRN, GRN, Y], alpha=0.88, edgecolor="none")
    ax.set_ylim(0, 110)
    ax.set_title("Join precision estimates (direct tconst strategy)", fontsize=13,
                 fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    ax.set_ylabel("Percentage (%)", fontsize=11, color=TXT, labelpad=8)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9, color=MUT)

    y_max = ax.get_ylim()[1]
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005 * y_max,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=8, color=TXT)

    fig.tight_layout(pad=2.5)
    fig.savefig(out_dir / "05_join_precision.png", dpi=130, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)


def _fig_delta_impact(delta_impact: pd.DataFrame, out_dir: Path) -> None:
    if delta_impact.empty:
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    x     = np.arange(len(delta_impact))
    width = 0.4
    bars_avail = ax.bar(x - width / 2, delta_impact["fill_available_pct"], width=width,
                        color=GRN, alpha=0.88, edgecolor="none", label="Fill available (%)")
    bars_delta = ax.bar(x + width / 2, delta_impact["material_delta_pct"], width=width,
                        color=RED, alpha=0.88, edgecolor="none", label="Material delta (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(delta_impact["field"], fontsize=9, color=MUT)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Percentage of rows", fontsize=11, color=TXT, labelpad=8)
    ax.set_title("Genre fill availability vs material delta vs MICE", fontsize=13,
                 fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    ax.legend(fontsize=9)

    y_max = ax.get_ylim()[1]
    for bar in list(bars_avail) + list(bars_delta):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005 * y_max,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8, color=TXT)

    fig.tight_layout(pad=2.5)
    fig.savefig(out_dir / "06_delta_impact.png", dpi=130, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)


def _fig_consistency_checks(consistency_checks: pd.DataFrame, out_dir: Path) -> None:
    if consistency_checks.empty:
        return
    colors = [GRN if s == "ok" else ORG for s in consistency_checks["severity"]]
    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(consistency_checks["check"], consistency_checks["flag_pct"], color=colors,
                  alpha=0.88, edgecolor="none")
    ax.set_title("Enrichment consistency checks — flag rate per rule", fontsize=13,
                 fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    ax.set_ylabel("Flag rate (%)", fontsize=11, color=TXT, labelpad=8)
    labels_str = [str(c)[:18] + "…" if len(str(c)) > 18 else str(c)
                  for c in consistency_checks["check"]]
    ax.set_xticklabels(labels_str, rotation=40, ha="right", fontsize=9, color=MUT)

    y_max = ax.get_ylim()[1]
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005 * y_max,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8, color=TXT)

    fig.tight_layout(pad=2.5)
    fig.savefig(out_dir / "07_consistency_checks.png", dpi=130, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)


# ── Output ────────────────────────────────────────────────────────────────────

def write_outputs(artifacts: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts["movies_with_genres"].to_csv(       out_dir / "movies_with_genres.csv",        index=False)
    artifacts["genre_file_summary"].to_csv(        out_dir / "genre_file_summary.csv",         index=False)
    artifacts["genre_conflict_summary"].to_csv(    out_dir / "genre_conflict_summary.csv",     index=False)
    artifacts["genre_join_strategy_audit"].to_csv( out_dir / "genre_join_strategy_audit.csv",  index=False)
    artifacts["genre_field_alignment"].to_csv(     out_dir / "genre_field_alignment.csv",      index=False)
    artifacts["genre_recovery_summary"].to_csv(    out_dir / "genre_recovery_summary.csv",     index=False)
    artifacts["genre_fill_candidates"].to_csv(     out_dir / "genre_fill_candidates.csv",      index=False)
    artifacts["genre_centers"].to_csv(             out_dir / "genre_token_centers.csv",        index=False)
    artifacts["genre_top_tokens"].to_csv(          out_dir / "genre_top_tokens.csv",           index=False)
    artifacts["genre_top_tokens_all"].to_csv(      out_dir / "genre_top_tokens_all.csv",       index=False)
    artifacts["genre_schema_profile"].to_csv(      out_dir / "genre_schema_profile.csv",       index=False)

    artifacts["genre_subgroup_coverage"].to_csv(  out_dir / "genre_subgroup_coverage.csv",   index=False)
    artifacts["genre_join_precision"].to_csv(     out_dir / "genre_join_precision.csv",      index=False)
    artifacts["genre_consistency_checks"].to_csv( out_dir / "genre_consistency_checks.csv",  index=False)
    artifacts["genre_delta_impact"].to_csv(       out_dir / "genre_delta_impact.csv",        index=False)

    _fig_join_strategy(artifacts["genre_join_strategy_audit"], out_dir)
    _fig_top_genres(artifacts["genre_top_tokens"], out_dir)
    _fig_recovery(artifacts["genre_recovery_summary"], out_dir)
    _fig_subgroup_coverage(artifacts["genre_subgroup_coverage"], out_dir)
    _fig_join_precision(artifacts["genre_join_precision"], out_dir)
    _fig_delta_impact(artifacts["genre_delta_impact"], out_dir)
    _fig_consistency_checks(artifacts["genre_consistency_checks"], out_dir)


def _attach_state(state: dict, artifacts: Dict[str, pd.DataFrame]) -> dict:
    for key in [
        "movies_with_genres", "genre_file_summary", "genre_conflict_summary",
        "genre_join_strategy_audit", "genre_field_alignment", "genre_recovery_summary",
        "genre_fill_candidates", "genre_top_tokens", "genre_top_tokens_all",
        "genre_centers", "genre_schema_profile",
        "genre_subgroup_coverage", "genre_join_precision",
        "genre_consistency_checks", "genre_delta_impact",
    ]:
        state[key] = artifacts[key]
    return state


# ── Entry point ───────────────────────────────────────────────────────────────

def run(state: dict, genre_dir: Path | None = None, out_dir: Path | None = None) -> dict:
    """Run genre enrichment.

    Parameters
    ----------
    state    : pipeline state dict; may contain 'movies_clean' (DataFrame).
    genre_dir: path to Movies_by_Genre folder (default: data/Movies_by_Genre/).
    out_dir  : output directory (default: pipeline/outputs/enrichment/).
    """
    genre_dir = Path(genre_dir) if genre_dir else _GENRE_DIR
    out_dir   = Path(out_dir)   if out_dir   else _OUT_DIR

    movies = state.get("movies_clean")
    if movies is None:
        parquet = _CLEAN_DIR / "train_clean.parquet"
        if not parquet.exists():
            raise FileNotFoundError(
                f"Clean parquet not found at {parquet} — run the cleaning stage first."
            )
        movies = pd.read_parquet(parquet)
        print(f"[enrich_genre] Loaded {parquet.name}  shape={movies.shape}")

    artifacts = analyze(movies, genre_dir)
    write_outputs(artifacts, out_dir)
    state = _attach_state(state, artifacts)

    mwg      = artifacts["movies_with_genres"]
    matched  = int(mwg["genre_match_flag"].sum())
    coverage = matched / len(mwg) * 100 if len(mwg) else 0.0
    print(f"[enrich_genre] Matched {matched:,} / {len(mwg):,} rows ({coverage:.1f}% coverage)")
    print(f"[enrich_genre] Outputs → {out_dir}")
    return state


if __name__ == "__main__":
    run({})
