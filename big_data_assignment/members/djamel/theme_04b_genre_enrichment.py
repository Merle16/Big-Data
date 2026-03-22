#!/usr/bin/env python3
"""
Theme 04b - Genre Enrichment and Recovery Audit

Audits the Movies_by_Genre folder as an external IMDb-aligned catalog, validates
the safest join key, couples genre labels back to the cleaned movie table, and
derives cautious fill candidates for fields that were originally missing.

This stage is intentionally conservative:
  * Production join uses direct IMDb title IDs only (movie_id -> tconst).
  * Title-based fallback matching is measured for audit purposes, then rejected.
  * External rating / votes are stored as enrichment columns, not injected into
    the model matrix automatically, because they need leakage review.
"""
from __future__ import annotations

import base64
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List

os.environ.setdefault("KMP_ENABLE_SHM", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MEMBER = Path(__file__).resolve().parent
ROOT = MEMBER.parent.parent
OUT = MEMBER / "outputs_restart"
OUT.mkdir(exist_ok=True)
GENRE_ROOT = ROOT / "data" / "raw" / "Movies_by_Genre"

Y = "#F5C518"
B = "#1848f5"
BG = "#0a0a0a"
CRD = "#141414"
TXT = "#ffffff"
MUT = "#888888"
GRN = "#2ecc71"
RED = "#e74c3c"
ORG = "#f39c12"

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

CSS = f"""
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:{BG};color:{TXT};font-family:'Segoe UI',sans-serif;padding:24px;line-height:1.6}}
h1{{color:{Y};font-size:2rem;margin-bottom:6px}}
h2{{color:{Y};font-size:1.3rem;margin:0 0 12px}}
h3{{color:{ORG};font-size:1rem;margin:12px 0 6px}}
.subtitle{{color:{MUT};margin-bottom:28px;font-size:.95rem}}
.card{{background:{CRD};border:1px solid #222;border-radius:12px;padding:20px;margin-bottom:24px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start}}
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
code{{color:{Y}}}
"""


def _b64(fig: plt.Figure) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _img(fig: plt.Figure) -> str:
    return f'<img src="data:image/png;base64,{_b64(fig)}">'


def _card(title: str, body: str, color: str = Y) -> str:
    return f'<div class="card"><h2 style="color:{color}">{title}</h2>{body}</div>'


def _kpi(val: str, lbl: str) -> str:
    return f'<div class="kpi"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>'


def _page(title: str, subtitle: str, kpis: str, sections: Iterable[str]) -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{title}</title><style>{CSS}</style></head><body>
<h1>{title}</h1><p class="subtitle">{subtitle}</p>
<div class="kpi-grid">{kpis}</div>{"".join(sections)}</body></html>"""


def _norm_title(title) -> str:
    if pd.isna(title):
        return ""
    text = str(title).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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


def _mode_or_first(series: pd.Series):
    s = series.dropna().astype(str).str.strip()
    s = s[s.ne("")]
    if s.empty:
        return pd.NA
    vc = s.value_counts()
    return vc.index[0]


def _join_unique(values: Iterable[str]):
    vals = sorted({str(v).strip() for v in values if pd.notna(v) and str(v).strip()})
    return "|".join(vals) if vals else pd.NA


def _schema_profile(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        non_null = int(df[col].notna().sum())
        rows.append(
            {
                "stage": label,
                "column": col,
                "dtype": str(df[col].dtype),
                "non_null": non_null,
                "missing_pct": round((1 - non_null / len(df)) * 100, 2) if len(df) else 0.0,
                "unique_non_null": int(df[col].dropna().nunique()),
            }
        )
    return pd.DataFrame(rows)


def build_file_summary(catalog: pd.DataFrame) -> pd.DataFrame:
    grouped = catalog.groupby("genre_source_file", dropna=False)
    summary = grouped.agg(
        rows=("tconst", "size"),
        unique_tconst=("tconst", "nunique"),
        missing_year_pct=("genre_year", lambda s: round(s.isna().mean() * 100, 2)),
        missing_runtime_pct=("genre_runtime_minutes", lambda s: round(s.isna().mean() * 100, 2)),
        missing_rating_pct=("genre_rating", lambda s: round(s.isna().mean() * 100, 2)),
        missing_votes_pct=("genre_votes", lambda s: round(s.isna().mean() * 100, 2)),
        missing_gross_pct=("genre_gross_usd", lambda s: round(s.isna().mean() * 100, 2)),
    ).reset_index().sort_values("rows", ascending=False).reset_index(drop=True)
    return summary


def load_genre_catalog() -> pd.DataFrame:
    if not GENRE_ROOT.exists():
        raise FileNotFoundError(f"Genre folder not found: {GENRE_ROOT}")

    frames: List[pd.DataFrame] = []
    for csv_path in sorted(GENRE_ROOT.glob("*.csv")):
        df = pd.read_csv(csv_path)
        df["genre_source_file"] = csv_path.stem
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    clean = raw.rename(
        columns={
            "movie_id": "tconst",
            "movie_name": "genre_movie_title",
            "genre": "genre_text",
            "certificate": "genre_certificate",
            "rating": "genre_rating",
            "description": "genre_description",
            "director": "genre_director",
            "director_id": "genre_director_id",
            "star": "genre_star",
            "star_id": "genre_star_id",
            "votes": "genre_votes",
            "gross(in $)": "genre_gross_usd",
        }
    ).copy()

    clean["tconst"] = clean["tconst"].astype("string").str.strip()
    clean["genre_source_tag"] = clean["genre_source_file"].map(SOURCE_TAG_MAP).fillna(clean["genre_source_file"])
    clean["genre_title_key"] = clean["genre_movie_title"].map(_norm_title)
    clean["genre_year"] = pd.to_numeric(clean["year"], errors="coerce")
    clean["genre_runtime_minutes"] = pd.to_numeric(
        clean["runtime"].astype("string").str.extract(r"(\d+)")[0],
        errors="coerce",
    )
    clean["genre_rating"] = pd.to_numeric(clean["genre_rating"], errors="coerce")
    clean["genre_votes"] = pd.to_numeric(clean["genre_votes"], errors="coerce")
    clean["genre_gross_usd"] = pd.to_numeric(
        clean["genre_gross_usd"].astype("string").str.replace(r"[$,]", "", regex=True),
        errors="coerce",
    )
    return clean


def build_token_long(catalog: pd.DataFrame) -> pd.DataFrame:
    parsed = catalog[["tconst", "genre_text"]].copy()
    parsed["genre_token"] = parsed["genre_text"].apply(_split_genres)
    parsed = parsed.explode("genre_token")
    parsed = parsed.drop(columns=["genre_text"])

    source_tags = catalog[["tconst", "genre_source_tag"]].rename(columns={"genre_source_tag": "genre_token"})
    tokens = pd.concat([parsed, source_tags], ignore_index=True)
    tokens["genre_token"] = tokens["genre_token"].astype("string").str.strip().str.lower()
    tokens["genre_token"] = tokens["genre_token"].replace({"<na>": pd.NA, "": pd.NA, "nan": pd.NA})
    tokens = tokens.dropna(subset=["genre_token"]).drop_duplicates()
    return tokens


def aggregate_genre_catalog(catalog: pd.DataFrame, tokens: pd.DataFrame) -> pd.DataFrame:
    numeric = (
        catalog.groupby("tconst", dropna=False)[
            ["genre_year", "genre_runtime_minutes", "genre_votes", "genre_rating", "genre_gross_usd"]
        ]
        .median()
        .rename(
            columns={
                "genre_year": "genre_year_direct",
                "genre_runtime_minutes": "genre_runtime_direct",
                "genre_votes": "genre_votes_direct",
                "genre_rating": "genre_rating_direct",
                "genre_gross_usd": "genre_gross_direct",
            }
        )
        .reset_index()
    )

    counts = pd.DataFrame(
        {
            "tconst": numeric["tconst"],
            "genre_external_rows": catalog.groupby("tconst", dropna=False).size().values,
            "genre_source_file_count": catalog.groupby("tconst", dropna=False)["genre_source_file"].nunique().values,
            "genre_title_variant_count": catalog.groupby("tconst", dropna=False)["genre_title_key"].nunique(dropna=True).values,
            "genre_year_variant_count": catalog.groupby("tconst", dropna=False)["genre_year"].nunique(dropna=True).values,
            "genre_runtime_variant_count": catalog.groupby("tconst", dropna=False)["genre_runtime_minutes"].nunique(dropna=True).values,
            "genre_votes_variant_count": catalog.groupby("tconst", dropna=False)["genre_votes"].nunique(dropna=True).values,
            "genre_rating_variant_count": catalog.groupby("tconst", dropna=False)["genre_rating"].nunique(dropna=True).values,
        }
    )

    title_first = (
        catalog.sort_values(["tconst", "genre_movie_title", "genre_title_key"], kind="mergesort")
        .drop_duplicates("tconst")[["tconst", "genre_movie_title", "genre_title_key"]]
        .rename(
            columns={
                "genre_movie_title": "genre_title_external",
                "genre_title_key": "genre_title_key_external",
            }
        )
    )

    token_summary = (
        tokens.sort_values(["tconst", "genre_token"], kind="mergesort")
        .groupby("tconst", dropna=False)
        .agg(genre_token_count=("genre_token", "size"), genre_labels=("genre_token", "|".join))
        .reset_index()
    )

    out = numeric.merge(counts, on="tconst", how="left").merge(title_first, on="tconst", how="left").merge(
        token_summary, on="tconst", how="left"
    )
    for col in [
        "genre_title_variant_count",
        "genre_year_variant_count",
        "genre_runtime_variant_count",
        "genre_votes_variant_count",
        "genre_rating_variant_count",
        "genre_token_count",
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

    centers = token_frame.groupby("genre_token", dropna=False).agg(
        matched_movies=("tconst", "nunique"),
        median_year=("genre_year", "median"),
        median_runtime_minutes=("genre_runtime_minutes", "median"),
        median_rating=("genre_rating", "median"),
        median_votes_log1p=("genre_votes_log1p", "median"),
        median_gross_log1p=("genre_gross_log1p", "median"),
        votes_iqr=("genre_votes_log1p", lambda s: s.quantile(0.75) - s.quantile(0.25)),
        runtime_iqr=("genre_runtime_minutes", lambda s: s.quantile(0.75) - s.quantile(0.25)),
    ).reset_index()

    centers["median_votes"] = np.expm1(centers["median_votes_log1p"])
    centers["median_gross_usd"] = np.expm1(centers["median_gross_log1p"])
    centers = centers.sort_values(["matched_movies", "genre_token"], ascending=[False, True]).reset_index(drop=True)
    return centers


def assess_join_strategies(movies: pd.DataFrame, genre_agg: pd.DataFrame) -> pd.DataFrame:
    direct_ids = set(genre_agg["tconst"].dropna().astype(str))
    unmatched = movies.loc[~movies["tconst"].astype(str).isin(direct_ids)].copy()
    unmatched["title_key"] = unmatched["primaryTitle"].map(_norm_title)
    unmatched["startYear_round"] = pd.to_numeric(unmatched["startYear"], errors="coerce").round().astype("Int64")
    unmatched["runtime_round"] = pd.to_numeric(unmatched["runtimeMinutes"], errors="coerce").round().astype("Int64")

    key_table = genre_agg[["tconst", "genre_title_key_external", "genre_year_direct", "genre_runtime_direct"]].copy()
    key_table["genre_year_round"] = pd.to_numeric(key_table["genre_year_direct"], errors="coerce").round().astype("Int64")
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
    rows = [
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
    ]
    return pd.DataFrame(rows)


def attach_genre_centers(movies_with_genres: pd.DataFrame, centers: pd.DataFrame) -> pd.DataFrame:
    center_lookup = centers.set_index("genre_token").to_dict(orient="index")

    def _per_row_center(label_text, key):
        if pd.isna(label_text):
            return np.nan
        vals = []
        for token in str(label_text).split("|"):
            token = token.strip().lower()
            if token and token in center_lookup:
                val = center_lookup[token].get(key)
                if pd.notna(val):
                    vals.append(float(val))
        if not vals:
            return np.nan
        return float(np.median(vals))

    out = movies_with_genres.copy()
    out["genre_center_year"] = out["genre_labels"].apply(lambda s: _per_row_center(s, "median_year"))
    out["genre_center_runtime"] = out["genre_labels"].apply(lambda s: _per_row_center(s, "median_runtime_minutes"))
    out["genre_center_rating"] = out["genre_labels"].apply(lambda s: _per_row_center(s, "median_rating"))
    out["genre_center_votes"] = out["genre_labels"].apply(lambda s: _per_row_center(s, "median_votes"))
    out["genre_center_gross_usd"] = out["genre_labels"].apply(lambda s: _per_row_center(s, "median_gross_usd"))
    return out


def build_movies_with_genres(movies: pd.DataFrame, genre_agg: pd.DataFrame, centers: pd.DataFrame) -> pd.DataFrame:
    out = movies.merge(genre_agg, on="tconst", how="left")
    out["genre_match_flag"] = out["genre_external_rows"].notna().astype(int)
    out["genre_title_key"] = out["primaryTitle"].map(_norm_title)

    out["genre_title_exact"] = (
        (out["genre_title_key"] != "")
        & out["genre_title_key"].eq(out["genre_title_key_external"].fillna(""))
    ).astype(int)
    out["genre_year_abs_diff"] = (pd.to_numeric(out["startYear"], errors="coerce") - out["genre_year_direct"]).abs()
    out["genre_runtime_abs_diff"] = (pd.to_numeric(out["runtimeMinutes"], errors="coerce") - out["genre_runtime_direct"]).abs()

    out = attach_genre_centers(out, centers)

    out["genre_fill_startYear"] = out["genre_year_direct"].where(out["genre_year_direct"].notna(), out["genre_center_year"])
    out["genre_fill_runtimeMinutes"] = out["genre_runtime_direct"].where(out["genre_runtime_direct"].notna(), out["genre_center_runtime"])
    out["genre_fill_numVotes"] = out["genre_votes_direct"].where(out["genre_votes_direct"].notna(), out["genre_center_votes"])

    out["genre_fill_startYear_source"] = np.select(
        [out["genre_year_direct"].notna(), out["genre_center_year"].notna()],
        ["direct_tconst", "genre_center"],
        default="none",
    )
    out["genre_fill_runtime_source"] = np.select(
        [out["genre_runtime_direct"].notna(), out["genre_center_runtime"].notna()],
        ["direct_tconst", "genre_center"],
        default="none",
    )
    out["genre_fill_numVotes_source"] = np.select(
        [out["genre_votes_direct"].notna(), out["genre_center_votes"].notna()],
        ["direct_tconst", "genre_center"],
        default="none",
    )
    return out


def build_field_alignment(movies_with_genres: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in FIELD_SPECS:
        field = spec["field"]
        indicator = spec["indicator"]
        direct = spec["direct"]
        sub = movies_with_genres[
            (movies_with_genres[indicator] == 0)
            & movies_with_genres[direct].notna()
            & movies_with_genres[field].notna()
        ].copy()
        if sub.empty:
            rows.append(
                {
                    "field": field,
                    "comparison_rows": 0,
                    "exact_match_pct": np.nan,
                    spec["tolerance_label"]: np.nan,
                    "median_abs_diff": np.nan,
                    "correlation": np.nan,
                    "note": "No comparable matched rows.",
                }
            )
            continue

        diff = (pd.to_numeric(sub[field], errors="coerce") - pd.to_numeric(sub[direct], errors="coerce")).abs()
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
            row["note"] = "High rank agreement but snapshot drift is visible, so direct fills need caution."
        else:
            row[spec["tolerance_label"]] = round((diff <= spec["tolerance"]).mean() * 100, 2)
            row["note"] = "Direct external values align closely enough to act as credible fill candidates."
        rows.append(row)
    return pd.DataFrame(rows)


def build_recovery_summary(movies_with_genres: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in FIELD_SPECS:
        indicator = spec["indicator"]
        direct = spec["direct"]
        center = spec["center"]
        sub = movies_with_genres[movies_with_genres[indicator] == 1].copy()
        total = int(len(sub))
        direct_hits = int(sub[direct].notna().sum())
        center_hits = int(sub[center].notna().sum())
        any_hits = int((sub[direct].notna() | sub[center].notna()).sum())
        rows.append(
            {
                "field": spec["field"],
                "missing_rows": total,
                "direct_recovered_rows": direct_hits,
                "direct_recovered_pct": round((direct_hits / total) * 100, 2) if total else 0.0,
                "center_recovered_rows": center_hits,
                "center_recovered_pct": round((center_hits / total) * 100, 2) if total else 0.0,
                "combined_recovered_rows": any_hits,
                "combined_recovered_pct": round((any_hits / total) * 100, 2) if total else 0.0,
                "recommended_use": (
                    "direct_tconst"
                    if spec["field"] in {"startYear", "runtimeMinutes"}
                    else "direct_tconst_with_snapshot_caution"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_fill_candidates(movies_with_genres: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "tconst",
        "primaryTitle",
        "split",
        "genre_labels",
        "genre_match_flag",
        "genre_source_file_count",
        "startYear_was_missing",
        "genre_year_direct",
        "genre_center_year",
        "genre_fill_startYear",
        "genre_fill_startYear_source",
        "runtimeMinutes_was_missing",
        "genre_runtime_direct",
        "genre_center_runtime",
        "genre_fill_runtimeMinutes",
        "genre_fill_runtime_source",
        "numVotes_was_missing",
        "genre_votes_direct",
        "genre_center_votes",
        "genre_fill_numVotes",
        "genre_fill_numVotes_source",
        "genre_rating_direct",
        "genre_gross_direct",
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
    top = (
        tokens[tokens["tconst"].astype(str).isin(matched_ids)]
        .groupby("genre_token", dropna=False)["tconst"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index(name="matched_movies")
    )
    return top


def build_top_tokens(tokens: pd.DataFrame) -> pd.DataFrame:
    return (
        tokens.groupby("genre_token", dropna=False)["tconst"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index(name="catalog_movies")
    )


def build_conflict_summary(catalog: pd.DataFrame, genre_agg: pd.DataFrame, tokens: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"metric": "raw_rows", "value": int(len(catalog)), "detail": "Rows across all 16 genre CSV files."},
        {"metric": "unique_tconst", "value": int(catalog["tconst"].nunique()), "detail": "Unique IMDb title IDs in the external catalog."},
        {
            "metric": "multi_source_movies",
            "value": int((genre_agg["genre_source_file_count"] > 1).sum()),
            "detail": "Movies appearing in more than one genre source file.",
        },
        {
            "metric": "title_conflict_movies",
            "value": int((genre_agg["genre_title_variant_count"] > 1).sum()),
            "detail": "Same IMDb ID with multiple normalized titles.",
        },
        {
            "metric": "year_conflict_movies",
            "value": int((genre_agg["genre_year_variant_count"] > 1).sum()),
            "detail": "Same IMDb ID with more than one external year value.",
        },
        {
            "metric": "runtime_conflict_movies",
            "value": int((genre_agg["genre_runtime_variant_count"] > 1).sum()),
            "detail": "Same IMDb ID with more than one runtime value.",
        },
        {
            "metric": "votes_conflict_movies",
            "value": int((genre_agg["genre_votes_variant_count"] > 1).sum()),
            "detail": "Same IMDb ID with more than one vote snapshot.",
        },
        {
            "metric": "rating_conflict_movies",
            "value": int((genre_agg["genre_rating_variant_count"] > 1).sum()),
            "detail": "Same IMDb ID with more than one rating snapshot.",
        },
        {
            "metric": "multi_token_movies",
            "value": int(tokens.groupby("tconst")["genre_token"].nunique().gt(1).sum()),
            "detail": "Movies associated with more than one genre/token after normalization.",
        },
    ]
    return pd.DataFrame(rows)


def analyze(movies: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    genre_catalog = load_genre_catalog()
    file_summary = build_file_summary(genre_catalog)
    tokens = build_token_long(genre_catalog)
    genre_agg = aggregate_genre_catalog(genre_catalog, tokens)
    conflict_summary = build_conflict_summary(genre_catalog, genre_agg, tokens)
    centers = compute_genre_centers(genre_catalog, tokens)
    movies_with_genres = build_movies_with_genres(movies.copy(), genre_agg, centers)
    join_audit = assess_join_strategies(movies.copy(), genre_agg)
    field_alignment = build_field_alignment(movies_with_genres)
    recovery_summary = build_recovery_summary(movies_with_genres)
    fill_candidates = build_fill_candidates(movies_with_genres)
    top_genres = build_top_genres(tokens, movies)
    top_tokens_all = build_top_tokens(tokens)
    schema_profile = _schema_profile(genre_catalog, "clean")

    return {
        "genre_catalog": genre_catalog,
        "genre_file_summary": file_summary,
        "genre_tokens": tokens,
        "genre_agg": genre_agg,
        "genre_conflict_summary": conflict_summary,
        "genre_centers": centers,
        "movies_with_genres": movies_with_genres,
        "genre_join_strategy_audit": join_audit,
        "genre_field_alignment": field_alignment,
        "genre_recovery_summary": recovery_summary,
        "genre_fill_candidates": fill_candidates,
        "genre_top_tokens": top_genres,
        "genre_top_tokens_all": top_tokens_all,
        "genre_schema_profile": schema_profile,
    }


def _fig_join_strategy(join_audit: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=BG)
    ax.set_facecolor(CRD)
    colors = [GRN if d == "use" else RED for d in join_audit["decision"]]
    bars = ax.bar(join_audit["strategy"], join_audit["matched_movies"], color=colors, alpha=0.9)
    ax.set_title("Join strategy comparison", color=TXT, fontsize=11)
    ax.set_ylabel("Matched movies", color=TXT)
    ax.tick_params(axis="x", rotation=20, labelcolor=TXT)
    ax.tick_params(axis="y", colors=TXT)
    for sp in ax.spines.values():
        sp.set_color(MUT)
    for bar, pct in zip(bars, join_audit["coverage_pct"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(5, bar.get_height() * 0.01),
            f"{pct:.2f}%",
            ha="center",
            va="bottom",
            color=TXT,
            fontsize=8,
        )
    fig.tight_layout()
    return _img(fig)


def _fig_top_genres(top_genres: pd.DataFrame) -> str:
    top = top_genres.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    ax.set_facecolor(CRD)
    ax.barh(top["genre_token"], top["matched_movies"], color=Y, alpha=0.9)
    ax.set_title("Most common matched genre tokens", color=TXT, fontsize=11)
    ax.set_xlabel("Matched movies", color=TXT)
    ax.tick_params(colors=TXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(MUT)
    fig.tight_layout()
    return _img(fig)


def _fig_recovery(recovery_summary: pd.DataFrame) -> str:
    x = np.arange(len(recovery_summary))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
    ax.set_facecolor(CRD)
    ax.bar(x - width / 2, recovery_summary["direct_recovered_pct"], width=width, color=GRN, label="direct key")
    ax.bar(x + width / 2, recovery_summary["center_recovered_pct"], width=width, color=ORG, label="genre center")
    ax.set_xticks(x)
    ax.set_xticklabels(recovery_summary["field"], color=TXT)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Recovered missing rows (%)", color=TXT)
    ax.set_title("Recovery potential for originally missing fields", color=TXT, fontsize=11)
    ax.tick_params(axis="y", colors=TXT)
    ax.legend(facecolor=CRD, edgecolor=MUT, labelcolor=TXT)
    for sp in ax.spines.values():
        sp.set_color(MUT)
    fig.tight_layout()
    return _img(fig)


def _table(df: pd.DataFrame, rows: int | None = None) -> str:
    if rows is not None:
        df = df.head(rows)
    return df.to_html(index=False, border=0, escape=False)


def write_outputs(artifacts: Dict[str, pd.DataFrame]) -> None:
    artifacts["movies_with_genres"].to_csv(OUT / "movies_with_genres.csv", index=False)
    artifacts["genre_file_summary"].to_csv(OUT / "genre_file_summary.csv", index=False)
    artifacts["genre_conflict_summary"].to_csv(OUT / "genre_conflict_summary.csv", index=False)
    artifacts["genre_join_strategy_audit"].to_csv(OUT / "genre_join_strategy_audit.csv", index=False)
    artifacts["genre_field_alignment"].to_csv(OUT / "genre_field_alignment.csv", index=False)
    artifacts["genre_recovery_summary"].to_csv(OUT / "genre_recovery_summary.csv", index=False)
    artifacts["genre_fill_candidates"].to_csv(OUT / "genre_fill_candidates.csv", index=False)
    artifacts["genre_centers"].to_csv(OUT / "genre_token_centers.csv", index=False)
    artifacts["genre_top_tokens"].to_csv(OUT / "genre_top_tokens.csv", index=False)
    artifacts["genre_top_tokens_all"].to_csv(OUT / "genre_top_tokens_all.csv", index=False)
    artifacts["genre_schema_profile"].to_csv(OUT / "genre_schema_profile.csv", index=False)


def attach_state(state: dict, artifacts: Dict[str, pd.DataFrame]) -> dict:
    state["movies_with_genres"] = artifacts["movies_with_genres"]
    state["genre_file_summary"] = artifacts["genre_file_summary"]
    state["genre_conflict_summary"] = artifacts["genre_conflict_summary"]
    state["genre_join_strategy_audit"] = artifacts["genre_join_strategy_audit"]
    state["genre_field_alignment"] = artifacts["genre_field_alignment"]
    state["genre_recovery_summary"] = artifacts["genre_recovery_summary"]
    state["genre_fill_candidates"] = artifacts["genre_fill_candidates"]
    state["genre_top_tokens"] = artifacts["genre_top_tokens"]
    state["genre_top_tokens_all"] = artifacts["genre_top_tokens_all"]
    state["genre_centers"] = artifacts["genre_centers"]
    state["genre_schema_profile"] = artifacts["genre_schema_profile"]
    return state


def write_html(artifacts: Dict[str, pd.DataFrame], movies: pd.DataFrame) -> None:
    join_audit = artifacts["genre_join_strategy_audit"]
    recovery = artifacts["genre_recovery_summary"]
    alignment = artifacts["genre_field_alignment"]
    top_genres = artifacts["genre_top_tokens"]
    fill_candidates = artifacts["genre_fill_candidates"]
    movies_with_genres = artifacts["movies_with_genres"]

    matched = int(movies_with_genres["genre_match_flag"].sum())
    coverage = matched / len(movies_with_genres) * 100 if len(movies_with_genres) else 0.0
    start_recovery = recovery.loc[recovery["field"] == "startYear", "direct_recovered_pct"].iloc[0]
    votes_recovery = recovery.loc[recovery["field"] == "numVotes", "direct_recovered_pct"].iloc[0]

    kpis = (
        _kpi(f"{matched:,}", "Matched movies")
        + _kpi(f"{coverage:.1f}%", "Coverage of cleaned table")
        + _kpi(f"{start_recovery:.1f}%", "Direct recovery for<br>missing startYear")
        + _kpi(f"{votes_recovery:.1f}%", "Direct recovery for<br>missing numVotes")
    )

    sections = [
        _card(
            "1. Join strategy audit",
            f"""
<div class="note">
The external genre files already carry IMDb title IDs, so the production join uses <code>movie_id -> tconst</code>.
Fallback title matching was tested only as an audit path and rejected.
</div>
{_table(join_audit)}
""",
        ),
        _card(
            "2. External field alignment",
            f"""
<div class="note">
<code>startYear</code> and <code>runtimeMinutes</code> align closely enough to be credible direct fill candidates.
<code>numVotes</code> has very high correlation but visible snapshot drift, so it should be treated as an external recovery suggestion, not a silent overwrite.
</div>
{_table(alignment)}
""",
        ),
        _card(
            "3. Missing-value recovery potential",
            f"""
{_table(recovery)}
""",
        ),
        _card(
            "4. Genre coverage and robust centers",
            f"""
<div class="grid2">
  <div>{_table(top_genres, rows=15)}</div>
  <div>
    <div class="note">
    Robust centers are genre-level medians taken from the external catalog. They are intended as fallback candidates when an exact direct external value is unavailable.
    </div>
    {_table(artifacts["genre_centers"], rows=15)}
  </div>
</div>
""",
        ),
        _card(
            "5. Candidate rows for manual review",
            f"""
<div class="note">
These rows had at least one field originally missing in the IMDb training pipeline and at least one external recovery candidate available.
The enriched file stores both the direct external value and the genre-center fallback.
</div>
{_table(fill_candidates, rows=25)}
""",
        ),
    ]

    html = _page(
        "Theme 04b - Genre Enrichment and Recovery Audit",
        "Direct IMDb-key join, genre coupling, cautious recovery candidates, and genre-level robust centers",
        kpis,
        sections,
    )
    (OUT / "theme_04b_genre_enrichment.html").write_text(html, encoding="utf-8")


def run(state: dict) -> dict:
    movies = state.get("movies_clean")
    if movies is None:
        movies_path = OUT / "movies_clean.csv"
        if not movies_path.exists():
            raise FileNotFoundError("movies_clean.csv not found - run theme_04 first.")
        movies = pd.read_csv(movies_path)

    artifacts = analyze(movies)
    write_outputs(artifacts)
    write_html(artifacts, movies)
    state = attach_state(state, artifacts)

    matched = int(artifacts["movies_with_genres"]["genre_match_flag"].sum())
    coverage = matched / len(artifacts["movies_with_genres"]) * 100 if len(artifacts["movies_with_genres"]) else 0.0
    print(f"[theme_04b] Matched cleaned movies to genre catalog: {matched:,} ({coverage:.1f}%)")
    print("[theme_04b] Production key: direct movie_id -> tconst")
    print("[theme_04b] Wrote movies_with_genres.csv, genre_join_strategy_audit.csv, genre_recovery_summary.csv")
    print(f"[theme_04b] Wrote {OUT}/theme_04b_genre_enrichment.html")
    return state


if __name__ == "__main__":
    run({})
    print("[theme_04b] Done.")
