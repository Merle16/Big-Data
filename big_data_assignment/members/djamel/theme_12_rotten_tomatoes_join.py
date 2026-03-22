#!/usr/bin/env python3
"""
Theme 12 — Rotten Tomatoes Join Strategy
========================================
Shows:
  • Rotten Tomatoes CSV audit and readiness checks
  • Multiple join strategies vs IMDb movies
  • Method scoring (coverage, ambiguity, alignment)
  • Selected join strategy + justification
  • Joined RT feature quality diagnostics

Reads  : data/raw/csv/Rotten Tomatoes Movies.csv
         outputs_restart/movies_clean.csv
Writes : outputs_restart/theme_12_rotten_tomatoes_join.html
         outputs_restart/rt_join_method_comparison.csv
         outputs_restart/rt_selected_matches.csv
         outputs_restart/movies_with_rt_features.csv
         outputs_restart/rt_clean.csv
         outputs_restart/rt_cleaning_actions.csv
         outputs_restart/rt_schema_profile_raw.csv
         outputs_restart/rt_schema_profile_clean.csv
         outputs_restart/rt_missingness_raw.csv
         outputs_restart/rt_missingness_clean.csv
         outputs_restart/rt_disguised_missing_audit.csv
         outputs_restart/rt_duplicates_audit.csv
         outputs_restart/rt_numeric_summary.csv
         outputs_restart/rt_engine_benchmark.csv
"""
from __future__ import annotations

import base64
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple
import unicodedata
from difflib import SequenceMatcher

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    import duckdb
    HAS_DUCKDB = True
except Exception:
    HAS_DUCKDB = False

ROOT = Path(__file__).resolve().parents[2]
MEMBER = Path(__file__).resolve().parent
OUT = MEMBER / "outputs_restart"
OUT.mkdir(exist_ok=True)
FIG_DIR = OUT / "figures" / "theme_12"
FIG_DIR.mkdir(parents=True, exist_ok=True)

Y = "#F5C518"; B = "#1848f5"; BG = "#0a0a0a"; CRD = "#141414"
TXT = "#ffffff"; MUT = "#888"; GRN = "#2ecc71"; RED = "#e74c3c"; ORG = "#f39c12"
_FIG_COUNTER = 0

CSS = f"""
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:{BG};color:{TXT};font-family:'Segoe UI',sans-serif;padding:24px;line-height:1.6}}
h1{{color:{Y};font-size:2rem;margin-bottom:6px}}
h2{{color:{Y};font-size:1.3rem;margin:0 0 12px}}
h3{{color:{ORG};font-size:1rem;margin:12px 0 6px}}
.subtitle{{color:{MUT};margin-bottom:28px;font-size:.95rem}}
.card{{background:{CRD};border:1px solid #222;border-radius:12px;padding:20px;margin-bottom:24px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start}}
.kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px}}
.kpi{{background:{CRD};border:1px solid #2a2a2a;border-radius:10px;padding:16px;text-align:center}}
.kpi .val{{font-size:2rem;font-weight:700;color:{Y}}}
.kpi .lbl{{color:{MUT};font-size:.8rem;margin-top:4px}}
table{{width:100%;border-collapse:collapse;font-size:.82rem}}
th{{background:{Y};color:#111;padding:8px 10px;text-align:left}}
td{{padding:7px 10px;border-bottom:1px solid #222}}
tr:hover td{{background:#1e1e1e}}
img{{max-width:100%;border-radius:8px;margin-top:8px}}
.note{{background:#1a1a2e;border-left:4px solid {Y};padding:10px 14px;
       border-radius:4px;font-size:.85rem;margin:10px 0;color:#ccc}}
.code{{background:#111;border:1px solid #333;border-radius:6px;padding:10px 14px;
       font-family:monospace;font-size:.78rem;color:{Y};overflow-x:auto;margin:8px 0;white-space:pre}}
"""

DISGUISED_TOKENS = {"", "\\n", "\\N", "n/a", "na", "null", "none", "unknown", "?"}
RT_NUMERIC_COLS = [
    "runtime_in_minutes", "tomatometer_rating", "tomatometer_count",
    "audience_rating", "audience_count",
]
RT_TEXT_COLS = [
    "movie_title", "movie_info", "critics_consensus", "rating", "genre",
    "directors", "writers", "cast", "studio_name", "tomatometer_status",
]


def _b64(fig):
    global _FIG_COUNTER
    _FIG_COUNTER += 1
    fig.savefig(
        FIG_DIR / f"figure_{_FIG_COUNTER:02d}.png",
        format="png",
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _img(fig):
    return f'<img src="data:image/png;base64,{_b64(fig)}">'


def _card(title, body, color=Y):
    return f'<div class="card"><h2 style="color:{color}">{title}</h2>{body}</div>'


def _kpi(val, lbl):
    return f'<div class="kpi"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>'


def _df_html(df: pd.DataFrame):
    rows = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>" for r in df.values)
    hdr = "".join(f"<th>{c}</th>" for c in df.columns)
    return f"<table><tr>{hdr}</tr>{rows}</table>"


def _page(title, subtitle, kpis, sections):
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{title}</title><style>{CSS}</style></head><body>
<h1>{title}</h1><p class="subtitle">{subtitle}</p>
<div class="kpi-grid">{kpis}</div>{"".join(sections)}</body></html>"""


def _norm_title(x: object) -> str:
    if pd.isna(x):
        return ""
    s = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _prep_movies(movies_raw: pd.DataFrame) -> pd.DataFrame:
    movies = movies_raw.copy()
    movies["startYear"] = pd.to_numeric(movies["startYear"], errors="coerce")
    movies["runtimeMinutes"] = pd.to_numeric(movies["runtimeMinutes"], errors="coerce")
    title_base = movies["primaryTitle"].fillna("")
    if "canonical_title" in movies.columns:
        title_base = movies["canonical_title"].fillna("").where(movies["canonical_title"].fillna("").str.len() > 0, title_base)
    movies["title_key"] = title_base.map(_norm_title)
    return movies


def _schema_profile(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "stage": stage,
            "column": col,
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "unique_non_null": int(s.nunique(dropna=True)),
        })
    return pd.DataFrame(rows)


def _missing_table(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    out = (
        df.isna().mean().mul(100).rename("missing_pct").reset_index()
          .rename(columns={"index": "column"})
    )
    out["stage"] = stage
    return out[["stage", "column", "missing_pct"]].sort_values("missing_pct", ascending=False)


def _scan_disguised_missing(df: pd.DataFrame) -> pd.DataFrame:
    audits = []
    token_set = {t.lower() for t in DISGUISED_TOKENS}
    for col in df.select_dtypes(include=["object", "string"]).columns:
        s = df[col].astype("string").str.strip()
        lowered = s.str.lower()
        for token in sorted(token_set):
            if token == "":
                mask = s.eq("")
            else:
                mask = lowered.eq(token)
            cnt = int(mask.fillna(False).sum())
            if cnt > 0:
                audits.append({"column": col, "token": token if token != "" else "<blank>", "count": cnt})
    if not audits:
        return pd.DataFrame(columns=["column", "token", "count"])
    return pd.DataFrame(audits).sort_values(["count", "column"], ascending=[False, True])


def _duplicate_audit(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    temp = df.copy()
    title_series = temp.get("movie_title", pd.Series(index=temp.index, dtype="string")).astype("string")
    temp["title_key"] = title_series.map(_norm_title)
    if "rt_year" in temp.columns:
        temp["rt_year_tmp"] = temp["rt_year"]
    else:
        temp["rt_year_tmp"] = pd.to_datetime(temp.get("in_theaters_date"), errors="coerce").dt.year
    return pd.DataFrame([{
        "stage": stage,
        "rows": int(len(temp)),
        "exact_duplicate_rows": int(temp.duplicated().sum()),
        "duplicate_title_keys": int(temp["title_key"].duplicated(keep=False).sum()),
        "duplicate_title_year_pairs": int(temp.duplicated(["title_key", "rt_year_tmp"], keep=False).sum()),
    }])


def _numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in RT_NUMERIC_COLS:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        non_na = s.dropna()
        if len(non_na) == 0:
            rows.append({
                "column": col, "count": 0, "mean": np.nan, "std": np.nan, "min": np.nan,
                "p25": np.nan, "p50": np.nan, "p75": np.nan, "max": np.nan,
                "iqr_outlier_pct": np.nan,
            })
            continue
        q1 = non_na.quantile(0.25)
        q3 = non_na.quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        outlier_pct = float(((non_na < low) | (non_na > high)).mean() * 100)
        rows.append({
            "column": col,
            "count": int(non_na.shape[0]),
            "mean": float(non_na.mean()),
            "std": float(non_na.std()),
            "min": float(non_na.min()),
            "p25": float(q1),
            "p50": float(non_na.quantile(0.5)),
            "p75": float(q3),
            "max": float(non_na.max()),
            "iqr_outlier_pct": round(outlier_pct, 2),
        })
    return pd.DataFrame(rows)


def _audit_and_clean_rt(rt_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rt = rt_raw.copy()
    actions = []

    # 1) Trim whitespace in text columns
    trim_changes = 0
    for col in [c for c in RT_TEXT_COLS if c in rt.columns]:
        s = rt[col].astype("string")
        stripped = s.str.strip()
        trim_changes += int(s.ne(stripped).fillna(False).sum())
        rt[col] = stripped
    actions.append({"step": "trim_whitespace_text", "affected_rows": trim_changes, "detail": "Stripped leading/trailing whitespace in text fields"})

    # 2) Replace disguised missing tokens with NaN
    disguised_changes = 0
    token_set = {t.lower() for t in DISGUISED_TOKENS}
    for col in rt.select_dtypes(include=["object", "string"]).columns:
        s = rt[col].astype("string").str.strip()
        lowered = s.str.lower()
        mask = lowered.isin(token_set)
        disguised_changes += int(mask.fillna(False).sum())
        rt.loc[mask, col] = pd.NA
    actions.append({"step": "replace_disguised_missing_tokens", "affected_rows": disguised_changes, "detail": "Replaced disguised null-like tokens with NA"})

    # 3) Parse numeric columns
    parse_failures = 0
    for col in [c for c in RT_NUMERIC_COLS if c in rt.columns]:
        raw_col = rt[col]
        parsed = pd.to_numeric(raw_col.astype("string").str.replace(",", "", regex=False), errors="coerce")
        parse_failures += int((raw_col.notna() & parsed.isna()).sum())
        rt[col] = parsed
    actions.append({"step": "parse_numeric_columns", "affected_rows": parse_failures, "detail": "Values that failed numeric parsing -> NA"})

    # 4) Parse date columns
    date_failures = 0
    for col in ["in_theaters_date", "on_streaming_date"]:
        if col in rt.columns:
            raw_col = rt[col]
            parsed = pd.to_datetime(raw_col, errors="coerce")
            date_failures += int((raw_col.notna() & parsed.isna()).sum())
            rt[col] = parsed
    actions.append({"step": "parse_date_columns", "affected_rows": date_failures, "detail": "Values that failed date parsing -> NA"})

    # 5) Basic domain validation
    invalid_ranges = 0
    for col in ["tomatometer_rating", "audience_rating"]:
        if col in rt.columns:
            mask = rt[col].notna() & ~rt[col].between(0, 100)
            invalid_ranges += int(mask.sum())
            rt.loc[mask, col] = np.nan
    if "runtime_in_minutes" in rt.columns:
        mask = rt["runtime_in_minutes"].notna() & (rt["runtime_in_minutes"] <= 0)
        invalid_ranges += int(mask.sum())
        rt.loc[mask, "runtime_in_minutes"] = np.nan
    for col in ["tomatometer_count", "audience_count"]:
        if col in rt.columns:
            mask = rt[col].notna() & (rt[col] < 0)
            invalid_ranges += int(mask.sum())
            rt.loc[mask, col] = np.nan
    actions.append({"step": "apply_domain_validations", "affected_rows": invalid_ranges, "detail": "Out-of-domain numeric values set to NA"})

    # 6) Drop exact duplicates
    dup_rows = int(rt.duplicated().sum())
    rt = rt.drop_duplicates().reset_index(drop=True)
    actions.append({"step": "drop_exact_duplicates", "affected_rows": dup_rows, "detail": "Removed exact duplicate rows"})

    # 7) Derive normalized and model-ready columns
    rt["rt_id"] = np.arange(len(rt))
    rt["movie_title"] = rt["movie_title"].fillna("")
    rt["title_key"] = rt["movie_title"].map(_norm_title)
    rt["rt_year"] = pd.to_datetime(rt["in_theaters_date"], errors="coerce").dt.year
    rt["runtime_rt"] = pd.to_numeric(rt["runtime_in_minutes"], errors="coerce")
    rt["rt_high"] = np.where(rt["tomatometer_rating"].notna(), (rt["tomatometer_rating"] >= 75).astype(int), np.nan)
    rt["rt_certified_fresh"] = rt["tomatometer_status"].fillna("").str.contains("Certified Fresh", case=False).astype(int)
    actions.append({"step": "derive_join_and_feature_columns", "affected_rows": int(len(rt)), "detail": "Added title_key, rt_year, rt_high, rt_certified_fresh"})

    action_df = pd.DataFrame(actions)
    return rt, action_df


def _prep_rt(rt_raw: pd.DataFrame) -> pd.DataFrame:
    rt = rt_raw.copy()
    rt["rt_id"] = np.arange(len(rt))
    rt["movie_title"] = rt["movie_title"].fillna("")
    rt["title_key"] = rt["movie_title"].map(_norm_title)
    rt["rt_year"] = pd.to_datetime(rt["in_theaters_date"], errors="coerce").dt.year
    rt["runtime_rt"] = pd.to_numeric(rt["runtime_in_minutes"], errors="coerce")
    rt["tomatometer_rating"] = pd.to_numeric(rt["tomatometer_rating"], errors="coerce")
    rt["audience_rating"] = pd.to_numeric(rt["audience_rating"], errors="coerce")
    rt["tomatometer_count"] = pd.to_numeric(rt["tomatometer_count"], errors="coerce")
    rt["audience_count"] = pd.to_numeric(rt["audience_count"], errors="coerce")
    rt["rt_high"] = np.where(rt["tomatometer_rating"].notna(), (rt["tomatometer_rating"] >= 75).astype(int), np.nan)
    rt["rt_certified_fresh"] = rt["tomatometer_status"].fillna("").str.contains("Certified Fresh", case=False).astype(int)
    return rt


def _build_candidates(movies: pd.DataFrame, rt: pd.DataFrame) -> pd.DataFrame:
    rt_cols = [
        "rt_id", "movie_title", "title_key", "rt_year", "genre", "directors", "writers",
        "studio_name", "tomatometer_status", "tomatometer_rating", "tomatometer_count",
        "audience_rating", "audience_count", "runtime_rt", "rt_high", "rt_certified_fresh",
    ]
    left_cols = ["tconst", "primaryTitle", "split", "label", "title_key", "startYear", "runtimeMinutes"]
    cand = movies[left_cols].merge(rt[rt_cols], on="title_key", how="inner")
    cand["year_diff"] = (cand["startYear"] - cand["rt_year"]).abs()
    cand["runtime_diff"] = (cand["runtimeMinutes"] - cand["runtime_rt"]).abs()
    return cand


def _build_fuzzy_candidates(movies: pd.DataFrame, rt: pd.DataFrame, min_ratio: float = 0.85) -> pd.DataFrame:
    # Block by year ±1 to keep matching tractable.
    rt_block = {}
    for _, r in rt[["rt_id", "movie_title", "title_key", "rt_year", "genre", "directors", "writers", "studio_name",
                    "tomatometer_status", "tomatometer_rating", "tomatometer_count", "audience_rating",
                    "audience_count", "runtime_rt", "rt_high", "rt_certified_fresh"]].iterrows():
        y = r["rt_year"]
        if pd.isna(y):
            continue
        rt_block.setdefault(int(y), []).append(r.to_dict())

    rows = []
    for _, m in movies[["tconst", "primaryTitle", "split", "label", "title_key", "startYear", "runtimeMinutes"]].iterrows():
        title = m["title_key"]
        if not isinstance(title, str) or not title:
            continue
        y = m["startYear"]
        if pd.isna(y):
            continue
        year = int(y)
        candidates = []
        for yy in (year - 1, year, year + 1):
            candidates.extend(rt_block.get(yy, []))
        if not candidates:
            continue

        matched = []
        first_char = title[0]
        title_len = len(title)
        for c in candidates:
            cand_title = c["title_key"]
            if not cand_title:
                continue
            # Fast prefilter to avoid expensive SequenceMatcher calls on clearly different titles.
            if cand_title[0] != first_char:
                continue
            if abs(len(cand_title) - title_len) > max(6, int(0.35 * max(title_len, len(cand_title)))):
                continue
            ratio = SequenceMatcher(None, title, c["title_key"]).ratio()
            if ratio >= min_ratio:
                row = {**m.to_dict(), **c}
                row["year_diff"] = abs(float(m["startYear"]) - float(c["rt_year"])) if pd.notna(c["rt_year"]) else np.nan
                row["runtime_diff"] = abs(float(m["runtimeMinutes"]) - float(c["runtime_rt"])) if pd.notna(c["runtime_rt"]) and pd.notna(m["runtimeMinutes"]) else np.nan
                row["title_similarity"] = ratio
                matched.append(row)
        if not matched:
            continue

        for row in matched:
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=[
            "tconst", "primaryTitle", "split", "label", "title_key", "startYear", "runtimeMinutes",
            "rt_id", "movie_title", "rt_year", "genre", "directors", "writers", "studio_name", "tomatometer_status",
            "tomatometer_rating", "tomatometer_count", "audience_rating", "audience_count", "runtime_rt",
            "rt_high", "rt_certified_fresh", "year_diff", "runtime_diff", "title_similarity",
        ])
    return pd.DataFrame(rows)


def _pick_best_candidate(cand: pd.DataFrame) -> pd.DataFrame:
    ranked = cand.copy()
    ranked["year_rank"] = ranked["year_diff"].fillna(99)
    ranked["runtime_rank"] = ranked["runtime_diff"].fillna(999)
    ranked["votes_rank"] = ranked["tomatometer_count"].fillna(0)
    ranked["sim_rank"] = -ranked["title_similarity"].fillna(-1) if "title_similarity" in ranked.columns else 0
    ranked = ranked.sort_values(
        ["tconst", "year_rank", "runtime_rank", "sim_rank", "votes_rank"],
        ascending=[True, True, True, True, False],
    )
    counts = ranked.groupby("tconst").size().rename("candidate_count").reset_index()
    best = ranked.drop_duplicates("tconst", keep="first").merge(counts, on="tconst", how="left")
    return best


def _evaluate_method(name: str, cand: pd.DataFrame, mask: pd.Series, total_movies: int) -> Tuple[dict, pd.DataFrame]:
    subset = cand.loc[mask].copy()
    if subset.empty:
        return {
            "method": name,
            "coverage": 0.0,
            "matched_movies": 0,
            "unique_ratio": 0.0,
            "avg_candidates": 0.0,
            "agreement_label_vs_rt_high": np.nan,
            "auc_label_vs_tomatometer": np.nan,
            "score": 0.0,
        }, subset

    best = _pick_best_candidate(subset)
    matched_movies = int(best["tconst"].nunique())
    coverage = matched_movies / max(total_movies, 1)
    unique_ratio = float((best["candidate_count"] == 1).mean())
    avg_candidates = float(best["candidate_count"].mean())

    agreement = np.nan
    auc_proxy = np.nan
    valid = best["label"].notna() & best["rt_high"].notna()
    if valid.sum() >= 30:
        y_true = best.loc[valid, "label"].astype(int)
        y_rt = best.loc[valid, "rt_high"].astype(int)
        agreement = float((y_true == y_rt).mean())
        score_rt = best.loc[valid, "tomatometer_rating"].astype(float)
        if y_true.nunique() > 1 and score_rt.nunique() > 1:
            auc_proxy = float(roc_auc_score(y_true, score_rt))

    align_component = 0.5 if np.isnan(agreement) else agreement
    score = 0.45 * coverage + 0.35 * unique_ratio + 0.20 * align_component
    metrics = {
        "method": name,
        "coverage": coverage,
        "matched_movies": matched_movies,
        "unique_ratio": unique_ratio,
        "avg_candidates": avg_candidates,
        "agreement_label_vs_rt_high": agreement,
        "auc_label_vs_tomatometer": auc_proxy,
        "score": score,
    }
    return metrics, best


def _fig_method_comparison(method_df: pd.DataFrame) -> str:
    methods = method_df["method"].tolist()
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(CRD)
        ax.tick_params(colors=TXT)
        for sp in ax.spines.values():
            sp.set_color(MUT)

    axes[0].bar(x - 0.18, method_df["coverage"], width=0.35, color=Y, label="coverage")
    axes[0].bar(x + 0.18, method_df["unique_ratio"], width=0.35, color=B, label="unique_ratio")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=20, ha="right", color=TXT)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Join quality components", color=TXT, fontsize=11)
    axes[0].legend(facecolor=CRD, edgecolor=MUT, labelcolor=TXT, fontsize=8)

    bars = axes[1].bar(x, method_df["score"], color=[GRN if v == method_df["score"].max() else ORG for v in method_df["score"]])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=20, ha="right", color=TXT)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Composite method score", color=TXT, fontsize=11)
    for b, v in zip(bars, method_df["score"]):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", color=TXT, fontsize=8)

    fig.tight_layout()
    return _img(fig)


def _engine_benchmark(movies: pd.DataFrame, rt: pd.DataFrame) -> pd.DataFrame:
    # Compare exact title+year join using three styles.
    out_rows = []

    left_cols = ["tconst", "title_key", "startYear"]
    right_cols = ["rt_id", "title_key", "rt_year"]

    def _to_int_year(series: pd.Series) -> pd.Series:
        """Safely coerce year-like values to nullable Int64 without unsafe casts."""
        numeric = pd.to_numeric(series, errors="coerce")
        numeric = numeric.where((numeric % 1 == 0) | numeric.isna(), np.nan)
        return numeric.astype("Int64")

    # Pandas join
    t0 = time.perf_counter()
    m = movies[left_cols].copy()
    m["startYear_i"] = _to_int_year(m["startYear"])
    r = rt[right_cols].copy()
    r["rt_year_i"] = _to_int_year(r["rt_year"])
    p_join = m.merge(r, left_on=["title_key", "startYear_i"], right_on=["title_key", "rt_year_i"], how="inner")
    t1 = time.perf_counter()
    out_rows.append({
        "engine": "pandas",
        "elapsed_sec": t1 - t0,
        "candidate_rows": int(len(p_join)),
        "matched_movies": int(p_join["tconst"].nunique()),
    })

    # DuckDB SQL join
    if HAS_DUCKDB:
        t0 = time.perf_counter()
        con = duckdb.connect()
        con.register("movies_df", m)
        con.register("rt_df", r)
        d_join = con.execute("""
            SELECT m.tconst, r.rt_id
            FROM movies_df m
            JOIN rt_df r
              ON m.title_key = r.title_key
             AND m.startYear_i = r.rt_year_i
        """).fetchdf()
        con.close()
        t1 = time.perf_counter()
        out_rows.append({
            "engine": "duckdb_sql",
            "elapsed_sec": t1 - t0,
            "candidate_rows": int(len(d_join)),
            "matched_movies": int(d_join["tconst"].nunique()),
        })
    else:
        out_rows.append({
            "engine": "duckdb_sql",
            "elapsed_sec": np.nan,
            "candidate_rows": np.nan,
            "matched_movies": np.nan,
        })

    # MapReduce-style key-group join (Python dict bucket)
    t0 = time.perf_counter()
    bucket = {}
    for row in r.itertuples(index=False):
        k = (row.title_key, row.rt_year_i)
        bucket.setdefault(k, []).append(row.rt_id)
    mr_rows = []
    for row in m.itertuples(index=False):
        k = (row.title_key, row.startYear_i)
        ids = bucket.get(k, [])
        if ids:
            for rid in ids:
                mr_rows.append((row.tconst, rid))
    mr_df = pd.DataFrame(mr_rows, columns=["tconst", "rt_id"]) if mr_rows else pd.DataFrame(columns=["tconst", "rt_id"])
    t1 = time.perf_counter()
    out_rows.append({
        "engine": "mapreduce_style",
        "elapsed_sec": t1 - t0,
        "candidate_rows": int(len(mr_df)),
        "matched_movies": int(mr_df["tconst"].nunique()) if not mr_df.empty else 0,
    })

    return pd.DataFrame(out_rows)


def _fig_engine_benchmark(engine_df: pd.DataFrame) -> str:
    plot_df = engine_df.dropna(subset=["elapsed_sec"]).copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(CRD)
        ax.tick_params(colors=TXT)
        for sp in ax.spines.values():
            sp.set_color(MUT)

    axes[0].bar(plot_df["engine"], plot_df["elapsed_sec"], color=[Y, B, ORG][:len(plot_df)])
    axes[0].set_title("Runtime (sec) — exact title+year join", color=TXT, fontsize=11)
    axes[0].set_ylabel("seconds", color=TXT)

    axes[1].bar(plot_df["engine"], plot_df["matched_movies"], color=[Y, B, ORG][:len(plot_df)])
    axes[1].set_title("Matched IMDb movies", color=TXT, fontsize=11)
    axes[1].set_ylabel("count", color=TXT)

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    return _img(fig)


def _fig_candidate_count(best_matches: pd.DataFrame, method_name: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 3.8), facecolor=BG)
    ax.set_facecolor(CRD)
    vc = best_matches["candidate_count"].value_counts().sort_index().head(12)
    ax.bar(vc.index.astype(str), vc.values, color=B, alpha=0.9)
    ax.set_title(f"{method_name} — candidate multiplicity per IMDb title", color=TXT, fontsize=11)
    ax.set_xlabel("Number of candidate RT rows before tie-break", color=TXT)
    ax.set_ylabel("IMDb movies", color=TXT)
    ax.tick_params(colors=TXT)
    for sp in ax.spines.values():
        sp.set_color(MUT)
    ax.grid(axis="y", color=MUT, alpha=0.2)
    fig.tight_layout()
    return _img(fig)


def _fig_rt_missingness(joined: pd.DataFrame) -> str:
    cols = [
        "rt_tomatometer_rating", "rt_audience_rating",
        "rt_tomatometer_count", "rt_audience_count", "rt_genre",
    ]
    miss = joined[cols].isna().mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=BG)
    ax.set_facecolor(CRD)
    ax.bar(miss.index, miss.values, color=ORG, alpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Missing rate", color=TXT)
    ax.set_title("Missingness of joined RT features", color=TXT, fontsize=11)
    ax.tick_params(colors=TXT, rotation=25)
    for sp in ax.spines.values():
        sp.set_color(MUT)
    for i, v in enumerate(miss.values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", color=TXT, fontsize=8)
    fig.tight_layout()
    return _img(fig)


def _fig_rt_vs_label(joined: pd.DataFrame) -> str:
    use = joined[joined["label"].notna() & joined["rt_tomatometer_rating"].notna()].copy()
    fig, ax = plt.subplots(figsize=(6.5, 3.8), facecolor=BG)
    ax.set_facecolor(CRD)
    if len(use) == 0:
        ax.text(0.5, 0.5, "No matched rows with label + tomatometer_rating", ha="center", va="center", color=TXT)
        ax.set_axis_off()
    else:
        vals0 = use.loc[use["label"] == 0, "rt_tomatometer_rating"].to_numpy()
        vals1 = use.loc[use["label"] == 1, "rt_tomatometer_rating"].to_numpy()
        bp = ax.boxplot([vals0, vals1], patch_artist=True, labels=["IMDb label 0", "IMDb label 1"])
        for patch, color in zip(bp["boxes"], [RED, GRN]):
            patch.set_facecolor(color); patch.set_alpha(0.5)
        ax.set_ylabel("RT tomatometer rating", color=TXT)
        ax.set_title("RT rating vs IMDb target (matched train rows)", color=TXT, fontsize=11)
        ax.tick_params(colors=TXT)
        for sp in ax.spines.values():
            sp.set_color(MUT)
    fig.tight_layout()
    return _img(fig)


def _fig_missingness_before_after(miss_raw: pd.DataFrame, miss_clean: pd.DataFrame) -> str:
    raw_map = miss_raw.set_index("column")["missing_pct"]
    clean_map = miss_clean.set_index("column")["missing_pct"]
    cols = raw_map.sort_values(ascending=False).head(10).index.tolist()
    raw_vals = raw_map.reindex(cols).fillna(0).to_numpy()
    clean_vals = clean_map.reindex(cols).fillna(0).to_numpy()

    x = np.arange(len(cols))
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
    ax.set_facecolor(CRD)
    ax.bar(x - 0.2, raw_vals, width=0.38, color=ORG, alpha=0.9, label="Raw")
    ax.bar(x + 0.2, clean_vals, width=0.38, color=B, alpha=0.9, label="Cleaned")
    ax.set_xticks(x)
    ax.set_xticklabels(cols, rotation=35, ha="right", color=TXT)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Missing %", color=TXT)
    ax.set_title("Missingness before vs after RT cleaning (top 10 columns)", color=TXT, fontsize=11)
    ax.legend(facecolor=CRD, edgecolor=MUT, labelcolor=TXT, fontsize=8)
    ax.tick_params(colors=TXT)
    for sp in ax.spines.values():
        sp.set_color(MUT)
    fig.tight_layout()
    return _img(fig)


def _fig_rt_numeric_distributions(rt_clean: pd.DataFrame) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), facecolor=BG)
    plots = [
        ("tomatometer_rating", False, Y),
        ("audience_rating", False, B),
        ("tomatometer_count", True, ORG),
        ("audience_count", True, GRN),
    ]
    for ax, (col, log_scale, color) in zip(axes.flat, plots):
        ax.set_facecolor(CRD)
        vals = pd.to_numeric(rt_clean[col], errors="coerce").dropna()
        if log_scale:
            vals = np.log1p(vals)
        ax.hist(vals, bins=40, color=color, alpha=0.85)
        title = f"log1p({col})" if log_scale else col
        ax.set_title(title, color=TXT, fontsize=10)
        ax.tick_params(colors=TXT, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(MUT)
    fig.suptitle("RT numeric distributions after cleaning", color=TXT, fontsize=12)
    fig.tight_layout()
    return _img(fig)


def _fig_duplicate_title_key(rt_clean: pd.DataFrame) -> str:
    vc = rt_clean["title_key"].value_counts()
    vc = vc[vc > 1]
    fig, ax = plt.subplots(figsize=(6.5, 3.8), facecolor=BG)
    ax.set_facecolor(CRD)
    if vc.empty:
        ax.text(0.5, 0.5, "No duplicate title_key rows after cleaning", ha="center", va="center", color=TXT)
        ax.set_axis_off()
    else:
        b = vc.value_counts().sort_index().head(10)
        ax.bar(b.index.astype(str), b.values, color=RED, alpha=0.85)
        ax.set_xlabel("Rows per title_key", color=TXT)
        ax.set_ylabel("Number of title_keys", color=TXT)
        ax.set_title("RT title-key collision profile", color=TXT, fontsize=11)
        ax.tick_params(colors=TXT)
        for sp in ax.spines.values():
            sp.set_color(MUT)
    fig.tight_layout()
    return _img(fig)


def run(state: dict) -> dict:
    global _FIG_COUNTER
    _FIG_COUNTER = 0
    for p in FIG_DIR.glob("*.png"):
        p.unlink()

    rt_path = ROOT / "data" / "raw" / "csv" / "Rotten Tomatoes Movies.csv"
    if not rt_path.exists():
        raise FileNotFoundError(f"Missing Rotten Tomatoes file: {rt_path}")

    movies = state.get("movies_clean")
    if movies is None:
        movies_path = OUT / "movies_clean.csv"
        if not movies_path.exists():
            raise FileNotFoundError("movies_clean.csv not found in outputs_restart. Run Theme 04 first.")
        movies = pd.read_csv(movies_path)

    rt_raw = pd.read_csv(rt_path)
    movies_p = _prep_movies(movies)

    raw_profile = _schema_profile(rt_raw, "raw")
    raw_missing = _missing_table(rt_raw, "raw")
    disguised_audit = _scan_disguised_missing(rt_raw)
    dup_raw = _duplicate_audit(rt_raw, "raw")

    rt_p, clean_actions = _audit_and_clean_rt(rt_raw)
    clean_profile = _schema_profile(rt_p, "clean")
    clean_missing = _missing_table(rt_p, "clean")
    dup_clean = _duplicate_audit(rt_p, "clean")
    dup_audit = pd.concat([dup_raw, dup_clean], ignore_index=True)
    numeric_summary = _numeric_summary(rt_p)

    candidates = _build_candidates(movies_p, rt_p)
    fuzzy_candidates = _build_fuzzy_candidates(movies_p, rt_p, min_ratio=0.85)
    engine_benchmark = _engine_benchmark(movies_p, rt_p)

    method_masks = {
        "title_exact_year": candidates["year_diff"].eq(0),
        "title_exact_year_runtime15": candidates["year_diff"].eq(0) & (candidates["runtime_diff"].le(15) | candidates["runtime_diff"].isna()),
        "title_year_pm1": candidates["year_diff"].le(1),
        "title_year_pm1_runtime20": candidates["year_diff"].le(1) & (candidates["runtime_diff"].le(20) | candidates["runtime_diff"].isna()),
        "title_year_pm2_runtime20": candidates["year_diff"].le(2) & (candidates["runtime_diff"].le(20) | candidates["runtime_diff"].isna()),
        "title_only": candidates["title_key"].ne(""),
        "title_year_pm1_or_runtime8": candidates["year_diff"].le(1) | (candidates["year_diff"].isna() & candidates["runtime_diff"].le(8)),
    }

    method_rows = []
    method_best: Dict[str, pd.DataFrame] = {}
    total_movies = int(movies_p["tconst"].nunique())
    for name, mask in method_masks.items():
        metrics, best_matches = _evaluate_method(name, candidates, mask, total_movies)
        method_rows.append(metrics)
        method_best[name] = best_matches

    fuzzy_mask_90 = fuzzy_candidates["title_similarity"].ge(0.90) & fuzzy_candidates["year_diff"].le(1) if not fuzzy_candidates.empty else pd.Series([], index=fuzzy_candidates.index, dtype=bool)
    fuzzy_metrics_90, fuzzy_best_90 = _evaluate_method("fuzzy_title90_year_pm1", fuzzy_candidates, fuzzy_mask_90, total_movies)
    method_rows.append(fuzzy_metrics_90)
    method_best["fuzzy_title90_year_pm1"] = fuzzy_best_90

    fuzzy_mask_90_rt = (
        fuzzy_candidates["title_similarity"].ge(0.90)
        & fuzzy_candidates["year_diff"].le(1)
        & (fuzzy_candidates["runtime_diff"].le(20) | fuzzy_candidates["runtime_diff"].isna())
    ) if not fuzzy_candidates.empty else pd.Series([], index=fuzzy_candidates.index, dtype=bool)
    fuzzy_metrics_90_rt, fuzzy_best_90_rt = _evaluate_method("fuzzy_title90_year_pm1_runtime20", fuzzy_candidates, fuzzy_mask_90_rt, total_movies)
    method_rows.append(fuzzy_metrics_90_rt)
    method_best["fuzzy_title90_year_pm1_runtime20"] = fuzzy_best_90_rt

    method_df = pd.DataFrame(method_rows).sort_values("score", ascending=False).reset_index(drop=True)
    best_method = method_df.iloc[0]["method"]
    best_matches = method_best[best_method].copy()

    rt_rename = {
        "movie_title": "rt_movie_title",
        "rt_year": "rt_year",
        "genre": "rt_genre",
        "directors": "rt_directors",
        "writers": "rt_writers",
        "studio_name": "rt_studio_name",
        "tomatometer_status": "rt_tomatometer_status",
        "tomatometer_rating": "rt_tomatometer_rating",
        "tomatometer_count": "rt_tomatometer_count",
        "audience_rating": "rt_audience_rating",
        "audience_count": "rt_audience_count",
        "runtime_rt": "rt_runtime_minutes",
        "rt_high": "rt_high",
        "rt_certified_fresh": "rt_certified_fresh",
    }
    keep_cols = [
        "tconst", "rt_id", "movie_title", "rt_year", "genre", "directors", "writers",
        "studio_name", "tomatometer_status", "tomatometer_rating", "tomatometer_count",
        "audience_rating", "audience_count", "runtime_rt", "rt_high", "rt_certified_fresh",
        "candidate_count", "year_diff", "runtime_diff",
    ]
    best_export = best_matches[keep_cols].rename(columns=rt_rename)

    joined = movies_p.merge(best_export, on="tconst", how="left")
    joined["rt_match_flag"] = joined["rt_id"].notna().astype(int)
    joined["rt_tomatometer_missing"] = joined["rt_tomatometer_rating"].isna().astype(int)
    joined["rt_audience_missing"] = joined["rt_audience_rating"].isna().astype(int)
    joined["rt_score_gap"] = joined["rt_tomatometer_rating"] - joined["rt_audience_rating"]
    joined["rt_score_gap_abs"] = joined["rt_score_gap"].abs()
    joined["rt_popularity_log1p"] = np.log1p(joined["rt_tomatometer_count"].fillna(0))

    rt_feature_cols = [
        "tconst", "rt_match_flag", "rt_tomatometer_rating", "rt_audience_rating",
        "rt_tomatometer_count", "rt_audience_count", "rt_certified_fresh", "rt_high",
        "rt_score_gap", "rt_score_gap_abs", "rt_popularity_log1p",
    ]
    rt_features = joined[rt_feature_cols].copy()

    # Persist artifacts
    rt_p.to_csv(OUT / "rt_clean.csv", index=False)
    clean_actions.to_csv(OUT / "rt_cleaning_actions.csv", index=False)
    raw_profile.to_csv(OUT / "rt_schema_profile_raw.csv", index=False)
    clean_profile.to_csv(OUT / "rt_schema_profile_clean.csv", index=False)
    raw_missing.to_csv(OUT / "rt_missingness_raw.csv", index=False)
    clean_missing.to_csv(OUT / "rt_missingness_clean.csv", index=False)
    disguised_audit.to_csv(OUT / "rt_disguised_missing_audit.csv", index=False)
    dup_audit.to_csv(OUT / "rt_duplicates_audit.csv", index=False)
    numeric_summary.to_csv(OUT / "rt_numeric_summary.csv", index=False)
    engine_benchmark.to_csv(OUT / "rt_engine_benchmark.csv", index=False)
    method_df.to_csv(OUT / "rt_join_method_comparison.csv", index=False)
    best_export.to_csv(OUT / "rt_selected_matches.csv", index=False)
    joined.to_csv(OUT / "movies_with_rt_features.csv", index=False)
    rt_features.to_csv(OUT / "rt_features.csv", index=False)

    method_show = method_df.copy()
    for col in ["coverage", "unique_ratio", "avg_candidates", "agreement_label_vs_rt_high", "auc_label_vs_tomatometer", "score"]:
        method_show[col] = method_show[col].astype(float).round(4)
    selected_metrics = method_show[method_show["method"] == best_method].iloc[0]
    match_rate = float(joined["rt_match_flag"].mean())
    disguised_fixed = int(clean_actions.loc[clean_actions["step"] == "replace_disguised_missing_tokens", "affected_rows"].iloc[0])

    kpis = (
        _kpi(f"{len(rt_raw):,}", "RT raw rows")
        + _kpi(f"{len(rt_p):,}", "RT clean rows")
        + _kpi(f"{disguised_fixed:,}", "Disguised tokens<br>cleaned")
        + _kpi(f"{match_rate:.1%}", "IMDb rows with RT match")
    )

    sections = []
    sections.append(_card("1. Pre-Join RT Data Audit (Raw)", f"""
<div class="grid2">
  <div>
    <div class="note">
      External dataset: <code>{rt_path}</code><br>
      IMDb base for this theme: <code>{OUT / "movies_clean.csv"}</code><br>
      Validation-first scope: schema, missingness, disguised nulls, duplicates, numeric checks.
    </div>
    <h3>Raw schema profile</h3>
    {_df_html(raw_profile[["column", "dtype", "non_null", "missing_pct", "unique_non_null"]])}
  </div>
  <div>
    <h3>Raw missingness (top 10)</h3>
    {_df_html(raw_missing.head(10)[["column", "missing_pct"]])}
    <h3>Disguised missing tokens (top 12)</h3>
    {_df_html(disguised_audit.head(12)) if not disguised_audit.empty else "<div class='note'>No disguised tokens detected.</div>"}
    <h3>Duplicate audit</h3>
    {_df_html(dup_audit)}
  </div>
</div>
"""))

    sections.append(_card("2. Cleaning Pipeline and Impact", f"""
<div class="grid2">
  <div>
    <h3>Cleaning action log</h3>
    {_df_html(clean_actions)}
    <h3>Clean schema profile</h3>
    {_df_html(clean_profile[["column", "dtype", "non_null", "missing_pct", "unique_non_null"]])}
  </div>
  <div>
    {_fig_missingness_before_after(raw_missing, clean_missing)}
    <div class="note">
      Cleaning is executed before join to reduce false candidate matches from token noise and malformed fields.
    </div>
  </div>
</div>
"""))

    sections.append(_card("3. RT Statistics After Cleaning", f"""
<div class="grid2">
  <div>{_fig_rt_numeric_distributions(rt_p)}</div>
  <div>{_fig_duplicate_title_key(rt_p)}</div>
</div>
<h3>Numeric summary (with IQR outlier %)</h3>
{_df_html(numeric_summary.round(3))}
"""))

    method_rules = pd.DataFrame([
        {"method": "title_exact_year", "rule": "normalized_title match AND abs(startYear - rt_year) == 0"},
        {"method": "title_exact_year_runtime15", "rule": "title_exact_year AND |runtime diff| <= 15 (or runtime missing)"},
        {"method": "title_year_pm1", "rule": "normalized_title match AND abs(startYear - rt_year) <= 1"},
        {"method": "title_year_pm1_runtime20", "rule": "title_year_pm1 AND |runtime diff| <= 20 (or runtime missing)"},
        {"method": "title_year_pm2_runtime20", "rule": "normalized_title match AND abs(startYear - rt_year) <= 2 AND runtime constraint"},
        {"method": "title_only", "rule": "normalized_title match only"},
        {"method": "title_year_pm1_or_runtime8", "rule": "title_year_pm1 OR (year missing AND |runtime diff| <= 8)"},
        {"method": "fuzzy_title90_year_pm1", "rule": "SequenceMatcher(title_key) >= 0.90 within year±1 block"},
        {"method": "fuzzy_title90_year_pm1_runtime20", "rule": "fuzzy_title90_year_pm1 AND runtime constraint"},
    ])
    sections.append(_card("4. Join Methods Evaluated", f"""
{_df_html(method_rules)}
<div class="note">
All methods enforce deterministic one-to-one tie-break after candidate generation:
minimum year difference, then minimum runtime difference, then maximum tomatometer_count.
</div>
<div class="note">
Director-name matching is <strong>not possible</strong> with current files because IMDb side has only director IDs (nm...) and no authoritative ID→name table.
To test director-aware joins correctly, add IMDb name mapping tables (for example <code>name.basics.tsv</code> + <code>title.principals.tsv</code>).
</div>
"""))

    sections.append(_card("5. Engine Choice Benchmark (DuckDB vs Pandas vs MapReduce-style)", f"""
{_fig_engine_benchmark(engine_benchmark)}
{_df_html(engine_benchmark.round(6))}
<div class="note">
<strong>DuckDB choice:</strong> selected for SQL clarity and reproducibility on local-scale iterative EDA.
Pandas and mapreduce-style key bucketing are valid alternatives; all three are compared here on the same exact join task.
</div>
"""))

    sections.append(_card("6. Method Comparison and Selection", f"""
{_fig_method_comparison(method_df)}
{_df_html(method_show)}
<div class="note">
Selected method: <strong>{best_method}</strong> with score <strong>{selected_metrics["score"]:.4f}</strong>.
This balances coverage (<strong>{selected_metrics["coverage"]:.2%}</strong>) and ambiguity control
(unique ratio <strong>{selected_metrics["unique_ratio"]:.2%}</strong>) while preserving target alignment.
</div>
"""))

    sections.append(_card("7. Selected-Method Diagnostics", f"""
<div class="grid2">
  <div>{_fig_candidate_count(best_matches, best_method)}</div>
  <div>{_fig_rt_missingness(joined)}</div>
</div>
<div class="note">
High multiplicity indicates ambiguous joins and potential leakage/noise risk.
The selected method keeps ambiguity manageable while avoiding overly aggressive false matches.
</div>
"""))

    sections.append(_card("8. Label Relationship and Feature Utility", f"""
<div class="grid2">
  <div>{_fig_rt_vs_label(joined)}</div>
  <div>
    <h3>Exported RT features</h3>
    <div class="code">{chr(10).join(rt_feature_cols)}</div>
    <h3>Saved RT artifacts</h3>
    <ul style="color:#ccc;margin-left:16px">
      <li><code>rt_clean.csv</code></li>
      <li><code>rt_cleaning_actions.csv</code></li>
      <li><code>rt_schema_profile_raw.csv</code></li>
      <li><code>rt_schema_profile_clean.csv</code></li>
      <li><code>rt_missingness_raw.csv</code></li>
      <li><code>rt_missingness_clean.csv</code></li>
      <li><code>rt_disguised_missing_audit.csv</code></li>
      <li><code>rt_duplicates_audit.csv</code></li>
      <li><code>rt_numeric_summary.csv</code></li>
      <li><code>rt_engine_benchmark.csv</code></li>
      <li><code>rt_join_method_comparison.csv</code></li>
      <li><code>rt_selected_matches.csv</code></li>
      <li><code>movies_with_rt_features.csv</code></li>
      <li><code>rt_features.csv</code></li>
    </ul>
  </div>
</div>
"""))

    sections.append(_card("9. Method Justification", f"""
<div class="note">
<strong>Why this join strategy:</strong> deterministic and auditable rules outperform fuzzy title matching for this project,
because sequel/remake naming collisions can create false joins. We explicitly optimize the trade-off between coverage
and one-to-one reliability.
</div>
<div class="note">
<strong>What to avoid:</strong> title-only joins for final modeling unless you explicitly accept higher ambiguity.
</div>
"""))

    html = _page(
        "Theme 12 — Rotten Tomatoes Join Strategy",
        "pre-join data audit + cleaning · join-method benchmark · deterministic selection",
        kpis, sections,
    )
    (OUT / "theme_12_rotten_tomatoes_join.html").write_text(html, encoding="utf-8")
    print(f"[theme_12] Wrote {OUT}/theme_12_rotten_tomatoes_join.html")

    state["rt_raw"] = rt_raw
    state["rt_clean"] = rt_p
    state["rt_cleaning_actions"] = clean_actions
    state["rt_engine_benchmark"] = engine_benchmark
    state["rt_method_comparison"] = method_df
    state["rt_best_method"] = best_method
    state["rt_features"] = rt_features
    state["movies_with_rt_features"] = joined
    return state


if __name__ == "__main__":
    run({})
    print("[theme_12] Done.")
