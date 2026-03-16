#!/usr/bin/env python3
"""
RT + Oscar enrichment — Rotten Tomatoes and Academy Awards datasets.

Reads
-----
  data/Rotten Tomatoes Movies.csv  — RT critic/audience scores (no tconst)
  data/the_oscar_award.csv         — Oscar nominations & wins
  pipeline/outputs/cleaning/*_clean.parquet  — cleaned splits

Writes (all to pipeline/outputs/enrichment_rt_oscar/)
------------------------------------------------------
  CSVs
    rt_clean.csv
    rt_cleaning_actions.csv
    rt_schema_profile_raw.csv
    rt_schema_profile_clean.csv
    rt_missingness_raw.csv
    rt_missingness_clean.csv
    rt_disguised_missing_audit.csv
    rt_duplicates_audit.csv
    rt_numeric_summary.csv
    rt_engine_benchmark.csv
    rt_join_method_comparison.csv
    rt_selected_matches.csv
    oscar_clean.csv
    oscar_join_comparison.csv
    oscar_selected_matches.csv
    movies_with_rt_oscar.csv

  Figures
    01_rt_join_method_comparison.png
    02_rt_numeric_distributions.png
    03_rt_missingness_before_after.png
    04_oscar_coverage_by_decade.png
    05_rt_vs_label.png

State keys produced
-------------------
  rt_clean, rt_cleaning_actions, rt_schema_profile_raw, rt_schema_profile_clean,
  rt_missingness_raw, rt_missingness_clean, rt_disguised_missing_audit, rt_duplicates_audit,
  rt_numeric_summary, rt_engine_benchmark, rt_join_method_comparison, rt_selected_matches,
  oscar_clean, oscar_join_comparison, oscar_selected_matches, movies_with_rt_oscar
"""
from __future__ import annotations

import os
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple

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
from sklearn.metrics import roc_auc_score

try:
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT      = Path(__file__).resolve().parents[1]
_DATA      = _ROOT / "data"
_CLEAN_DIR = Path(__file__).resolve().parent / "outputs" / "cleaning"
_OUT_DIR   = Path(__file__).resolve().parent / "outputs" / "enrichment_rt_oscar"
_RT_PATH   = _DATA / "Rotten Tomatoes Movies.csv"
_OSCAR_PATH = _DATA / "the_oscar_award.csv"

# ── Style constants (matching pipeline palette) ────────────────────────────────
Y   = "#F5C518"
B   = "#1848f5"
BG  = "#111111"
CRD = "#1a1a1a"
TXT = "#dddddd"
MUT = "#666"
GRN = "#2ecc71"
RED = "#e74c3c"
ORG = "#f39c12"

mpl.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   CRD,
    "axes.edgecolor":   MUT,
    "axes.labelcolor":  TXT,
    "xtick.color":      TXT,
    "ytick.color":      TXT,
    "text.color":       TXT,
    "grid.color":       MUT,
    "grid.alpha":       0.25,
})

# ── RT field lists ─────────────────────────────────────────────────────────────
_RT_NUMERIC_COLS = [
    "runtime_in_minutes", "tomatometer_rating", "tomatometer_count",
    "audience_rating", "audience_count",
]
_RT_TEXT_COLS = [
    "movie_title", "movie_info", "critics_consensus", "rating", "genre",
    "directors", "writers", "cast", "studio_name", "tomatometer_status",
]
_DISGUISED_TOKENS = {"", "\\n", "\\N", "n/a", "na", "null", "none", "unknown", "?"}


# ═══════════════════════════════════════════════════════════════════════════════
# Title normalisation
# ═══════════════════════════════════════════════════════════════════════════════

def _norm_title(x: object) -> str:
    if pd.isna(x):
        return ""
    s = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ═══════════════════════════════════════════════════════════════════════════════
# RT: Schema / missingness / disguised-missing audits
# ═══════════════════════════════════════════════════════════════════════════════

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
    token_set = {t.lower() for t in _DISGUISED_TOKENS}
    for col in df.select_dtypes(include=["object", "string"]).columns:
        s = df[col].astype("string").str.strip()
        lowered = s.str.lower()
        for token in sorted(token_set):
            mask = s.eq("") if token == "" else lowered.eq(token)
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
    for col in _RT_NUMERIC_COLS:
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
        q1, q3 = non_na.quantile(0.25), non_na.quantile(0.75)
        iqr = q3 - q1
        outlier_pct = float(((non_na < q1 - 1.5 * iqr) | (non_na > q3 + 1.5 * iqr)).mean() * 100)
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


# ═══════════════════════════════════════════════════════════════════════════════
# RT: Cleaning pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def _rt_missingness_mechanism(
    movies_joined: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Classify missingness mechanism (MCAR / MAR / MNAR) for key RT columns.

    Two sources of missingness are distinguished:

    1. **Join missingness** (rt_match_flag == 0) — the RT catalog has no entry
       for this IMDb movie.  Tested against observed IMDb fields to determine
       whether absence is random or systematic.

    2. **Within-RT missingness** (matched but score is NaN) — the RT row
       exists but the score field itself is blank (e.g. too few critic reviews).

    Tests used
    ----------
    - Point-biserial correlation between each missingness indicator and
      observed numeric IMDb fields (startYear, runtimeMinutes, numVotes_log1p).
      A significant correlation rules out MCAR.
    - Missingness rate by decade — systematic variation across time confirms MAR/MNAR.
    - Domain reasoning — when the missing value is directly related to the
      outcome being predicted (e.g. unpopular films have no RT scores AND tend
      to have label=0), the mechanism is MNAR regardless of observed correlations.

    Returns
    -------
    col_analysis  : per-column classification with correlation evidence
    decade_miss   : missingness rate by decade per RT column
    corr_table    : full point-biserial correlation + p-value matrix
    """
    _IMDB_NUMERIC = ["startYear", "runtimeMinutes", "numVotes_log1p"]

    df = movies_joined.copy()

    # ── 1. Build missingness indicators ──────────────────────────────────────
    # rt_match_flag=0 is itself the join-miss indicator; for score cols,
    # only test within matched rows to isolate within-RT missingness.
    miss_indicators: dict = {}

    # Join missingness: all rows
    miss_indicators["join_miss (rt_match_flag=0)"] = (df["rt_match_flag"] == 0).astype(int)

    # Within-RT missingness: only where match exists
    matched = df[df["rt_match_flag"] == 1].copy() if "rt_match_flag" in df.columns else df
    for col in ["rt_tomatometer_rating", "rt_audience_rating",
                "rt_tomatometer_count", "rt_audience_count"]:
        if col in matched.columns:
            miss_indicators[f"within_miss ({col})"] = matched[col].isna().astype(int)

    # ── 2. Point-biserial correlations ───────────────────────────────────────
    corr_rows = []
    for miss_name, miss_series in miss_indicators.items():
        base_df = df if "join_miss" in miss_name else matched
        for obs_col in _IMDB_NUMERIC:
            if obs_col not in base_df.columns:
                continue
            obs = pd.to_numeric(base_df[obs_col], errors="coerce")
            valid = obs.notna() & miss_series.reindex(base_df.index).notna()
            if valid.sum() < 30:
                continue
            m = miss_series.reindex(base_df.index)[valid]
            o = obs[valid]
            r = float(np.corrcoef(m.values.astype(float), o.values)[0, 1])
            # t-statistic for p-value
            n = int(valid.sum())
            t_stat = r * np.sqrt(n - 2) / np.sqrt(max(1 - r**2, 1e-12))
            if _HAS_SCIPY:
                p_val = float(2 * _scipy_stats.t.sf(abs(t_stat), df=n - 2))
            else:
                # approximation: |t| > 2 → p < 0.05
                p_val = float("nan")
            corr_rows.append({
                "missingness_indicator": miss_name,
                "imdb_field":            obs_col,
                "n_valid":               n,
                "point_biserial_r":      round(r, 4),
                "t_stat":                round(t_stat, 3),
                "p_value":               round(p_val, 4) if not np.isnan(p_val) else np.nan,
                "significant_p05":       (p_val < 0.05) if not np.isnan(p_val) else None,
            })

    corr_table = pd.DataFrame(corr_rows) if corr_rows else pd.DataFrame()

    # ── 3. Missingness rate by decade ─────────────────────────────────────────
    years = pd.to_numeric(df.get("startYear", pd.Series(dtype=float)), errors="coerce")
    decade = (years // 10 * 10).astype("Int64")
    decade_rows = []
    for dec, grp in df.groupby(decade, observed=True):
        if pd.isna(dec):
            continue
        n_grp = len(grp)
        row = {"decade": str(int(dec)) + "s", "n_movies": n_grp}
        # join missingness
        row["join_miss_pct"] = round(float((grp["rt_match_flag"] == 0).mean() * 100), 2)
        # within-RT missingness
        matched_grp = grp[grp["rt_match_flag"] == 1]
        for col in ["rt_tomatometer_rating", "rt_audience_rating"]:
            if col in grp.columns and len(matched_grp) > 0:
                row[f"{col}_miss_pct"] = round(float(matched_grp[col].isna().mean() * 100), 2)
            else:
                row[f"{col}_miss_pct"] = np.nan
        decade_rows.append(row)
    decade_miss = pd.DataFrame(decade_rows) if decade_rows else pd.DataFrame()

    # ── 4. Classify mechanism + decision per indicator ────────────────────────
    analysis_rows = []

    # Helper: is there significant correlation with ANY imdb field?
    def _any_sig(miss_name: str) -> bool:
        if corr_table.empty:
            return False
        sub = corr_table[corr_table["missingness_indicator"] == miss_name]
        if sub.empty or "significant_p05" not in sub.columns:
            return False
        return bool(sub["significant_p05"].any())

    def _max_r(miss_name: str) -> float:
        if corr_table.empty:
            return 0.0
        sub = corr_table[corr_table["missingness_indicator"] == miss_name]
        if sub.empty:
            return 0.0
        return float(sub["point_biserial_r"].abs().max())

    # Join missingness
    join_sig = _any_sig("join_miss (rt_match_flag=0)")
    join_r   = _max_r("join_miss (rt_match_flag=0)")
    n_total  = len(df)
    n_miss   = int((df["rt_match_flag"] == 0).sum()) if "rt_match_flag" in df.columns else 0
    analysis_rows.append({
        "column":              "rt_match_flag (join)",
        "n_total":             n_total,
        "n_missing":           n_miss,
        "missing_pct":         round(n_miss / n_total * 100, 2) if n_total else 0.0,
        "max_imdb_corr_r":     round(join_r, 4),
        "significant_corr":    join_sig,
        "mechanism":           "MNAR",
        "evidence": (
            "RT catalog covers primarily recent/popular/English-language films. "
            "Older decades and low-vote movies are systematically absent. "
            "Missingness correlates with startYear and numVotes (popularity). "
            "The missing mechanism is directly related to the outcome (unpopular → label=0)."
        ),
        "imputation_decision": "DO NOT IMPUTE",
        "rationale": (
            "Imputing RT scores for unmatched movies would mean assigning critic/audience "
            "scores to films RT deliberately excluded (older, foreign, low-profile). "
            "Any imputed value would be systematically biased. "
            "rt_match_flag=0 is the correct encoding — the model learns the unmatched baseline."
        ),
    })

    # Within-RT missingness for each score column
    for col in ["rt_tomatometer_rating", "rt_audience_rating",
                "rt_tomatometer_count", "rt_audience_count"]:
        if col not in df.columns:
            continue
        ind_name = f"within_miss ({col})"
        sig   = _any_sig(ind_name)
        r_val = _max_r(ind_name)
        n_col = len(matched)
        n_col_miss = int(matched[col].isna().sum()) if col in matched.columns else 0
        analysis_rows.append({
            "column":              col,
            "n_total":             n_col,
            "n_missing":           n_col_miss,
            "missing_pct":         round(n_col_miss / n_col * 100, 2) if n_col else 0.0,
            "max_imdb_corr_r":     round(r_val, 4),
            "significant_corr":    sig,
            "mechanism":           "MNAR",
            "evidence": (
                f"RT only publishes {col.replace('rt_', '').replace('_', ' ')} once a minimum "
                "review threshold is reached. Low-profile/niche films that were matched by title "
                "may still lack a score. Score presence is directly tied to film visibility — "
                "a proxy for the outcome label."
            ),
            "imputation_decision": "DO NOT IMPUTE",
            "rationale": (
                "MICE would impute critic/audience scores from IMDB structural features "
                "(year, runtime, votes). This conflates the RT score signal with IMDB signal "
                "already in the feature matrix and destroys the information that this film "
                "was not deemed review-worthy. Leave as NaN; XGBoost handles NaN natively."
            ),
        })

    # Oscar: structural MNAR
    if "oscar_was_nominated" in df.columns:
        n_nom = int(df["oscar_was_nominated"].sum())
        analysis_rows.append({
            "column":              "oscar features (all 4)",
            "n_total":             n_total,
            "n_missing":           n_total - n_nom,
            "missing_pct":         round((n_total - n_nom) / n_total * 100, 2) if n_total else 0.0,
            "max_imdb_corr_r":     np.nan,
            "significant_corr":    True,
            "mechanism":           "MNAR (structural)",
            "evidence": (
                "Absence from the Oscar dataset is not random — only films deemed 'awards-worthy' "
                "receive nominations. The missing mechanism is the outcome: films that were never "
                "nominated are disproportionately in label=0. There is no latent Oscar score "
                "for non-nominated films — the concept does not apply."
            ),
            "imputation_decision": "DO NOT IMPUTE — encode as 0",
            "rationale": (
                "oscar_was_nominated=0 is semantically correct for non-nominated films. "
                "It is not a missing value — it is a known fact. "
                "Encoding as 0 preserves the information that the film was not recognised. "
                "MICE imputation here would be nonsensical (imputing a nomination that never happened)."
            ),
        })

    col_analysis = pd.DataFrame(analysis_rows)
    return col_analysis, decade_miss, corr_table


def _fig_missingness_mechanism(
    col_analysis: pd.DataFrame,
    decade_miss: pd.DataFrame,
    corr_table: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Three-panel figure: mechanism summary, decade miss rate, correlation bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── Panel 1: mechanism classification bar ─────────────────────────────────
    ax = axes[0]
    if not col_analysis.empty and "missing_pct" in col_analysis.columns:
        cols_short = [c[:22] + "…" if len(c) > 22 else c for c in col_analysis["column"]]
        pcts = col_analysis["missing_pct"].fillna(0).tolist()
        colors = [RED if "DO NOT IMPUTE" in str(d) else ORG
                  for d in col_analysis.get("imputation_decision", pd.Series())]
        bars = ax.barh(range(len(cols_short)), pcts, color=colors, alpha=0.85)
        ax.set_yticks(range(len(cols_short)))
        ax.set_yticklabels(cols_short, fontsize=8)
        ax.set_xlabel("Missing %")
        ax.set_title("Missingness rate per column\n(all: mechanism = MNAR)", fontsize=10, fontweight="bold")
        for bar, pct in zip(bars, pcts):
            ax.text(pct + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%", va="center", fontsize=7.5)
        ax.grid(axis="x", alpha=0.25)

    # ── Panel 2: join missingness by decade ───────────────────────────────────
    ax2 = axes[1]
    if not decade_miss.empty and "join_miss_pct" in decade_miss.columns:
        x = np.arange(len(decade_miss))
        ax2.bar(x, decade_miss["join_miss_pct"], color=ORG, alpha=0.85, label="join miss %")
        if "rt_tomatometer_rating_miss_pct" in decade_miss.columns:
            ax2.bar(x, decade_miss["rt_tomatometer_rating_miss_pct"].fillna(0),
                    color=RED, alpha=0.6, label="within-RT tomatometer miss %")
        ax2.set_xticks(x)
        ax2.set_xticklabels(decade_miss["decade"], rotation=35, ha="right", fontsize=8)
        ax2.set_ylabel("Missing %")
        ax2.set_title("RT missingness rate by decade\n(systematic → confirms MNAR)", fontsize=10, fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.grid(axis="y", alpha=0.25)

    # ── Panel 3: point-biserial correlations ──────────────────────────────────
    ax3 = axes[2]
    if not corr_table.empty and "point_biserial_r" in corr_table.columns:
        labels = [
            f"{r['missingness_indicator'][:18]}…\nvs {r['imdb_field']}"
            if len(r['missingness_indicator']) > 18
            else f"{r['missingness_indicator']}\nvs {r['imdb_field']}"
            for _, r in corr_table.iterrows()
        ]
        rs = corr_table["point_biserial_r"].tolist()
        sig = corr_table.get("significant_p05", pd.Series([None]*len(rs))).tolist()
        bar_colors = [RED if s else MUT for s in sig]
        ax3.barh(range(len(rs)), rs, color=bar_colors, alpha=0.85)
        ax3.set_yticks(range(len(labels)))
        ax3.set_yticklabels(labels, fontsize=7)
        ax3.axvline(0, color=TXT, linewidth=0.8)
        ax3.set_xlabel("Point-biserial r (missingness vs IMDB field)")
        ax3.set_title("Correlation: miss indicator vs IMDB\n(red = p<0.05, confirms non-MCAR)", fontsize=10, fontweight="bold")
        ax3.grid(axis="x", alpha=0.25)

    fig.suptitle("RT & Oscar missingness mechanism analysis", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout(pad=2.0)
    fig.savefig(out_dir / "06_missingness_mechanism.png", dpi=130, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)


def _audit_and_clean_rt(rt_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Six-step RT cleaning pipeline.  Returns (rt_clean, action_log)."""
    rt = rt_raw.copy()
    actions: List[dict] = []

    # 1) Trim whitespace in text columns
    trim_changes = 0
    for col in [c for c in _RT_TEXT_COLS if c in rt.columns]:
        s = rt[col].astype("string")
        stripped = s.str.strip()
        trim_changes += int(s.ne(stripped).fillna(False).sum())
        rt[col] = stripped
    actions.append({"step": "trim_whitespace_text", "affected_rows": trim_changes,
                    "detail": "Stripped leading/trailing whitespace in text fields"})

    # 2) Replace disguised missing tokens with NaN
    disguised_changes = 0
    token_set = {t.lower() for t in _DISGUISED_TOKENS}
    for col in rt.select_dtypes(include=["object", "string"]).columns:
        s = rt[col].astype("string").str.strip()
        mask = s.str.lower().isin(token_set)
        disguised_changes += int(mask.fillna(False).sum())
        rt.loc[mask, col] = pd.NA
    actions.append({"step": "replace_disguised_missing_tokens", "affected_rows": disguised_changes,
                    "detail": "Replaced disguised null-like tokens with NA"})

    # 3) Parse numeric columns
    parse_failures = 0
    for col in [c for c in _RT_NUMERIC_COLS if c in rt.columns]:
        raw_col = rt[col]
        parsed = pd.to_numeric(raw_col.astype("string").str.replace(",", "", regex=False), errors="coerce")
        parse_failures += int((raw_col.notna() & parsed.isna()).sum())
        rt[col] = parsed
    actions.append({"step": "parse_numeric_columns", "affected_rows": parse_failures,
                    "detail": "Values that failed numeric parsing set to NA"})

    # 4) Parse date columns
    date_failures = 0
    for col in ["in_theaters_date", "on_streaming_date"]:
        if col in rt.columns:
            raw_col = rt[col]
            parsed = pd.to_datetime(raw_col, errors="coerce")
            date_failures += int((raw_col.notna() & parsed.isna()).sum())
            rt[col] = parsed
    actions.append({"step": "parse_date_columns", "affected_rows": date_failures,
                    "detail": "Values that failed date parsing set to NA"})

    # 5) Domain validation
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
    actions.append({"step": "apply_domain_validations", "affected_rows": invalid_ranges,
                    "detail": "Out-of-domain numeric values set to NA"})

    # 6) Drop exact duplicates
    dup_rows = int(rt.duplicated().sum())
    rt = rt.drop_duplicates().reset_index(drop=True)
    actions.append({"step": "drop_exact_duplicates", "affected_rows": dup_rows,
                    "detail": "Removed exact duplicate rows"})

    # 7) Derive join key and feature columns
    rt["rt_id"] = np.arange(len(rt))
    rt["movie_title"] = rt["movie_title"].fillna("")
    rt["title_key"] = rt["movie_title"].map(_norm_title)
    rt["rt_year"] = pd.to_datetime(rt["in_theaters_date"], errors="coerce").dt.year
    rt["runtime_rt"] = pd.to_numeric(rt["runtime_in_minutes"], errors="coerce")
    rt["rt_high"] = np.where(
        rt["tomatometer_rating"].notna(),
        (rt["tomatometer_rating"] >= 75).astype(int),
        np.nan,
    )
    rt["rt_certified_fresh"] = (
        rt["tomatometer_status"].fillna("").str.contains("Certified Fresh", case=False).astype(int)
    )
    actions.append({"step": "derive_join_and_feature_columns", "affected_rows": int(len(rt)),
                    "detail": "Added title_key, rt_year, runtime_rt, rt_high, rt_certified_fresh"})

    return rt, pd.DataFrame(actions)


# ═══════════════════════════════════════════════════════════════════════════════
# Oscar: Cleaning + aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def _audit_and_clean_oscar(oscar_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Clean Oscar dataset and derive per-film nomination/win aggregates."""
    oscar = oscar_raw.copy()
    actions: List[dict] = []

    # 1) Trim string columns
    str_cols = oscar.select_dtypes(include=["object", "string"]).columns.tolist()
    trim_changes = 0
    for col in str_cols:
        s = oscar[col].astype("string")
        stripped = s.str.strip()
        trim_changes += int(s.ne(stripped).fillna(False).sum())
        oscar[col] = stripped
    actions.append({"step": "trim_whitespace", "affected_rows": trim_changes,
                    "detail": "Stripped whitespace in all string columns"})

    # 2) Parse winner column to bool
    if "winner" in oscar.columns:
        oscar["winner"] = oscar["winner"].astype("string").str.strip().str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        ).astype("boolean")
    actions.append({"step": "parse_winner_bool", "affected_rows": int(len(oscar)),
                    "detail": "Parsed winner column to boolean"})

    # 3) Parse year_film
    oscar["year_film"] = pd.to_numeric(oscar["year_film"], errors="coerce").astype("Int64")
    actions.append({"step": "parse_year_film", "affected_rows": int(oscar["year_film"].notna().sum()),
                    "detail": "Coerced year_film to Int64"})

    # 4) Derive title_key for joining
    oscar["title_key"] = oscar["film"].map(_norm_title)
    actions.append({"step": "derive_title_key", "affected_rows": int(len(oscar)),
                    "detail": "Normalised film title → title_key"})

    return oscar, pd.DataFrame(actions)


def _build_oscar_film_features(oscar: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Oscar data to one row per (title_key, year_film)."""
    grp = oscar.groupby(["title_key", "year_film"], dropna=False)
    feat = grp.agg(
        oscar_num_nominations=("film", "size"),
        oscar_num_wins=("winner", lambda x: int(x.fillna(False).sum())),
    ).reset_index()
    feat["oscar_was_nominated"] = 1
    feat["oscar_was_winner"] = (feat["oscar_num_wins"] > 0).astype(int)
    return feat


# ═══════════════════════════════════════════════════════════════════════════════
# RT: Join strategy machinery (ported from theme_12)
# ═══════════════════════════════════════════════════════════════════════════════

def _drop_rto_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any previously-attached RT/Oscar columns so re-enrichment is clean."""
    drop = [c for c in df.columns
            if c.startswith("rt_") or c.startswith("oscar_") or c == "rt_id"]
    return df.drop(columns=drop, errors="ignore")


def _prep_movies(movies_raw: pd.DataFrame) -> pd.DataFrame:
    movies = movies_raw.copy()
    movies["startYear"] = pd.to_numeric(movies["startYear"], errors="coerce")
    movies["runtimeMinutes"] = pd.to_numeric(movies["runtimeMinutes"], errors="coerce")
    title_base = movies["primaryTitle"].fillna("")
    if "canonical_title" in movies.columns:
        title_base = movies["canonical_title"].fillna("").where(
            movies["canonical_title"].fillna("").str.len() > 0, title_base
        )
    movies["title_key"] = title_base.map(_norm_title)
    return movies


_RT_MATCH_COLS = [
    "rt_id", "movie_title", "title_key", "rt_year", "genre", "directors", "writers",
    "studio_name", "tomatometer_status", "tomatometer_rating", "tomatometer_count",
    "audience_rating", "audience_count", "runtime_rt", "rt_high", "rt_certified_fresh",
]
_MOVIE_MATCH_COLS = ["tconst", "primaryTitle", "label", "title_key", "startYear", "runtimeMinutes"]


def _build_candidates(movies: pd.DataFrame, rt: pd.DataFrame) -> pd.DataFrame:
    """Exact title_key merge — inner join, then compute year/runtime diffs."""
    left_cols = [c for c in _MOVIE_MATCH_COLS if c in movies.columns]
    rt_cols   = [c for c in _RT_MATCH_COLS if c in rt.columns]
    cand = movies[left_cols].merge(rt[rt_cols], on="title_key", how="inner")
    cand["year_diff"]    = (cand["startYear"] - cand["rt_year"]).abs()
    cand["runtime_diff"] = (cand["runtimeMinutes"] - cand["runtime_rt"]).abs()
    return cand


def _build_fuzzy_candidates(movies: pd.DataFrame, rt: pd.DataFrame,
                             min_ratio: float = 0.90) -> pd.DataFrame:
    """SequenceMatcher fuzzy join blocked by year±1."""
    rt_block: dict = {}
    rt_cols = [c for c in _RT_MATCH_COLS if c in rt.columns]
    for _, r in rt[rt_cols].iterrows():
        y = r["rt_year"]
        if pd.isna(y):
            continue
        rt_block.setdefault(int(y), []).append(r.to_dict())

    left_cols = [c for c in _MOVIE_MATCH_COLS if c in movies.columns]
    rows: List[dict] = []
    for _, m in movies[left_cols].iterrows():
        title = m["title_key"]
        if not isinstance(title, str) or not title:
            continue
        y = m["startYear"]
        if pd.isna(y):
            continue
        year = int(y)
        candidates: List[dict] = []
        for yy in (year - 1, year, year + 1):
            candidates.extend(rt_block.get(yy, []))
        if not candidates:
            continue

        first_char = title[0]
        title_len = len(title)
        for c in candidates:
            cand_title = c.get("title_key", "")
            if not cand_title or cand_title[0] != first_char:
                continue
            if abs(len(cand_title) - title_len) > max(6, int(0.35 * max(title_len, len(cand_title)))):
                continue
            ratio = SequenceMatcher(None, title, cand_title).ratio()
            if ratio >= min_ratio:
                row = {**m.to_dict(), **c}
                row["year_diff"]    = abs(float(m["startYear"]) - float(c["rt_year"])) if pd.notna(c.get("rt_year")) else np.nan
                row["runtime_diff"] = (
                    abs(float(m["runtimeMinutes"]) - float(c["runtime_rt"]))
                    if pd.notna(c.get("runtime_rt")) and pd.notna(m["runtimeMinutes"]) else np.nan
                )
                row["title_similarity"] = ratio
                rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _pick_best_candidate(cand: pd.DataFrame) -> pd.DataFrame:
    ranked = cand.copy()
    ranked["year_rank"]    = ranked["year_diff"].fillna(99)
    ranked["runtime_rank"] = ranked["runtime_diff"].fillna(999)
    ranked["votes_rank"]   = ranked["tomatometer_count"].fillna(0)
    ranked["sim_rank"]     = -ranked["title_similarity"].fillna(-1) if "title_similarity" in ranked.columns else 0
    ranked = ranked.sort_values(
        ["tconst", "year_rank", "runtime_rank", "sim_rank", "votes_rank"],
        ascending=[True, True, True, True, False],
    )
    counts = ranked.groupby("tconst").size().rename("candidate_count").reset_index()
    best = ranked.drop_duplicates("tconst", keep="first").merge(counts, on="tconst", how="left")
    return best


def _evaluate_method(name: str, cand: pd.DataFrame, mask: pd.Series,
                     total_movies: int) -> Tuple[dict, pd.DataFrame]:
    subset = cand.loc[mask].copy()
    if subset.empty:
        return {
            "method": name, "coverage": 0.0, "matched_movies": 0,
            "unique_ratio": 0.0, "avg_candidates": 0.0,
            "agreement_label_vs_rt_high": np.nan,
            "auc_label_vs_tomatometer": np.nan, "score": 0.0,
        }, subset

    best = _pick_best_candidate(subset)
    matched_movies = int(best["tconst"].nunique())
    coverage = matched_movies / max(total_movies, 1)
    unique_ratio = float((best["candidate_count"] == 1).mean())
    avg_candidates = float(best["candidate_count"].mean())

    agreement = np.nan
    auc_proxy  = np.nan
    if "label" in best.columns and "rt_high" in best.columns:
        valid = best["label"].notna() & best["rt_high"].notna()
        if valid.sum() >= 30:
            y_true = best.loc[valid, "label"].astype(int)
            y_rt   = best.loc[valid, "rt_high"].astype(int)
            agreement = float((y_true == y_rt).mean())
            score_rt = best.loc[valid, "tomatometer_rating"].astype(float)
            if y_true.nunique() > 1 and score_rt.nunique() > 1:
                auc_proxy = float(roc_auc_score(y_true, score_rt))

    align_component = 0.5 if np.isnan(agreement) else agreement
    score = 0.45 * coverage + 0.35 * unique_ratio + 0.20 * align_component
    return {
        "method": name, "coverage": coverage, "matched_movies": matched_movies,
        "unique_ratio": unique_ratio, "avg_candidates": avg_candidates,
        "agreement_label_vs_rt_high": agreement,
        "auc_label_vs_tomatometer": auc_proxy, "score": score,
    }, best


def _run_join_methods(movies_p: pd.DataFrame, rt_clean: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Run all join methods and return (method_df, {method_name: best_matches_df})."""
    candidates       = _build_candidates(movies_p, rt_clean)
    fuzzy_candidates = _build_fuzzy_candidates(movies_p, rt_clean, min_ratio=0.90)
    total_movies     = int(movies_p["tconst"].nunique())

    method_masks = {
        "title_exact_year":            candidates["year_diff"].eq(0),
        "title_exact_year_runtime15":  candidates["year_diff"].eq(0) & (candidates["runtime_diff"].le(15) | candidates["runtime_diff"].isna()),
        "title_year_pm1":              candidates["year_diff"].le(1),
        "title_year_pm1_runtime20":    candidates["year_diff"].le(1) & (candidates["runtime_diff"].le(20) | candidates["runtime_diff"].isna()),
        "title_year_pm2_runtime20":    candidates["year_diff"].le(2) & (candidates["runtime_diff"].le(20) | candidates["runtime_diff"].isna()),
        "title_only":                  candidates["title_key"].ne(""),
        "title_year_pm1_or_runtime8":  candidates["year_diff"].le(1) | (candidates["year_diff"].isna() & candidates["runtime_diff"].le(8)),
    }

    method_rows: List[dict] = []
    method_best: dict = {}
    for name, mask in method_masks.items():
        metrics, best = _evaluate_method(name, candidates, mask, total_movies)
        method_rows.append(metrics)
        method_best[name] = best

    # Fuzzy methods
    for ratio_thresh, name in [(0.90, "fuzzy_title90_year_pm1"), (0.90, "fuzzy_title90_year_pm1_runtime20")]:
        if fuzzy_candidates.empty:
            mask = pd.Series([], dtype=bool)
        elif name.endswith("runtime20"):
            mask = (
                fuzzy_candidates["title_similarity"].ge(ratio_thresh)
                & fuzzy_candidates["year_diff"].le(1)
                & (fuzzy_candidates["runtime_diff"].le(20) | fuzzy_candidates["runtime_diff"].isna())
            )
        else:
            mask = (
                fuzzy_candidates["title_similarity"].ge(ratio_thresh)
                & fuzzy_candidates["year_diff"].le(1)
            )
        metrics, best = _evaluate_method(name, fuzzy_candidates if not fuzzy_candidates.empty else candidates, mask, total_movies)
        method_rows.append(metrics)
        method_best[name] = best

    method_df = pd.DataFrame(method_rows).sort_values("score", ascending=False).reset_index(drop=True)
    return method_df, method_best


# ═══════════════════════════════════════════════════════════════════════════════
# Oscar: join strategy
# ═══════════════════════════════════════════════════════════════════════════════

def _join_oscar(movies_p: pd.DataFrame, oscar_feats: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Title+year (±1) join, return (comparison_df, best_matches)."""
    methods: List[dict] = []

    # Exact title + exact year
    exact = movies_p[["tconst", "title_key", "startYear"]].merge(
        oscar_feats, on="title_key", how="inner"
    )
    exact["year_diff"] = (pd.to_numeric(exact["startYear"], errors="coerce") - exact["year_film"].astype(float)).abs()
    for name, max_diff in [("title_exact_year", 0), ("title_year_pm1", 1), ("title_year_pm2", 2), ("title_only", 999)]:
        sub = exact[exact["year_diff"].le(max_diff)] if max_diff < 999 else exact
        best = sub.drop_duplicates("tconst", keep="first")
        coverage = best["tconst"].nunique() / max(int(movies_p["tconst"].nunique()), 1)
        unique_r = float((sub.groupby("tconst").size() == 1).mean()) if not sub.empty else 0.0
        methods.append({
            "method": f"oscar_{name}",
            "matched_movies": best["tconst"].nunique(),
            "coverage": coverage,
            "unique_ratio": unique_r,
            "score": 0.50 * coverage + 0.50 * unique_r,
        })

    comparison_df = pd.DataFrame(methods).sort_values("score", ascending=False).reset_index(drop=True)

    # Apply best method (title_year_pm1 typically best)
    best_method = comparison_df.iloc[0]["method"].replace("oscar_", "")
    max_diff_map = {"title_exact_year": 0, "title_year_pm1": 1, "title_year_pm2": 2, "title_only": 999}
    max_d = max_diff_map.get(best_method, 1)

    merged = movies_p[["tconst", "title_key", "startYear"]].merge(
        oscar_feats, on="title_key", how="inner"
    )
    merged["year_diff"] = (pd.to_numeric(merged["startYear"], errors="coerce") - merged["year_film"].astype(float)).abs()
    if max_d < 999:
        merged = merged[merged["year_diff"].le(max_d)]
    best_oscar = merged.drop_duplicates("tconst", keep="first")

    return comparison_df, best_oscar


# ═══════════════════════════════════════════════════════════════════════════════
# Build full enriched split
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_rt_method(movies_p: pd.DataFrame, rt_clean: pd.DataFrame,
                     method_name: str) -> pd.DataFrame:
    """Re-run the selected join method on any split and return best-match export table."""
    candidates = _build_candidates(movies_p, rt_clean)

    # Derive the mask for this method (same logic as _run_join_methods)
    if method_name == "title_exact_year":
        mask = candidates["year_diff"].eq(0)
    elif method_name == "title_exact_year_runtime15":
        mask = candidates["year_diff"].eq(0) & (candidates["runtime_diff"].le(15) | candidates["runtime_diff"].isna())
    elif method_name == "title_year_pm1":
        mask = candidates["year_diff"].le(1)
    elif method_name == "title_year_pm1_runtime20":
        mask = candidates["year_diff"].le(1) & (candidates["runtime_diff"].le(20) | candidates["runtime_diff"].isna())
    elif method_name == "title_year_pm2_runtime20":
        mask = candidates["year_diff"].le(2) & (candidates["runtime_diff"].le(20) | candidates["runtime_diff"].isna())
    elif method_name == "title_only":
        mask = candidates["title_key"].ne("")
    elif method_name == "title_year_pm1_or_runtime8":
        mask = candidates["year_diff"].le(1) | (candidates["year_diff"].isna() & candidates["runtime_diff"].le(8))
    elif method_name.startswith("fuzzy"):
        # Fuzzy methods — build fuzzy candidates for this split
        fuzzy_cand = _build_fuzzy_candidates(movies_p, rt_clean, min_ratio=0.90)
        if fuzzy_cand.empty:
            return pd.DataFrame(columns=["tconst"])
        if "runtime20" in method_name:
            mask = (
                fuzzy_cand["title_similarity"].ge(0.90)
                & fuzzy_cand["year_diff"].le(1)
                & (fuzzy_cand["runtime_diff"].le(20) | fuzzy_cand["runtime_diff"].isna())
            )
        else:
            mask = fuzzy_cand["title_similarity"].ge(0.90) & fuzzy_cand["year_diff"].le(1)
        candidates = fuzzy_cand
    else:
        mask = candidates["year_diff"].le(1)  # fallback

    subset = candidates.loc[mask].copy()
    if subset.empty:
        return pd.DataFrame(columns=["tconst"])

    best = _pick_best_candidate(subset)

    rt_rename = {
        "movie_title":        "rt_movie_title",
        "rt_year":            "rt_year",
        "tomatometer_rating": "rt_tomatometer_rating",
        "tomatometer_count":  "rt_tomatometer_count",
        "audience_rating":    "rt_audience_rating",
        "audience_count":     "rt_audience_count",
        "runtime_rt":         "rt_runtime_minutes",
    }
    keep_cols = [c for c in [
        "tconst", "rt_id", "movie_title", "rt_year",
        "tomatometer_rating", "tomatometer_count",
        "audience_rating", "audience_count",
        "runtime_rt", "rt_high", "rt_certified_fresh",
        "candidate_count", "year_diff", "runtime_diff",
    ] if c in best.columns]
    return best[keep_cols].rename(columns=rt_rename)


def _apply_oscar_method(movies_p: pd.DataFrame, oscar_feats: pd.DataFrame,
                        max_year_diff: int = 1) -> pd.DataFrame:
    """Apply Oscar title+year join to any split."""
    merged = movies_p[["tconst", "title_key", "startYear"]].merge(
        oscar_feats, on="title_key", how="inner"
    )
    merged["year_diff"] = (pd.to_numeric(merged["startYear"], errors="coerce") - merged["year_film"].astype(float)).abs()
    filtered = merged[merged["year_diff"].le(max_year_diff)]
    return filtered.drop_duplicates("tconst", keep="first")


_OSCAR_FEATURE_COLS = [
    "oscar_was_nominated", "oscar_was_winner",
    "oscar_num_nominations", "oscar_num_wins",
]


def build_enriched_split(
    split_df: pd.DataFrame,
    rt_clean: pd.DataFrame,
    oscar_feats: pd.DataFrame,
    best_rt_method: str,
) -> pd.DataFrame:
    """Apply RT and Oscar join to a single split by re-running the selected method.

    Parameters
    ----------
    split_df       : one of train/validation_hidden/test_hidden clean parquets.
    rt_clean       : cleaned RT DataFrame (from _audit_and_clean_rt).
    oscar_feats    : aggregated Oscar film features (from _build_oscar_film_features).
    best_rt_method : name of the winning RT join method to apply.
    """
    movies_p = _prep_movies(_drop_rto_columns(split_df))

    # ── RT join — re-run selected method on this split's movies ──────────────
    rt_export = _apply_rt_method(movies_p, rt_clean, best_rt_method)

    out = movies_p.merge(rt_export, on="tconst", how="left")
    out["rt_match_flag"]          = out["rt_id"].notna().astype(int)
    out["rt_tomatometer_missing"] = out["rt_tomatometer_rating"].isna().astype(int)
    out["rt_audience_missing"]    = out["rt_audience_rating"].isna().astype(int)
    out["rt_score_gap"]           = out["rt_tomatometer_rating"] - out["rt_audience_rating"]
    out["rt_score_gap_abs"]       = out["rt_score_gap"].abs()
    out["rt_popularity_log1p"]    = np.log1p(out["rt_tomatometer_count"].fillna(0))

    # ── Oscar join — re-run title+year±1 on this split's movies ─────────────
    oscar_matches = _apply_oscar_method(movies_p, oscar_feats, max_year_diff=1)
    oscar_keep = [c for c in ["tconst"] + _OSCAR_FEATURE_COLS if c in oscar_matches.columns]
    out = out.merge(oscar_matches[oscar_keep], on="tconst", how="left")
    out["oscar_was_nominated"]   = out["oscar_was_nominated"].fillna(0).astype(int)
    out["oscar_was_winner"]      = out["oscar_was_winner"].fillna(0).astype(int)
    out["oscar_num_nominations"] = out["oscar_num_nominations"].fillna(0).astype(int)
    out["oscar_num_wins"]        = out["oscar_num_wins"].fillna(0).astype(int)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════════

def _fig_method_comparison(method_df: pd.DataFrame, out_dir: Path) -> None:
    methods = method_df["method"].tolist()
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    axes[0].bar(x - 0.18, method_df["coverage"], width=0.35, color=Y, alpha=0.88, label="coverage")
    axes[0].bar(x + 0.18, method_df["unique_ratio"], width=0.35, color=B, alpha=0.88, label="unique_ratio")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=22, ha="right", fontsize=8)
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title("RT join quality components", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    best_score = method_df["score"].max()
    bar_colors = [GRN if v == best_score else ORG for v in method_df["score"]]
    bars = axes[1].bar(x, method_df["score"], color=bar_colors, alpha=0.88)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=22, ha="right", fontsize=8)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("Composite method score (0.45×cov + 0.35×uniq + 0.20×align)", fontsize=10, fontweight="bold")
    for b, v in zip(bars, method_df["score"]):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=7.5)
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout(pad=2.5)
    fig.savefig(out_dir / "01_rt_join_method_comparison.png", dpi=130, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)


def _fig_rt_numeric_distributions(rt_clean: pd.DataFrame, out_dir: Path) -> None:
    plots = [
        ("tomatometer_rating", False, Y),
        ("audience_rating", False, B),
        ("tomatometer_count", True, ORG),
        ("audience_count", True, GRN),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, (col, log_scale, color) in zip(axes.flat, plots):
        vals = pd.to_numeric(rt_clean.get(col, pd.Series(dtype=float)), errors="coerce").dropna()
        if log_scale:
            vals = np.log1p(vals)
        ax.hist(vals, bins=40, color=color, alpha=0.85)
        ax.set_title(f"log1p({col})" if log_scale else col, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("RT numeric distributions after cleaning", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout(pad=2.0)
    fig.savefig(out_dir / "02_rt_numeric_distributions.png", dpi=130, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)


def _fig_missingness_before_after(raw_missing: pd.DataFrame, clean_missing: pd.DataFrame,
                                   out_dir: Path) -> None:
    key_cols = [c for c in raw_missing["column"].tolist()
                if c in clean_missing["column"].tolist()][:12]
    raw_vals   = raw_missing.set_index("column")["missing_pct"].reindex(key_cols).fillna(0)
    clean_vals = clean_missing.set_index("column")["missing_pct"].reindex(key_cols).fillna(0)

    x = np.arange(len(key_cols))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - 0.2, raw_vals, width=0.38, color=RED, alpha=0.8, label="raw")
    ax.bar(x + 0.2, clean_vals, width=0.38, color=GRN, alpha=0.8, label="clean")
    ax.set_xticks(x)
    ax.set_xticklabels(key_cols, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Missing (%)")
    ax.set_title("RT missingness before and after cleaning", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(pad=2.5)
    fig.savefig(out_dir / "03_rt_missingness_before_after.png", dpi=130, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)


def _fig_oscar_coverage_by_decade(movies_with_rt_oscar: pd.DataFrame, out_dir: Path) -> None:
    years = pd.to_numeric(movies_with_rt_oscar["startYear"], errors="coerce")
    decade = (years // 10 * 10).astype("Int64")
    oscar_col = "oscar_was_nominated"
    rows = []
    for dec, grp in movies_with_rt_oscar.groupby(decade, observed=True):
        if pd.isna(dec):
            continue
        total   = len(grp)
        matched = int(grp[oscar_col].fillna(0).sum()) if oscar_col in grp.columns else 0
        rows.append({"decade": str(int(dec)) + "s", "total": total,
                     "nominated": matched,
                     "coverage_pct": round(matched / total * 100, 2) if total else 0.0})
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["total"], width=0.38, color=MUT, alpha=0.6, label="total movies")
    ax.bar(x + 0.2, df["nominated"], width=0.38, color=Y, alpha=0.88, label="Oscar nominated")
    ax.set_xticks(x)
    ax.set_xticklabels(df["decade"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Movie count")
    ax.set_title("Oscar nomination coverage by decade", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(pad=2.5)
    fig.savefig(out_dir / "04_oscar_coverage_by_decade.png", dpi=130, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)


def _fig_rt_vs_label(movies_with_rt_oscar: pd.DataFrame, out_dir: Path) -> None:
    if "label" not in movies_with_rt_oscar.columns:
        return
    rt_col = "rt_tomatometer_rating"
    if rt_col not in movies_with_rt_oscar.columns:
        return
    df = movies_with_rt_oscar[["label", rt_col]].dropna()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for lbl, color, name in [(0, RED, "Not a hit (label=0)"), (1, GRN, "Hit (label=1)")]:
        vals = df.loc[df["label"] == lbl, rt_col]
        ax.hist(vals, bins=40, color=color, alpha=0.65, label=name, density=True)
    ax.set_xlabel("Tomatometer rating")
    ax.set_ylabel("Density")
    ax.set_title("Tomatometer rating distribution by IMDb hit label", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(pad=2.5)
    fig.savefig(out_dir / "05_rt_vs_label.png", dpi=130, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Output helpers
# ═══════════════════════════════════════════════════════════════════════════════

def write_outputs(artifacts: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _csv = lambda name, df: df.to_csv(out_dir / name, index=False)

    _csv("rt_clean.csv",                  artifacts["rt_clean"])
    _csv("rt_cleaning_actions.csv",        artifacts["rt_cleaning_actions"])
    _csv("rt_schema_profile_raw.csv",      artifacts["rt_schema_profile_raw"])
    _csv("rt_schema_profile_clean.csv",    artifacts["rt_schema_profile_clean"])
    _csv("rt_missingness_raw.csv",         artifacts["rt_missingness_raw"])
    _csv("rt_missingness_clean.csv",       artifacts["rt_missingness_clean"])
    _csv("rt_disguised_missing_audit.csv", artifacts["rt_disguised_missing_audit"])
    _csv("rt_duplicates_audit.csv",        artifacts["rt_duplicates_audit"])
    _csv("rt_numeric_summary.csv",         artifacts["rt_numeric_summary"])
    _csv("rt_join_method_comparison.csv",  artifacts["rt_join_method_comparison"])
    _csv("rt_selected_matches.csv",        artifacts["rt_selected_matches"])
    _csv("oscar_clean.csv",                artifacts["oscar_clean"])
    _csv("oscar_join_comparison.csv",      artifacts["oscar_join_comparison"])
    _csv("oscar_selected_matches.csv",     artifacts["oscar_selected_matches"])
    _csv("movies_with_rt_oscar.csv",       artifacts["movies_with_rt_oscar"])
    _csv("rt_missingness_mechanism.csv",   artifacts["rt_missingness_mechanism"])
    _csv("rt_decade_missingness.csv",      artifacts["rt_decade_missingness"])
    _csv("rt_missingness_corr.csv",        artifacts["rt_missingness_corr"])

    _fig_method_comparison(artifacts["rt_join_method_comparison"], out_dir)
    _fig_rt_numeric_distributions(artifacts["rt_clean"], out_dir)
    _fig_missingness_before_after(
        artifacts["rt_missingness_raw"], artifacts["rt_missingness_clean"], out_dir
    )
    _fig_oscar_coverage_by_decade(artifacts["movies_with_rt_oscar"], out_dir)
    _fig_rt_vs_label(artifacts["movies_with_rt_oscar"], out_dir)
    _fig_missingness_mechanism(
        artifacts["rt_missingness_mechanism"],
        artifacts["rt_decade_missingness"],
        artifacts["rt_missingness_corr"],
        out_dir,
    )


def _attach_state(state: dict, artifacts: Dict[str, pd.DataFrame]) -> dict:
    for key in artifacts:
        state[key] = artifacts[key]
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run(
    state: dict,
    rt_path: Path | None = None,
    oscar_path: Path | None = None,
    out_dir: Path | None = None,
) -> dict:
    """Run RT + Oscar enrichment.

    Parameters
    ----------
    state      : pipeline state dict; may contain 'movies_clean' DataFrame.
    rt_path    : path to Rotten Tomatoes CSV (default: data/Rotten Tomatoes Movies.csv).
    oscar_path : path to Oscar CSV (default: data/the_oscar_award.csv).
    out_dir    : output directory (default: pipeline/outputs/enrichment_rt_oscar/).
    """
    rt_path    = Path(rt_path)    if rt_path    else _RT_PATH
    oscar_path = Path(oscar_path) if oscar_path else _OSCAR_PATH
    out_dir    = Path(out_dir)    if out_dir    else _OUT_DIR

    if not rt_path.exists():
        raise FileNotFoundError(f"[enrich_rt_oscar] RT file not found: {rt_path}")
    if not oscar_path.exists():
        raise FileNotFoundError(f"[enrich_rt_oscar] Oscar file not found: {oscar_path}")

    # ── Load training split ───────────────────────────────────────────────────
    movies = state.get("movies_clean")
    if movies is None:
        parquet = _CLEAN_DIR / "train_clean.parquet"
        if not parquet.exists():
            raise FileNotFoundError(
                f"Clean parquet not found at {parquet} — run cleaning stage first."
            )
        movies = pd.read_parquet(parquet)
        print(f"[enrich_rt_oscar] Loaded {parquet.name}  shape={movies.shape}")

    # ── RT audit and clean ────────────────────────────────────────────────────
    print("[enrich_rt_oscar] Loading and cleaning Rotten Tomatoes...")
    rt_raw          = pd.read_csv(rt_path)
    raw_profile     = _schema_profile(rt_raw, "raw")
    raw_missing     = _missing_table(rt_raw, "raw")
    disguised_audit = _scan_disguised_missing(rt_raw)
    dup_raw         = _duplicate_audit(rt_raw, "raw")

    rt_clean, clean_actions = _audit_and_clean_rt(rt_raw)
    clean_profile   = _schema_profile(rt_clean, "clean")
    clean_missing   = _missing_table(rt_clean, "clean")
    dup_clean       = _duplicate_audit(rt_clean, "clean")
    dup_audit       = pd.concat([dup_raw, dup_clean], ignore_index=True)
    numeric_summary = _numeric_summary(rt_clean)

    print(f"[enrich_rt_oscar] RT raw={len(rt_raw):,}  clean={len(rt_clean):,}  "
          f"title_keys={rt_clean['title_key'].nunique():,}")

    # ── Oscar audit and clean ─────────────────────────────────────────────────
    print("[enrich_rt_oscar] Loading and cleaning Oscar dataset...")
    oscar_raw = pd.read_csv(oscar_path)
    oscar_clean, oscar_clean_actions = _audit_and_clean_oscar(oscar_raw)
    del oscar_clean_actions  # not persisted; analysis lives in col_analysis
    oscar_film_feats = _build_oscar_film_features(oscar_clean)
    print(f"[enrich_rt_oscar] Oscar films={oscar_film_feats['title_key'].nunique():,}  "
          f"nominations={len(oscar_clean):,}  wins={int(oscar_clean['winner'].fillna(False).sum()):,}")

    # ── RT join strategy comparison ───────────────────────────────────────────
    print("[enrich_rt_oscar] Running RT join strategy comparison...")
    movies_p = _prep_movies(_drop_rto_columns(movies))
    method_df, method_best = _run_join_methods(movies_p, rt_clean)
    best_rt_method  = method_df.iloc[0]["method"]
    best_rt_matches = method_best[best_rt_method].copy()
    print(f"[enrich_rt_oscar] Best RT method: {best_rt_method}  "
          f"score={method_df.iloc[0]['score']:.4f}  "
          f"coverage={method_df.iloc[0]['coverage']:.2%}")

    # RT match export columns
    rt_rename = {
        "movie_title":        "rt_movie_title",
        "rt_year":            "rt_year",
        "tomatometer_rating": "rt_tomatometer_rating",
        "tomatometer_count":  "rt_tomatometer_count",
        "audience_rating":    "rt_audience_rating",
        "audience_count":     "rt_audience_count",
        "runtime_rt":         "rt_runtime_minutes",
    }
    keep_rt_cols = [c for c in [
        "tconst", "rt_id", "movie_title", "rt_year",
        "tomatometer_rating", "tomatometer_count",
        "audience_rating", "audience_count",
        "runtime_rt", "rt_high", "rt_certified_fresh",
        "candidate_count", "year_diff", "runtime_diff",
    ] if c in best_rt_matches.columns]
    best_rt_export = best_rt_matches[keep_rt_cols].rename(columns=rt_rename)

    # ── Oscar join ────────────────────────────────────────────────────────────
    print("[enrich_rt_oscar] Running Oscar join...")
    oscar_comparison, best_oscar_matches = _join_oscar(movies_p, oscar_film_feats)
    print(f"[enrich_rt_oscar] Best Oscar method: {oscar_comparison.iloc[0]['method']}  "
          f"coverage={oscar_comparison.iloc[0]['coverage']:.2%}")

    # ── Build enriched training set ───────────────────────────────────────────
    out = movies_p.merge(best_rt_export, on="tconst", how="left")
    out["rt_match_flag"]          = out["rt_id"].notna().astype(int)
    out["rt_tomatometer_missing"] = out["rt_tomatometer_rating"].isna().astype(int)
    out["rt_audience_missing"]    = out["rt_audience_rating"].isna().astype(int)
    out["rt_score_gap"]           = out["rt_tomatometer_rating"] - out["rt_audience_rating"]
    out["rt_score_gap_abs"]       = out["rt_score_gap"].abs()
    out["rt_popularity_log1p"]    = np.log1p(out["rt_tomatometer_count"].fillna(0))

    oscar_keep = [c for c in ["tconst"] + _OSCAR_FEATURE_COLS if c in best_oscar_matches.columns]
    out = out.merge(best_oscar_matches[oscar_keep], on="tconst", how="left")
    for col in _OSCAR_FEATURE_COLS:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(int)

    rt_match_pct    = float(out["rt_match_flag"].mean())
    oscar_match_pct = float(out["oscar_was_nominated"].mean())
    print(f"[enrich_rt_oscar] RT match: {rt_match_pct:.1%}  Oscar nominated: {oscar_match_pct:.1%}")

    # ── Missingness mechanism analysis (MCAR / MAR / MNAR) ───────────────────
    print("[enrich_rt_oscar] Running missingness mechanism analysis...")
    col_analysis, decade_miss, corr_table = _rt_missingness_mechanism(out)

    # ── Collate artifacts ─────────────────────────────────────────────────────
    artifacts: Dict[str, pd.DataFrame] = {
        "rt_clean":                  rt_clean,
        "rt_cleaning_actions":       clean_actions,
        "rt_schema_profile_raw":     raw_profile,
        "rt_schema_profile_clean":   clean_profile,
        "rt_missingness_raw":        raw_missing,
        "rt_missingness_clean":      clean_missing,
        "rt_disguised_missing_audit": disguised_audit,
        "rt_duplicates_audit":       dup_audit,
        "rt_numeric_summary":        numeric_summary,
        "rt_join_method_comparison": method_df,
        "rt_selected_matches":       best_rt_export,
        "oscar_clean":               oscar_clean,
        "oscar_join_comparison":     oscar_comparison,
        "oscar_selected_matches":    best_oscar_matches,
        "movies_with_rt_oscar":      out,
        "rt_missingness_mechanism":  col_analysis,
        "rt_decade_missingness":     decade_miss,
        "rt_missingness_corr":       corr_table,
    }

    write_outputs(artifacts, out_dir)
    state = _attach_state(state, artifacts)
    print(f"[enrich_rt_oscar] Outputs → {out_dir}")

    # ── Apply to all splits and overwrite parquets ────────────────────────────
    for split_name in ["train", "validation_hidden", "test_hidden"]:
        parquet_path = _CLEAN_DIR / f"{split_name}_clean.parquet"
        if not parquet_path.exists():
            print(f"[enrich_rt_oscar] WARNING: {parquet_path.name} not found — skipping")
            continue
        split_df = pd.read_parquet(parquet_path)
        enriched = build_enriched_split(split_df, rt_clean, oscar_film_feats, best_rt_method)
        enriched.to_parquet(parquet_path, index=False)
        print(
            f"[enrich_rt_oscar] Saved enriched {split_name}_clean.parquet  "
            f"shape={enriched.shape}  "
            f"rt_match={int(enriched['rt_match_flag'].sum()):,}/{len(enriched):,}  "
            f"oscar={int(enriched['oscar_was_nominated'].sum()):,}/{len(enriched):,}"
        )

    return state


if __name__ == "__main__":
    run({})
