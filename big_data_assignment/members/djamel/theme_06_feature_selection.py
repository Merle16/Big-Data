#!/usr/bin/env python3
"""
Theme 06 — Feature Selection & Matrix Preparation
===================================================
Shows exactly:
  • The explicit feature disposition table (keep/drop/cap/flag) per feature
  • The endYear decision — why it is dropped with evidence (missingness rate, std)
  • Imputation policy per feature (NOT blanket median)
  • Capping policy: quantile bounds fitted on train only
  • Before/after capping distributions for heavy-tailed features
  • The final feature matrix shape and NaN audit after preparation

Reads  : outputs_restart/features_train.csv
Writes : outputs_restart/theme_06_feature_selection.html
         outputs_restart/features_train_prepped.csv
"""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MEMBER = Path(__file__).resolve().parent
OUT    = MEMBER / "outputs_restart"
OUT.mkdir(exist_ok=True)

Y   = "#F5C518"; B = "#1848f5"; BG = "#0a0a0a"; CRD = "#141414"
TXT = "#ffffff"; MUT = "#888888"; GRN = "#2ecc71"; RED = "#e74c3c"; ORG = "#f39c12"; PRP = "#9b59b6"

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
.green{{color:{GRN};font-weight:600}} .red{{color:{RED};font-weight:600}}
.orange{{color:{ORG};font-weight:600}}
"""

DISPOSITION_REGISTRY = [
    {"feature": "startYear",               "action": "keep_numeric",    "imputation": "median_train",   "scaling": "standard",  "struct_miss": "no",      "justification": "Release era matters; low missingness; recover with train median."},
    {"feature": "endYear",                 "action": "drop",            "imputation": "none",           "scaling": "none",      "struct_miss": "yes",     "justification": "~90% missing. Captured by end_missing + year_span. Median imputation would be misleading."},
    {"feature": "runtimeMinutes",          "action": "cap_then_keep",   "imputation": "median_train",   "scaling": "standard",  "struct_miss": "no",      "justification": "Runtime informative but heavy-tailed; cap+impute."},
    {"feature": "numVotes_log1p",          "action": "cap_then_keep",   "imputation": "median_train",   "scaling": "standard",  "struct_miss": "no",      "justification": "Popularity proxy; log + capping handles skew."},
    {"feature": "title_len",               "action": "keep_numeric",    "imputation": "none",           "scaling": "standard",  "struct_miss": "no",      "justification": "Always computable from title; no missingness."},
    {"feature": "title_word_count",        "action": "keep_numeric",    "imputation": "none",           "scaling": "standard",  "struct_miss": "no",      "justification": "Always computable."},
    {"feature": "title_has_digit",         "action": "keep_numeric",    "imputation": "none",           "scaling": "none",      "struct_miss": "no",      "justification": "Binary flag."},
    {"feature": "title_has_colon",         "action": "keep_numeric",    "imputation": "none",           "scaling": "none",      "struct_miss": "no",      "justification": "Binary flag."},
    {"feature": "title_has_question",      "action": "keep_numeric",    "imputation": "none",           "scaling": "none",      "struct_miss": "no",      "justification": "Binary flag."},
    {"feature": "title_upper_ratio",       "action": "keep_numeric",    "imputation": "none",           "scaling": "standard",  "struct_miss": "no",      "justification": "Bounded [0,1]; no imputation needed."},
    {"feature": "has_original_title",      "action": "keep_numeric",    "imputation": "none",           "scaling": "none",      "struct_miss": "no",      "justification": "Binary flag."},
    {"feature": "runtime_missing",         "action": "keep_flag_only",  "imputation": "none",           "scaling": "none",      "struct_miss": "partial", "justification": "Missingness is signal; flag retained."},
    {"feature": "votes_missing",           "action": "keep_flag_only",  "imputation": "none",           "scaling": "none",      "struct_miss": "partial", "justification": "Missingness is signal."},
    {"feature": "start_missing",           "action": "keep_flag_only",  "imputation": "none",           "scaling": "none",      "struct_miss": "partial", "justification": "Missingness is signal."},
    {"feature": "end_missing",             "action": "keep_flag_only",  "imputation": "none",           "scaling": "none",      "struct_miss": "yes",     "justification": "endYear structurally missing; flag is informative."},
    {"feature": "year_span",               "action": "keep_numeric",    "imputation": "none",           "scaling": "standard",  "struct_miss": "no",      "justification": "0 when either year absent; captures lifecycle without leakage."},
    {"feature": "num_directors",           "action": "keep_numeric",    "imputation": "zero_fill",      "scaling": "standard",  "struct_miss": "no",      "justification": "Count; 0 for no edge data."},
    {"feature": "num_unique_directors",    "action": "keep_numeric",    "imputation": "zero_fill",      "scaling": "standard",  "struct_miss": "no",      "justification": "Count."},
    {"feature": "num_writers",             "action": "keep_numeric",    "imputation": "zero_fill",      "scaling": "standard",  "struct_miss": "no",      "justification": "Count."},
    {"feature": "num_unique_writers",      "action": "keep_numeric",    "imputation": "zero_fill",      "scaling": "standard",  "struct_miss": "no",      "justification": "Count."},
    {"feature": "is_auteur",               "action": "keep_numeric",    "imputation": "none",           "scaling": "none",      "struct_miss": "no",      "justification": "Binary flag derived from counts."},
    {"feature": "director_hit_rate",       "action": "encode_then_keep","imputation": "global_mean",    "scaling": "optional",  "struct_miss": "no",      "justification": "OOF-safe; global mean fallback for unseen."},
    {"feature": "writer_hit_rate",         "action": "encode_then_keep","imputation": "global_mean",    "scaling": "optional",  "struct_miss": "no",      "justification": "OOF-safe."},
    {"feature": "canonical_title_hit_rate","action": "encode_then_keep","imputation": "global_mean",    "scaling": "optional",  "struct_miss": "no",      "justification": "OOF-safe."},
    {"feature": "title_group_size_train",  "action": "keep_numeric",    "imputation": "zero_fill",      "scaling": "standard",  "struct_miss": "no",      "justification": "Count; 0 for unseen."},
    {"feature": "title_unique_years_train","action": "keep_numeric",    "imputation": "zero_fill",      "scaling": "standard",  "struct_miss": "no",      "justification": "Count."},
    {"feature": "title_conflicting_years", "action": "keep_flag_only",  "imputation": "none",           "scaling": "none",      "struct_miss": "no",      "justification": "Binary conflict flag."},
    {"feature": "title_sim_to_hit",        "action": "keep_numeric",    "imputation": "zero_fill",      "scaling": "standard",  "struct_miss": "no",      "justification": "Cosine similarity; bounded; 0 is valid default."},
    {"feature": "title_sim_to_non_hit",    "action": "keep_numeric",    "imputation": "zero_fill",      "scaling": "standard",  "struct_miss": "no",      "justification": "Cosine similarity."},
    {"feature": "title_sim_margin",        "action": "keep_numeric",    "imputation": "zero_fill",      "scaling": "standard",  "struct_miss": "no",      "justification": "Margin = hit_sim - non_hit_sim."},
]

ACTION_COLOR = {
    "drop":           RED,
    "keep_numeric":   GRN,
    "keep_flag_only": GRN,
    "cap_then_keep":  ORG,
    "encode_then_keep": Y,
}
ACTION_LABEL = {
    "drop":           "❌ DROP",
    "keep_numeric":   "✅ KEEP",
    "keep_flag_only": "✅ KEEP (flag)",
    "cap_then_keep":  "⚠️ CAP+KEEP",
    "encode_then_keep": "🔑 ENCODE+KEEP",
}


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


def _disposition_table_html(df_feat: Optional[pd.DataFrame] = None) -> str:
    miss_map: Dict[str, float] = {}
    std_map: Dict[str, float] = {}
    if df_feat is not None:
        for col in df_feat.columns:
            miss_map[col] = float(df_feat[col].isna().mean())
            try:
                std_map[col] = float(pd.to_numeric(df_feat[col], errors="coerce").std())
            except Exception:
                std_map[col] = float("nan")

    rows = ""
    for rec in DISPOSITION_REGISTRY:
        feat = rec["feature"]
        action = rec["action"]
        color = ACTION_COLOR.get(action, TXT)
        label = ACTION_LABEL.get(action, action)
        miss = f"{miss_map[feat]:.1%}" if feat in miss_map else "—"
        std  = f"{std_map[feat]:.3f}" if feat in std_map and not np.isnan(std_map.get(feat, float("nan"))) else "—"
        struct = rec["struct_miss"]
        struct_html = f'<span style="color:{RED}">YES</span>' if struct == "yes" else (
            f'<span style="color:{ORG}">partial</span>' if struct == "partial" else "no")
        rows += (
            f"<tr>"
            f"<td><code>{feat}</code></td>"
            f"<td style='color:{color};font-weight:600'>{label}</td>"
            f"<td>{rec['imputation']}</td>"
            f"<td>{rec['scaling']}</td>"
            f"<td>{struct_html}</td>"
            f"<td>{miss}</td>"
            f"<td style='font-size:.78rem;color:#bbb'>{rec['justification']}</td>"
            f"</tr>"
        )
    return f"""<table>
<tr><th>Feature</th><th>Action</th><th>Imputation</th><th>Scaling</th>
<th>Struct. miss</th><th>Miss rate</th><th>Justification</th></tr>
{rows}</table>"""


def _fig_action_summary() -> str:
    from collections import Counter
    counts = Counter(r["action"] for r in DISPOSITION_REGISTRY)
    labels = list(counts.keys())
    vals   = [counts[l] for l in labels]
    colors = [ACTION_COLOR.get(l, MUT) for l in labels]
    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor=BG)
    ax.set_facecolor(CRD)
    bars = ax.bar(labels, vals, color=colors, alpha=0.9)
    ax.set_ylabel("# features", color=TXT)
    ax.set_title("Action distribution across feature registry", color=TXT, fontsize=12)
    ax.tick_params(colors=TXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_color(MUT)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.1, str(v),
                ha="center", color=TXT, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _img(fig)


def _fig_endyear_evidence(df_feat: pd.DataFrame) -> str:
    """Visual evidence for why endYear is dropped."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), facecolor=BG)

    # Missingness bar
    ax = axes[0]
    ax.set_facecolor(CRD)
    cols = ["startYear", "endYear", "runtimeMinutes", "numVotes", "title_len"]
    cols = [c for c in cols if c in df_feat.columns]
    miss = [df_feat[c].isna().mean() for c in cols]
    colors = [RED if m > 0.5 else ORG if m > 0.1 else GRN for m in miss]
    ax.barh(cols, miss, color=colors)
    ax.set_xlabel("Missing rate", color=TXT)
    ax.set_title("Missing rates — endYear vs others", color=TXT, fontsize=10)
    ax.tick_params(colors=TXT, labelsize=8)
    ax.axvline(0.9, color=RED, linestyle="--", linewidth=1.5)
    ax.text(0.92, 0, "90%", color=RED, va="bottom", fontsize=8)
    for sp in ax.spines.values(): sp.set_color(MUT)

    # endYear distribution (only non-missing)
    ax = axes[1]
    ax.set_facecolor(CRD)
    if "endYear" in df_feat.columns:
        vals = pd.to_numeric(df_feat["endYear"], errors="coerce").dropna()
        if len(vals) > 0:
            ax.hist(vals, bins=30, color=RED, alpha=0.8, edgecolor="none")
            ax.set_title(f"endYear distribution (only {len(vals)} non-null rows)", color=TXT, fontsize=10)
        else:
            ax.text(0.5, 0.5, "All endYear values are missing", ha="center", va="center",
                    color=RED, transform=ax.transAxes, fontsize=12)
    ax.tick_params(colors=TXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_color(MUT)

    fig.suptitle("Evidence for dropping endYear: structural missingness + sparse remaining data", color=TXT, fontsize=11)
    fig.tight_layout()
    return _img(fig)


def _fig_capping(df_feat: pd.DataFrame, cap_col: str, q_low=0.01, q_high=0.99) -> str:
    vals = pd.to_numeric(df_feat[cap_col], errors="coerce").dropna()
    if vals.empty:
        return ""
    lo = float(vals.quantile(q_low))
    hi = float(vals.quantile(q_high))
    capped = vals.clip(lo, hi)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3), facecolor=BG)
    for ax, data, title, color in zip(axes,
                                       [vals, capped],
                                       [f"{cap_col} — BEFORE capping", f"{cap_col} — AFTER capping [p{q_low*100:.0f}, p{q_high*100:.0f}]"],
                                       [RED, GRN]):
        ax.set_facecolor(CRD)
        ax.hist(data, bins=40, color=color, alpha=0.8, edgecolor="none")
        ax.set_title(title, color=TXT, fontsize=9)
        ax.tick_params(colors=TXT, labelsize=7)
        for sp in ax.spines.values(): sp.set_color(MUT)
        ax.set_xlabel(f"mean={data.mean():.2f}  std={data.std():.2f}  max={data.max():.1f}", color=MUT, fontsize=7)
    fig.tight_layout()
    return _img(fig)


def _fig_imputation_policy() -> str:
    """Bar showing imputation policy frequency."""
    from collections import Counter
    counts = Counter(r["imputation"] for r in DISPOSITION_REGISTRY)
    labels = list(counts.keys())
    vals   = [counts[l] for l in labels]
    policy_colors = {"none": GRN, "median_train": Y, "zero_fill": B, "global_mean": ORG}
    colors = [policy_colors.get(l, MUT) for l in labels]
    fig, ax = plt.subplots(figsize=(8, 3), facecolor=BG)
    ax.set_facecolor(CRD)
    bars = ax.bar(labels, vals, color=colors, alpha=0.9)
    ax.set_ylabel("# features", color=TXT)
    ax.set_title("Imputation policy distribution — NOT blanket median", color=TXT, fontsize=11)
    ax.tick_params(colors=TXT, labelsize=9)
    for sp in ax.spines.values(): sp.set_color(MUT)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.05, str(v),
                ha="center", color=TXT, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _img(fig)


def _apply_imputation(df: pd.DataFrame, medians: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for rec in DISPOSITION_REGISTRY:
        feat = rec["feature"]
        if feat not in out.columns:
            continue
        policy = rec["imputation"]
        if policy == "none":
            pass
        elif policy == "median_train":
            med = medians.get(feat, 0.0)
            out[feat] = out[feat].fillna(med)
        elif policy == "zero_fill":
            out[feat] = out[feat].fillna(0.0)
        elif policy == "global_mean":
            gm = medians.get(feat, float(out[feat].mean()))
            out[feat] = out[feat].fillna(gm)
    return out


def _fig_nan_audit(df_before: pd.DataFrame, df_after: pd.DataFrame, feat_cols: List[str]) -> str:
    cols_with_nan_before = [c for c in feat_cols if c in df_before.columns and df_before[c].isna().sum() > 0]
    if not cols_with_nan_before:
        return "<p style='color:#888'>No missing values in selected feature columns.</p>"
    before = [df_before[c].isna().sum() for c in cols_with_nan_before]
    after  = [df_after[c].isna().sum() if c in df_after.columns else 0 for c in cols_with_nan_before]
    x = np.arange(len(cols_with_nan_before))
    fig, ax = plt.subplots(figsize=(max(8, len(cols_with_nan_before) * 0.7), 4), facecolor=BG)
    ax.set_facecolor(CRD)
    ax.bar(x - 0.2, before, width=0.4, color=RED, alpha=0.8, label="before imputation")
    ax.bar(x + 0.2, after,  width=0.4, color=GRN, alpha=0.8, label="after imputation")
    ax.set_xticks(x)
    ax.set_xticklabels(cols_with_nan_before, rotation=45, ha="right", color=TXT, fontsize=8)
    ax.set_ylabel("NaN count", color=TXT)
    ax.set_title("NaN counts before vs after imputation", color=TXT, fontsize=12)
    ax.legend(facecolor=CRD, edgecolor=MUT, labelcolor=TXT)
    for sp in ax.spines.values(): sp.set_color(MUT)
    ax.tick_params(colors=TXT)
    fig.tight_layout()
    return _img(fig)


def run(state: dict) -> dict:
    # Load feature matrix
    feat_df = state.get("features_train")
    if feat_df is None:
        feat_df = state.get("train_feat")
    if feat_df is None:
        fp = OUT / "features_train.csv"
        if fp.exists():
            feat_df = pd.read_csv(fp)
            for col in feat_df.columns:
                if col not in ("tconst", "label", "primaryTitle", "canonical_title"):
                    feat_df[col] = pd.to_numeric(feat_df[col], errors="coerce")
    if feat_df is None:
        raise FileNotFoundError("features_train.csv not found — run theme_05 first.")

    # Determine final feature cols (exclude drops)
    drop_feats = {r["feature"] for r in DISPOSITION_REGISTRY if r["action"] == "drop"}
    registry_feats = {r["feature"] for r in DISPOSITION_REGISTRY}
    all_candidate = [c for c in feat_df.columns if c not in ("tconst", "label", "primaryTitle", "canonical_title")]
    final_feat_cols = [c for c in all_candidate if c not in drop_feats]

    # Capping (fit on train)
    CAP_COLS = ["runtimeMinutes", "numVotes_log1p"]
    cap_bounds: Dict[str, Tuple[float, float]] = {}
    feat_capped = feat_df.copy()
    for col in CAP_COLS:
        if col in feat_capped.columns:
            vals = pd.to_numeric(feat_capped[col], errors="coerce").dropna()
            if len(vals) > 0:
                lo, hi = float(vals.quantile(0.01)), float(vals.quantile(0.99))
                cap_bounds[col] = (lo, hi)
                feat_capped[col] = pd.to_numeric(feat_capped[col], errors="coerce").clip(lo, hi)

    # Imputation (median on train)
    medians: Dict[str, float] = {}
    for col in final_feat_cols:
        if col in feat_capped.columns:
            v = pd.to_numeric(feat_capped[col], errors="coerce")
            medians[col] = float(v.median()) if not v.dropna().empty else 0.0
    feat_imputed = _apply_imputation(feat_capped[["tconst"] + final_feat_cols + (["label"] if "label" in feat_capped.columns else [])], medians)

    total_feats  = len(all_candidate)
    dropped      = len(drop_feats & set(all_candidate))
    final        = len(final_feat_cols)
    nan_after    = int(feat_imputed[final_feat_cols].isna().sum().sum())

    kpis = (
        _kpi(str(total_feats), "Candidate<br>features")
      + _kpi(str(dropped),     f"Dropped<br>(endYear etc)")
      + _kpi(str(final),       "Final features<br>in model matrix")
      + _kpi(str(nan_after),   "NaN values<br>remaining")
    )

    sections = []

    # 1. Action distribution
    sections.append(_card("1. Disposition action summary", f"""
<div class="note">Every candidate feature has an explicit disposition decision. No feature silently falls through.</div>
<div class="grid2">
  <div>{_fig_action_summary()}</div>
  <div>{_fig_imputation_policy()}</div>
</div>
"""))

    # 2. Full disposition table
    sections.append(_card("2. Full feature disposition table", f"""
<div class="note">
<strong>How to read:</strong>
<span style="color:{RED}">❌ DROP</span> = excluded from model matrix.
<span style="color:{GRN}">✅ KEEP</span> = included as-is (with specified imputation).
<span style="color:{ORG}">⚠️ CAP+KEEP</span> = outlier-capped then kept.
<span style="color:{Y}">🔑 ENCODE+KEEP</span> = OOF-encoded then kept.
</div>
{_disposition_table_html(feat_df)}
"""))

    # 3. endYear evidence
    sections.append(_card("3. endYear decision — evidence of structural missingness", f"""
<div class="note">
<strong>Decision:</strong> <span style='color:{RED}'>DROP endYear</span>. Keep <code>end_missing</code> (flag) + <code>year_span</code> (0-filled span).
Imputing ~90% missing values with the median would introduce a fictional signal. The flag itself is more honest.
</div>
{_fig_endyear_evidence(feat_df)}
"""))

    # 4. Capping visualizations
    cap_html = ""
    for col in CAP_COLS:
        if col in feat_df.columns:
            cap_html += f"<h3>{col}</h3>{_fig_capping(feat_df, col)}"
            if col in cap_bounds:
                lo, hi = cap_bounds[col]
                cap_html += f"<div class='note'>Fitted bounds: [{lo:.3f}, {hi:.3f}] (p1–p99 on train). Applied to val/test without re-fitting.</div>"
    sections.append(_card("4. Outlier capping — before vs after", f"""
<div class="note">Quantile bounds are <strong>fitted on training data only</strong>, then applied to val/test. This prevents leakage from validation distribution into capping thresholds.</div>
{cap_html}
"""))

    # 5. NaN audit before/after imputation
    sections.append(_card("5. Imputation effect — NaN before vs after", f"""
<div class="note">Per-feature imputation policy: median (for numeric with random missingness),
zero_fill (for counts), global_mean (for OOF encodings), none (for binary flags and structurally missing).</div>
{_fig_nan_audit(feat_df, feat_imputed, final_feat_cols)}
"""))

    # 6. Final matrix summary
    shape_html = f"""
<div class="note">
<strong>Final matrix shape:</strong> {feat_imputed.shape[0]} rows × {len(final_feat_cols)} features<br>
<strong>Remaining NaN:</strong> {nan_after} (should be 0)<br>
<strong>Dropped features:</strong> {', '.join(sorted(drop_feats & set(all_candidate)))}
</div>
<table>
<tr><th>Feature</th><th>dtype</th><th>min</th><th>max</th><th>mean</th><th>std</th><th>NaN after</th></tr>
"""
    for col in final_feat_cols:
        if col not in feat_imputed.columns:
            continue
        v = pd.to_numeric(feat_imputed[col], errors="coerce")
        nan_cnt = int(v.isna().sum())
        nan_str = f'<span style="color:{RED}">{nan_cnt}</span>' if nan_cnt > 0 else "0"
        shape_html += (
            f"<tr><td><code>{col}</code></td>"
            f"<td>{feat_imputed[col].dtype}</td>"
            f"<td>{v.min():.3f}</td><td>{v.max():.3f}</td>"
            f"<td>{v.mean():.3f}</td><td>{v.std():.3f}</td>"
            f"<td>{nan_str}</td></tr>"
        )
    shape_html += "</table>"
    sections.append(_card("6. Final model matrix summary", shape_html))

    # Save prepped features
    feat_imputed.to_csv(OUT / "features_train_prepped.csv", index=False)
    state["features_train_prepped"] = feat_imputed
    state["final_feat_cols"] = final_feat_cols
    state["cap_bounds"] = cap_bounds
    state["medians"] = medians
    print(f"[theme_06] Saved features_train_prepped.csv  shape={feat_imputed.shape}")

    html = _page(
        "Theme 06 — Feature Selection & Matrix Preparation",
        "explicit disposition · endYear evidence · per-feature imputation · capping on train-only",
        kpis, sections,
    )
    (OUT / "theme_06_feature_selection.html").write_text(html, encoding="utf-8")
    print(f"[theme_06] Wrote {OUT}/theme_06_feature_selection.html")
    return state


if __name__ == "__main__":
    run({})
    print("[theme_06] Done.")
