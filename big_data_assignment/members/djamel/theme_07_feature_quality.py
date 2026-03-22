#!/usr/bin/env python3
"""
Theme 07 — Feature Quality Diagnostics
========================================
Shows per-feature:
  • Univariate ROC-AUC (train & val)
  • Mutual information with label
  • Spearman correlation with label
  • PSI (Population Stability Index) train vs val
  • Composite goodness score (weighted rank combination)
  • Feature status: keep / drop_candidate / review
  • Color-coded feature ranking heatmap

Reads  : outputs_restart/features_train_prepped.csv
Writes : outputs_restart/theme_07_feature_quality.html
         outputs_restart/feature_goodness.csv
"""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

MEMBER = Path(__file__).resolve().parent
OUT    = MEMBER / "outputs_restart"
OUT.mkdir(exist_ok=True)

SEED = 42
Y   = "#F5C518"; B = "#1848f5"; BG = "#0a0a0a"; CRD = "#141414"
TXT = "#ffffff"; MUT = "#888888"; GRN = "#2ecc71"; RED = "#e74c3c"; ORG = "#f39c12"

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
"""


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


def _safe_auc(y, x) -> float:
    if x.nunique() <= 1 or len(np.unique(y)) != 2:
        return float("nan")
    auc = float(roc_auc_score(y, x))
    return max(auc, 1.0 - auc)


def compute_psi(train_vals: pd.Series, val_vals: pd.Series, n_bins=10) -> float:
    tr = pd.to_numeric(train_vals, errors="coerce").dropna()
    vl = pd.to_numeric(val_vals,   errors="coerce").dropna()
    if tr.empty or vl.empty:
        return float("nan")
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(tr, quantiles))
    if len(edges) < 3:
        return float("nan")
    tr_dist = pd.cut(tr, bins=edges, include_lowest=True).value_counts(normalize=True).sort_index()
    vl_dist = pd.cut(vl, bins=edges, include_lowest=True).value_counts(normalize=True).sort_index()
    al = pd.concat([tr_dist, vl_dist], axis=1).fillna(1e-6).clip(lower=1e-6)
    al.columns = ["t", "v"]
    return float(((al["v"] - al["t"]) * np.log(al["v"] / al["t"])).sum())


def compute_goodness(train_X, train_y, val_X, val_y, feat_cols: List[str]) -> pd.DataFrame:
    y_tr = train_y.astype(int).to_numpy()
    y_vl = val_y.astype(int).to_numpy()
    rows = []
    for feat in feat_cols:
        x_tr = pd.to_numeric(train_X[feat], errors="coerce") if feat in train_X.columns else pd.Series(dtype=float)
        x_vl = pd.to_numeric(val_X[feat],   errors="coerce") if feat in val_X.columns   else pd.Series(dtype=float)
        med = float(x_tr.median()) if not x_tr.dropna().empty else 0.0
        x_tr_f = x_tr.fillna(med)
        x_vl_f = x_vl.fillna(med)

        auc_tr = _safe_auc(y_tr, x_tr_f)
        auc_vl = _safe_auc(y_vl, x_vl_f)
        mi = float(mutual_info_classif(
            x_tr_f.to_numpy().reshape(-1, 1), y_tr,
            discrete_features=False, random_state=SEED)[0])
        spear_tr = float(pd.Series(x_tr_f).corr(pd.Series(y_tr), method="spearman")) if x_tr_f.nunique() > 1 else float("nan")
        spear_vl = float(pd.Series(x_vl_f).corr(pd.Series(y_vl), method="spearman")) if x_vl_f.nunique() > 1 else float("nan")
        psi = compute_psi(x_tr, x_vl)

        rows.append({
            "feature":              feat,
            "missing_rate_train":   float(x_tr.isna().mean()),
            "missing_rate_val":     float(x_vl.isna().mean()),
            "std_train":            float(x_tr_f.std(ddof=0)),
            "univariate_auc_train": auc_tr,
            "univariate_auc_val":   auc_vl,
            "mutual_info":          mi,
            "spearman_train":       spear_tr,
            "spearman_val":         spear_vl,
            "abs_spearman_val":     abs(spear_vl) if not np.isnan(spear_vl) else float("nan"),
            "psi_train_vs_val":     psi,
        })

    df = pd.DataFrame(rows)
    auc_r   = df["univariate_auc_val"].fillna(0.5).rank(pct=True)
    mi_r    = df["mutual_info"].fillna(0).rank(pct=True)
    sp_r    = df["abs_spearman_val"].fillna(0).rank(pct=True)
    psi_max = df["psi_train_vs_val"].max(skipna=True)
    psi_max = psi_max if not pd.isna(psi_max) else 1.0
    psi_r   = (-df["psi_train_vs_val"].fillna(psi_max)).rank(pct=True)
    miss_r  = (-df["missing_rate_train"].fillna(1.0)).rank(pct=True)
    df["goodness_score"] = 0.35*auc_r + 0.25*mi_r + 0.20*sp_r + 0.10*psi_r + 0.10*miss_r

    median_good = float(df["goodness_score"].median())
    df["status"] = "review"
    df.loc[(df["psi_train_vs_val"].fillna(0) > 0.2), "status"] = "review"
    df.loc[(df["goodness_score"] >= 0.60), "status"] = "keep"
    df.loc[
        (df["univariate_auc_val"].fillna(0.5) < 0.52) &
        (df["mutual_info"].fillna(0) < df["mutual_info"].median()) &
        (df["goodness_score"] < median_good),
        "status"
    ] = "drop_candidate"

    return df.sort_values("goodness_score", ascending=False).reset_index(drop=True)


# ── figures ────────────────────────────────────────────────────────────────────

def _fig_auc_bar(diag: pd.DataFrame) -> str:
    df = diag.dropna(subset=["univariate_auc_val"]).sort_values("univariate_auc_val", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.35)), facecolor=BG)
    ax.set_facecolor(CRD)
    colors = [GRN if v >= 0.6 else Y if v >= 0.55 else ORG if v >= 0.52 else RED for v in df["univariate_auc_val"]]
    ax.barh(df["feature"], df["univariate_auc_val"], color=colors, alpha=0.85)
    ax.axvline(0.5, color=MUT, linestyle="--", linewidth=1)
    ax.axvline(0.55, color=ORG, linestyle="--", linewidth=1)
    ax.axvline(0.60, color=GRN, linestyle="--", linewidth=1)
    ax.set_xlabel("Univariate ROC-AUC (validation)", color=TXT)
    ax.set_title("Feature univariate predictive power", color=TXT, fontsize=12)
    ax.tick_params(colors=TXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_color(MUT)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["univariate_auc_val"] + 0.002, i, f"{row['univariate_auc_val']:.3f}",
                va="center", color=TXT, fontsize=7)
    fig.tight_layout()
    return _img(fig)


def _fig_mi_bar(diag: pd.DataFrame) -> str:
    df = diag.sort_values("mutual_info", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.35)), facecolor=BG)
    ax.set_facecolor(CRD)
    ax.barh(df["feature"], df["mutual_info"], color=B, alpha=0.85)
    ax.set_xlabel("Mutual Information (train)", color=TXT)
    ax.set_title("Mutual information with label", color=TXT, fontsize=12)
    ax.tick_params(colors=TXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_color(MUT)
    fig.tight_layout()
    return _img(fig)


def _fig_psi_bar(diag: pd.DataFrame) -> str:
    df = diag.dropna(subset=["psi_train_vs_val"]).sort_values("psi_train_vs_val", ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.35)), facecolor=BG)
    ax.set_facecolor(CRD)
    colors = [RED if v > 0.25 else ORG if v > 0.1 else GRN for v in df["psi_train_vs_val"]]
    ax.barh(df["feature"], df["psi_train_vs_val"], color=colors, alpha=0.85)
    ax.axvline(0.1,  color=ORG, linestyle="--", linewidth=1)
    ax.axvline(0.25, color=RED, linestyle="--", linewidth=1)
    ax.set_xlabel("PSI (train vs val)", color=TXT)
    ax.set_title("Population Stability Index — drift between train and validation", color=TXT, fontsize=12)
    ax.tick_params(colors=TXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_color(MUT)
    ax.text(0.11, -0.5, "caution", color=ORG, fontsize=8)
    ax.text(0.26, -0.5, "high drift", color=RED, fontsize=8)
    fig.tight_layout()
    return _img(fig)


def _fig_goodness_heatmap(diag: pd.DataFrame) -> str:
    cols = ["univariate_auc_val", "mutual_info", "abs_spearman_val", "psi_train_vs_val", "goodness_score"]
    cols = [c for c in cols if c in diag.columns]
    df   = diag.set_index("feature")[cols].fillna(0)
    # Normalize each col to [0,1] for display
    df_n = (df - df.min()) / (df.max() - df.min() + 1e-9)
    # Invert PSI (lower = better)
    if "psi_train_vs_val" in df_n.columns:
        df_n["psi_train_vs_val"] = 1.0 - df_n["psi_train_vs_val"]

    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 1.5), max(5, len(df) * 0.35)), facecolor=BG)
    im = ax.imshow(df_n.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right", color=TXT, fontsize=8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index.tolist(), color=TXT, fontsize=8)
    ax.set_title("Feature quality heatmap (green=good, red=poor/high-drift)", color=TXT, fontsize=11, pad=10)
    for i in range(len(df)):
        for j in range(len(cols)):
            raw_val = df.iloc[i, j]
            ax.text(j, i, f"{raw_val:.2f}", ha="center", va="center",
                    fontsize=6, color="black" if 0.3 < df_n.iloc[i, j] < 0.7 else "white")
    plt.colorbar(im, ax=ax, label="Normalized score", shrink=0.6)
    fig.tight_layout()
    return _img(fig)


def _fig_status_pie(diag: pd.DataFrame) -> str:
    counts = diag["status"].value_counts()
    colors_map = {"keep": GRN, "review": ORG, "drop_candidate": RED}
    labels = counts.index.tolist()
    vals   = counts.values.tolist()
    clrs   = [colors_map.get(l, MUT) for l in labels]
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=BG)
    wedges, texts, autotexts = ax.pie(vals, labels=labels, colors=clrs, autopct="%1.0f%%",
                                       startangle=90, textprops={"color": TXT})
    for at in autotexts: at.set_color(BG)
    ax.set_title("Feature status distribution", color=TXT, fontsize=11)
    fig.tight_layout()
    return _img(fig)


def _goodness_table(diag: pd.DataFrame) -> str:
    status_color = {"keep": GRN, "review": ORG, "drop_candidate": RED}
    rows = ""
    for _, row in diag.iterrows():
        sc = status_color.get(row.get("status", "review"), TXT)
        psi = row.get("psi_train_vs_val", float("nan"))
        psi_color = RED if (not np.isnan(psi) and psi > 0.25) else ORG if (not np.isnan(psi) and psi > 0.1) else GRN
        auc = row.get("univariate_auc_val", float("nan"))
        auc_color = GRN if (not np.isnan(auc) and auc >= 0.6) else ORG if (not np.isnan(auc) and auc >= 0.55) else RED
        rows += (
            f"<tr>"
            f"<td><code>{row['feature']}</code></td>"
            f"<td style='color:{auc_color}'>{auc:.3f}" if not np.isnan(auc) else "<td>—"
            f"</td>"
            f"<td>{row.get('mutual_info', float('nan')):.4f}" if not np.isnan(row.get('mutual_info', float('nan'))) else "<td>—"
            f"</td>"
            f"<td>{row.get('abs_spearman_val', float('nan')):.3f}" if not np.isnan(row.get('abs_spearman_val', float('nan'))) else "<td>—"
            f"</td>"
            f"<td style='color:{psi_color}'>{psi:.3f}" if not np.isnan(psi) else "<td>—"
            f"</td>"
            f"<td>{row.get('goodness_score', 0):.3f}</td>"
            f"<td style='color:{sc};font-weight:600'>{row.get('status','—')}</td>"
            f"</tr>"
        )
    return f"""<table>
<tr><th>Feature</th><th>AUC-val</th><th>MI</th><th>|Spearman|</th><th>PSI</th><th>Goodness</th><th>Status</th></tr>
{rows}</table>"""


def run(state: dict) -> dict:
    feat_df = state.get("features_train_prepped")
    if feat_df is None:
        fp = OUT / "features_train_prepped.csv"
        if fp.exists():
            feat_df = pd.read_csv(fp)
        else:
            fp2 = OUT / "features_train.csv"
            if fp2.exists():
                feat_df = pd.read_csv(fp2)
    if feat_df is None:
        raise FileNotFoundError("features_train_prepped.csv not found — run theme_06 first.")

    for col in feat_df.columns:
        if col not in ("tconst", "primaryTitle", "canonical_title"):
            feat_df[col] = pd.to_numeric(feat_df[col], errors="coerce")

    if "label" not in feat_df.columns:
        raise ValueError("label column missing from features_train_prepped.csv")

    label_num = pd.to_numeric(feat_df["label"], errors="coerce")
    dropped = int(label_num.isna().sum())
    if dropped:
        print(f"[theme_07] Dropping {dropped} rows with missing/non-numeric label before split.")
    feat_df = feat_df.loc[label_num.notna()].copy()
    feat_df["label"] = label_num.loc[label_num.notna()].astype(int).to_numpy()

    feat_cols = [c for c in feat_df.columns if c not in ("tconst", "label", "primaryTitle", "canonical_title")]
    feat_cols = [c for c in feat_cols if feat_df[c].notna().sum() > 0]

    # Internal 80/20 split for val set (stratified)
    train_idx, val_idx = train_test_split(
        feat_df.index, test_size=0.20, random_state=SEED,
        stratify=feat_df["label"].astype(int)
    )
    train_X = feat_df.loc[train_idx, feat_cols].reset_index(drop=True)
    val_X   = feat_df.loc[val_idx,   feat_cols].reset_index(drop=True)
    train_y = feat_df.loc[train_idx, "label"].reset_index(drop=True)
    val_y   = feat_df.loc[val_idx,   "label"].reset_index(drop=True)

    print(f"[theme_07] Computing goodness for {len(feat_cols)} features...")
    diag = compute_goodness(train_X, train_y, val_X, val_y, feat_cols)

    keeps  = (diag["status"] == "keep").sum()
    drops  = (diag["status"] == "drop_candidate").sum()
    review = (diag["status"] == "review").sum()
    top3   = ", ".join(diag.head(3)["feature"].tolist())

    kpis = (
        _kpi(str(len(feat_cols)),       "Features<br>evaluated")
      + _kpi(str(keeps),               f"Status: keep<br>goodness≥0.60")
      + _kpi(str(drops),               f"Status: drop_candidate")
      + _kpi(str(review),              f"Status: review")
    )

    sections = []

    sections.append(_card("1. Feature goodness score — ranked heatmap", f"""
<div class="note">
<strong>Goodness score</strong> = 0.35×AUC_rank + 0.25×MI_rank + 0.20×|Spearman|_rank + 0.10×(1-PSI)_rank + 0.10×(1-miss)_rank.
Green = good signal + stable. Red = low power or high drift.
</div>
{_fig_goodness_heatmap(diag)}
"""))

    sections.append(_card("2. Univariate ROC-AUC per feature (validation set)", f"""
<div class="note">
AUC is computed one feature at a time against the label. Threshold lines: 0.50 (random) · 0.55 (weak) · 0.60 (useful).
</div>
{_fig_auc_bar(diag)}
"""))

    sections.append(_card("3. Mutual information with label (training set)", f"""
<div class="note">MI is non-negative and model-agnostic. It captures non-linear relationships that Pearson misses.</div>
{_fig_mi_bar(diag)}
"""))

    sections.append(_card("4. PSI — distribution drift train vs validation", f"""
<div class="note">
<strong>PSI interpretation:</strong> &lt;0.1 = stable (green) · 0.1–0.25 = moderate drift (orange) · &gt;0.25 = high drift (red).
High drift features may be unstable at test time.
</div>
{_fig_psi_bar(diag)}
"""))

    sections.append(_card("5. Feature status summary + full diagnostics table", f"""
<div class="grid2">
  <div>{_fig_status_pie(diag)}</div>
  <div>
    <div class="note">
    <strong>Top 3 features by goodness:</strong> {top3}<br>
    Status rules: keep if goodness≥0.60; drop_candidate if AUC&lt;0.52 AND MI below median AND goodness&lt;median;
    review otherwise.
    </div>
  </div>
</div>
{_goodness_table(diag)}
"""))

    diag.to_csv(OUT / "feature_goodness.csv", index=False)
    state["feature_goodness"] = diag
    state["feat_cols_quality"] = feat_cols
    print(f"[theme_07] Saved feature_goodness.csv  ({len(diag)} features)")

    html = _page(
        "Theme 07 — Feature Quality Diagnostics",
        "univariate AUC · mutual info · Spearman · PSI drift · goodness score · status",
        kpis, sections,
    )
    (OUT / "theme_07_feature_quality.html").write_text(html, encoding="utf-8")
    print(f"[theme_07] Wrote {OUT}/theme_07_feature_quality.html")
    return state


if __name__ == "__main__":
    run({})
    print("[theme_07] Done.")
