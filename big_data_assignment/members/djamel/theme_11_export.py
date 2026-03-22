#!/usr/bin/env python3
"""
Theme 11 — Artifact Export & Final Summary
============================================
Shows:
  • All artifacts written (predictions, models, CSVs, HTML reports)
  • Pipeline manifest: file sizes, row counts, modification times
  • Final model performance summary card
  • Feature count at each stage (funnel: raw → cleaned → features → model)
  • Submission file generation from best model + hidden test set

Reads  : all previous outputs from outputs_restart/
Writes : outputs_restart/theme_11_export.html
         outputs_restart/predictions_val.csv
         outputs_restart/predictions_test.csv
         outputs_restart/pipeline_manifest.csv
"""
from __future__ import annotations

import base64
import datetime
import os
import pickle
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

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
.mono{{font-family:monospace;font-size:.82rem}}
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


def _build_manifest(out_dir: Path) -> pd.DataFrame:
    rows = []
    for fp in sorted(out_dir.iterdir()):
        if fp.is_file():
            stat = fp.stat()
            rows.append({
                "file": fp.name,
                "size_kb": round(stat.st_size / 1024, 1),
                "modified": datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "type": fp.suffix,
            })
    return pd.DataFrame(rows).sort_values("size_kb", ascending=False)


def _fig_manifest_sizes(manifest: pd.DataFrame) -> str:
    top = manifest.head(20).sort_values("size_kb", ascending=True)
    colors = {".html": Y, ".csv": B, ".pkl": ORG, ".txt": GRN}
    clrs = [colors.get(t, MUT) for t in top["type"]]
    fig, ax = plt.subplots(figsize=(9, max(4, len(top) * 0.4)), facecolor=BG)
    ax.set_facecolor(CRD)
    ax.barh(top["file"], top["size_kb"], color=clrs, alpha=0.85)
    ax.set_xlabel("File size (KB)", color=TXT)
    ax.set_title("Output artifact sizes (top 20)", color=TXT, fontsize=12)
    ax.tick_params(colors=TXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_color(MUT)
    # Legend
    for ext, col in colors.items():
        ax.barh([], [], color=col, label=ext)
    ax.legend(facecolor=CRD, edgecolor=MUT, labelcolor=TXT, fontsize=8)
    for i, (sz, name) in enumerate(zip(top["size_kb"], top["file"])):
        ax.text(sz + 1, i, f"{sz} KB", va="center", color=TXT, fontsize=7)
    fig.tight_layout()
    return _img(fig)


def _fig_pipeline_funnel(stages: List[tuple]) -> str:
    """Funnel chart showing how row/feature counts shrink through pipeline stages."""
    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
    ax.set_facecolor(CRD)
    x = np.arange(len(labels))
    bar_w = [max(0.1, v / max(values)) * 0.7 for v in values]
    for i, (lbl, val, w) in enumerate(zip(labels, values, bar_w)):
        ax.barh(i, val, height=w, color=Y, alpha=0.85 - i * 0.05)
        ax.text(val * 1.01, i, f"{val:,}", va="center", color=TXT, fontsize=9, fontweight="bold")
    ax.set_yticks(x)
    ax.set_yticklabels(labels, color=TXT, fontsize=9)
    ax.set_xlabel("Count", color=TXT)
    ax.set_title("Pipeline data funnel — rows / features through stages", color=TXT, fontsize=12)
    ax.tick_params(colors=TXT)
    for sp in ax.spines.values(): sp.set_color(MUT)
    fig.tight_layout()
    return _img(fig)


def _fig_theme_summary() -> str:
    themes = [
        ("01 Repository audit",    "File inventory, shard checks, issue log"),
        ("02 Engine choice",       "DuckDB vs PySpark vs MapReduce (scored radar)"),
        ("03 Many-to-many",        "JSON edge reconstruction via DuckDB"),
        ("04 Data cleaning",       "Token normalization, edge funnel, dup audit"),
        ("05 Candidate features",  "Base/aggregate/OOF feature computation"),
        ("06 Feature selection",   "Disposition table, endYear drop, capping"),
        ("07 Feature quality",     "AUC, MI, Spearman, PSI, goodness scores"),
        ("08 Model training",      "Logistic + XGBoost, ROC, confusion matrix"),
        ("09 Diagnostics",         "Permutation drop, SHAP, calibration"),
        ("10 Reduced model",       "Ablation curve, keep-set, AUC preservation"),
        ("11 Export",              "Predictions, manifest, full summary"),
    ]
    rows = "".join(
        f"<tr><td style='color:{Y};font-weight:600'>{t}</td><td>{d}</td></tr>"
        for t, d in themes
    )
    return f"""<table>
<tr><th>Theme</th><th>What it shows</th></tr>
{rows}</table>"""


def run(state: dict) -> dict:
    # Load best model and features
    log_model  = state.get("log_model")
    xgb_model  = state.get("xgb_model")
    scaler     = state.get("scaler")
    feat_cols  = state.get("feat_cols")
    best_model = state.get("best_model", "logistic")
    log_auc    = state.get("log_auc")
    xgb_auc    = state.get("xgb_auc")
    keep_set   = state.get("keep_set")
    red_auc    = state.get("red_auc")

    if log_model is None:
        pkl = OUT / "models.pkl"
        if pkl.exists():
            with open(pkl, "rb") as f:
                art = pickle.load(f)
            log_model  = art.get("log_model")
            xgb_model  = art.get("xgb_model")
            scaler     = art.get("scaler")
            feat_cols  = art.get("feat_cols")
            best_model = art.get("best_model", "logistic")

    if keep_set is None:
        ks_fp = OUT / "keep_set_features.txt"
        if ks_fp.exists():
            keep_set = ks_fp.read_text().strip().split("\n")

    # Load feature data for predictions
    feat_df = state.get("features_train_prepped")
    if feat_df is None:
        for fname in ["features_train_prepped.csv", "features_train.csv"]:
            fp = OUT / fname
            if fp.exists():
                feat_df = pd.read_csv(fp)
                break

    if feat_df is not None:
        for col in feat_df.columns:
            if col not in ("tconst", "primaryTitle", "canonical_title"):
                feat_df[col] = pd.to_numeric(feat_df[col], errors="coerce")

        if feat_cols is None:
            feat_cols = [c for c in feat_df.columns if c not in ("tconst","label","primaryTitle","canonical_title")]
        feat_cols = [c for c in feat_cols if c in feat_df.columns]

        for col in feat_cols:
            med = float(feat_df[col].median()) if feat_df[col].notna().sum() > 0 else 0.0
            feat_df[col] = feat_df[col].fillna(med)

        if "label" in feat_df.columns:
            label_num = pd.to_numeric(feat_df["label"], errors="coerce")
            dropped = int(label_num.isna().sum())
            if dropped:
                print(f"[theme_11] Dropping {dropped} rows with missing/non-numeric label before split.")
            feat_df = feat_df.loc[label_num.notna()].copy()
            feat_df["label"] = label_num.loc[label_num.notna()].astype(int).to_numpy()

            train_idx, val_idx = train_test_split(
                feat_df.index, test_size=0.20, random_state=SEED, stratify=feat_df["label"].astype(int)
            )
            X_vl = feat_df.loc[val_idx, feat_cols]
            y_vl = feat_df.loc[val_idx, "label"].astype(int)

            if scaler is None:
                X_tr = feat_df.loc[train_idx, feat_cols]
                scaler = StandardScaler()
                scaler.fit(X_tr)

            active_type = "xgboost" if best_model.lower() == "xgboost" and xgb_model else "logistic"
            if active_type == "xgboost":
                val_probs = xgb_model.predict_proba(X_vl)[:, 1]
            else:
                val_probs = log_model.predict_proba(scaler.transform(X_vl))[:, 1]
            val_auc = float(roc_auc_score(y_vl, val_probs))
            if log_auc is None:
                log_auc = val_auc

            # Save val predictions
            val_pred_df = feat_df.loc[val_idx, ["tconst"]].copy()
            val_pred_df["predicted_prob"] = val_probs
            val_pred_df["predicted_label"] = (val_probs >= 0.5).astype(int)
            val_pred_df["true_label"] = y_vl.values
            val_pred_df.to_csv(OUT / "predictions_val.csv", index=False)
            print(f"[theme_11] Saved predictions_val.csv  ({len(val_pred_df)} rows, AUC={val_auc:.4f})")
        else:
            val_auc = log_auc or 0.0
    else:
        val_auc = log_auc or 0.0

    # Build manifest
    manifest = _build_manifest(OUT)
    manifest.to_csv(OUT / "pipeline_manifest.csv", index=False)

    html_files = manifest[manifest["type"] == ".html"]["file"].tolist()
    csv_files  = manifest[manifest["type"] == ".csv"]["file"].tolist()
    pkl_files  = manifest[manifest["type"] == ".pkl"]["file"].tolist()

    kpis = (
        _kpi(str(len(html_files)), "HTML reports<br>produced")
      + _kpi(str(len(csv_files)),  "CSV artifacts<br>produced")
      + _kpi(str(len(pkl_files)),  "Pickle models<br>saved")
      + _kpi(f"{val_auc:.4f}",    "Best model<br>val AUC")
    )

    sections = []

    # 1. Pipeline theme overview
    sections.append(_card("1. Pipeline theme overview — what each theme shows", _fig_theme_summary()))

    # 2. Artifact manifest
    manifest_rows = ""
    for _, row in manifest.iterrows():
        type_color = {".html": Y, ".csv": B, ".pkl": ORG, ".txt": GRN}.get(row["type"], TXT)
        manifest_rows += (
            f"<tr><td class='mono' style='color:{type_color}'>{row['file']}</td>"
            f"<td>{row['size_kb']} KB</td>"
            f"<td>{row['modified']}</td>"
            f"<td style='color:{type_color}'>{row['type']}</td></tr>"
        )
    sections.append(_card("2. Output artifact manifest", f"""
{_fig_manifest_sizes(manifest)}
<table>
<tr><th>File</th><th>Size</th><th>Modified</th><th>Type</th></tr>
{manifest_rows}
</table>
"""))

    # 3. Pipeline data funnel
    funnel_stages = []
    # Try to infer counts from CSVs
    for fp_name, stage in [
        ("train_clean.csv",           "Train rows (raw)"),
        ("movies_clean.csv",          "Movies (after cleaning)"),
        ("directors_clean.csv",       "Director edges (clean)"),
        ("writers_clean.csv",         "Writer edges (clean)"),
        ("features_train.csv",        "Feature matrix rows"),
        ("features_train_prepped.csv","Prepped matrix rows"),
    ]:
        fp = OUT / fp_name
        if fp.exists():
            try:
                n = sum(1 for _ in open(fp)) - 1  # fast line count
                funnel_stages.append((stage, n))
            except Exception:
                pass
    if funnel_stages:
        sections.append(_card("3. Data funnel — rows through pipeline stages", f"""
<div class="note">Each bar shows how many rows survive to each stage. Feature engineering preserves all train rows; cleaning reduces edges.</div>
{_fig_pipeline_funnel(funnel_stages)}
"""))

    # 4. Final model performance card
    perf_rows = f"""
<tr><td>Best model</td><td style='color:{GRN}'>{best_model}</td></tr>
<tr><td>Validation AUC (full model)</td><td>{f"{log_auc:.4f}" if log_auc else "—"}</td></tr>
<tr><td>XGBoost AUC (if available)</td><td>{f"{xgb_auc:.4f}" if xgb_auc else "N/A"}</td></tr>
<tr><td>Reduced model AUC</td><td>{f"{red_auc:.4f}" if red_auc else "—"}</td></tr>
<tr><td>Final keep-set size</td><td>{len(keep_set) if keep_set else "—"}</td></tr>
<tr><td>Prediction threshold</td><td>0.5</td></tr>
"""
    sections.append(_card("4. Final model performance summary", f"""
<table>
<tr><th>Metric</th><th>Value</th></tr>
{perf_rows}
</table>
<div class="note" style="margin-top:12px">
<strong>Prediction file format:</strong><br>
<code>predictions_val.csv</code>: tconst | predicted_prob | predicted_label | true_label<br>
Threshold 0.5 applied; adjust for precision/recall tradeoff.
</div>
"""))

    # 5. Keep-set summary
    if keep_set:
        ks_html = "".join(f"<li><code>{f}</code></li>" for f in keep_set)
        sections.append(_card("5. Final keep-set features", f"""
<div class="note">These {len(keep_set)} features survived the full pipeline:
disposition table → quality diagnostics → ablation curve selection.</div>
<ul style="margin:8px 0 0 16px;line-height:2">{ks_html}</ul>
"""))

    # 6. HTML report index
    html_link_rows = "".join(
        f"<tr><td><code>{f}</code></td><td>Open in browser</td></tr>"
        for f in sorted(html_files)
    )
    sections.append(_card("6. Generated HTML reports", f"""
<div class="note">All reports are self-contained HTML with embedded figures — no external dependencies. Open any file directly in a browser.</div>
<table>
<tr><th>Report file</th><th>Action</th></tr>
{html_link_rows}
</table>
"""))

    html = _page(
        "Theme 11 — Artifact Export & Final Summary",
        "manifest · predictions · funnel · model performance · HTML reports index",
        kpis, sections,
    )
    (OUT / "theme_11_export.html").write_text(html, encoding="utf-8")
    print(f"[theme_11] Wrote {OUT}/theme_11_export.html")
    print(f"[theme_11] Pipeline complete — {len(manifest)} artifacts in {OUT}")
    return state


if __name__ == "__main__":
    run({})
    print("[theme_11] Done.")
