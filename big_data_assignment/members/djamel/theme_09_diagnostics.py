#!/usr/bin/env python3
"""
Theme 09 — Model Diagnostics
==============================
Shows:
  • Permutation AUC drop per feature (which features hurt the model when shuffled)
  • XGBoost SHAP mean absolute values
  • Combined diagnostic score and feature status classification
  • Train vs validation AUC gap (overfitting check)
  • Calibration curve

Reads  : outputs_restart/features_train_prepped.csv
         outputs_restart/models.pkl
Writes : outputs_restart/theme_09_diagnostics.html
         outputs_restart/feature_diagnostics.csv
"""
from __future__ import annotations

import base64
import pickle
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    import xgboost as xgb_lib
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


def _predict_proba(model_name, model_obj, X, scaler=None):
    if model_name == "xgboost":
        return model_obj.predict_proba(X)[:, 1]
    return model_obj.predict_proba(scaler.transform(X))[:, 1]


def compute_permutation_drop(model_name, model_obj, X_val, y_val, feat_cols, scaler=None):
    y = y_val.astype(int).to_numpy()
    base_probs = _predict_proba(model_name, model_obj, X_val[feat_cols], scaler)
    baseline_auc = float(roc_auc_score(y, base_probs))
    rng = np.random.default_rng(SEED)
    rows = []
    for feat in feat_cols:
        shuffled = X_val[feat_cols].copy()
        shuffled[feat] = rng.permutation(shuffled[feat].to_numpy())
        perm_probs = _predict_proba(model_name, model_obj, shuffled, scaler)
        perm_auc = float(roc_auc_score(y, perm_probs))
        rows.append({
            "feature": feat,
            "baseline_auc": baseline_auc,
            "permuted_auc": perm_auc,
            "perm_auc_drop": baseline_auc - perm_auc,
        })
    return pd.DataFrame(rows).sort_values("perm_auc_drop", ascending=False).reset_index(drop=True)


def compute_shap(xgb_model, X_val, feat_cols):
    if xgb_model is None or not HAS_XGB:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])
    booster = xgb_model.get_booster()
    dval = xgb_lib.DMatrix(X_val[feat_cols], feature_names=feat_cols)
    contrib = booster.predict(dval, pred_contribs=True)
    mean_abs = np.abs(contrib[:, :-1]).mean(axis=0)
    return pd.DataFrame({"feature": feat_cols, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)


# ── figures ────────────────────────────────────────────────────────────────────

def _fig_perm_drop(perm_df: pd.DataFrame, model_name: str) -> str:
    top = perm_df.head(20).sort_values("perm_auc_drop", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4, len(top) * 0.4)), facecolor=BG)
    ax.set_facecolor(CRD)
    colors = [GRN if v > 0.005 else ORG if v > 0 else RED for v in top["perm_auc_drop"]]
    ax.barh(top["feature"], top["perm_auc_drop"], color=colors, alpha=0.85)
    ax.axvline(0, color=MUT, linewidth=1)
    ax.axvline(0.002, color=GRN, linestyle="--", linewidth=1)
    ax.set_xlabel("AUC drop when feature is shuffled", color=TXT)
    ax.set_title(f"Permutation AUC drop — {model_name} (top 20)", color=TXT, fontsize=12)
    ax.tick_params(colors=TXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_color(MUT)
    baseline = top["baseline_auc"].iloc[0] if len(top) > 0 else 0
    ax.text(0.003, -1, f"▶ 0.002 threshold", color=GRN, fontsize=7)
    fig.tight_layout()
    return _img(fig)


def _fig_shap(shap_df: pd.DataFrame) -> str:
    if shap_df.empty:
        return "<p style='color:#888'>XGBoost SHAP not available (XGBoost not installed).</p>"
    top = shap_df.head(20).sort_values("mean_abs_shap", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4, len(top) * 0.4)), facecolor=BG)
    ax.set_facecolor(CRD)
    ax.barh(top["feature"], top["mean_abs_shap"], color=Y, alpha=0.85)
    ax.set_xlabel("Mean |SHAP value|", color=TXT)
    ax.set_title("XGBoost SHAP importance (top 20 features)", color=TXT, fontsize=12)
    ax.tick_params(colors=TXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_color(MUT)
    fig.tight_layout()
    return _img(fig)


def _fig_status_scatter(diag_df: pd.DataFrame) -> str:
    sc = {"keep": GRN, "review": ORG, "drop_candidate": RED}
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    ax.set_facecolor(CRD)
    for status, color in sc.items():
        subset = diag_df[diag_df["status"] == status]
        ax.scatter(subset["perm_auc_drop"].clip(lower=-0.01),
                   subset.get("goodness_score", subset["perm_auc_drop"] * 0),
                   color=color, alpha=0.85, s=60, label=status, zorder=3)
        for _, row in subset.iterrows():
            ax.annotate(row["feature"],
                        (row["perm_auc_drop"],
                         row.get("goodness_score", 0)),
                        fontsize=6, color=TXT, alpha=0.7)
    ax.axvline(0.002, color=GRN, linestyle="--", linewidth=1)
    ax.axvline(0, color=MUT, linewidth=1)
    ax.set_xlabel("Permutation AUC drop", color=TXT)
    ax.set_ylabel("Goodness score", color=TXT)
    ax.set_title("Feature diagnostic scatter — keep / review / drop_candidate", color=TXT, fontsize=11)
    ax.legend(facecolor=CRD, edgecolor=MUT, labelcolor=TXT)
    ax.tick_params(colors=TXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_color(MUT)
    fig.tight_layout()
    return _img(fig)


def _fig_calibration(log_probs, xgb_probs, y_val) -> str:
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
    ax.set_facecolor(CRD)
    ax.plot([0, 1], [0, 1], color=MUT, linestyle="--", linewidth=1, label="Perfect calibration")

    frac_pos, mean_pred = calibration_curve(y_val, log_probs, n_bins=10)
    ax.plot(mean_pred, frac_pos, marker="o", color=B, linewidth=2, label="Logistic")

    if xgb_probs is not None:
        frac_pos2, mean_pred2 = calibration_curve(y_val, xgb_probs, n_bins=10)
        ax.plot(mean_pred2, frac_pos2, marker="s", color=Y, linewidth=2, label="XGBoost")

    ax.set_xlabel("Mean predicted probability", color=TXT)
    ax.set_ylabel("Fraction of positives", color=TXT)
    ax.set_title("Calibration curve (reliability diagram)", color=TXT, fontsize=12)
    ax.legend(facecolor=CRD, edgecolor=MUT, labelcolor=TXT)
    ax.tick_params(colors=TXT)
    for sp in ax.spines.values(): sp.set_color(MUT)
    fig.tight_layout()
    return _img(fig)


def _fig_auc_gap(train_auc, val_auc, model_name) -> str:
    fig, ax = plt.subplots(figsize=(5, 3.5), facecolor=BG)
    ax.set_facecolor(CRD)
    labels = ["Train AUC", "Val AUC"]
    vals   = [train_auc, val_auc]
    colors = [Y, GRN]
    bars = ax.bar(labels, vals, color=colors, alpha=0.85, width=0.4)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("ROC-AUC", color=TXT)
    ax.set_title(f"{model_name} — Train vs Validation AUC (overfitting check)", color=TXT, fontsize=11)
    ax.tick_params(colors=TXT)
    for sp in ax.spines.values(): sp.set_color(MUT)
    gap = train_auc - val_auc
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.003, f"{v:.4f}",
                ha="center", color=TXT, fontsize=11, fontweight="bold")
    ax.text(0.5, 0.55, f"Gap: {gap:.4f}", ha="center", color=RED if gap > 0.05 else GRN,
            fontsize=12, fontweight="bold", transform=ax.transData)
    fig.tight_layout()
    return _img(fig)


def _diag_table(diag_df: pd.DataFrame) -> str:
    sc = {"keep": GRN, "review": ORG, "drop_candidate": RED}
    rows = ""
    for _, row in diag_df.iterrows():
        color = sc.get(row.get("status", "review"), TXT)
        drop = row.get("perm_auc_drop", float("nan"))
        shap = row.get("mean_abs_shap", float("nan"))
        good = row.get("goodness_score", float("nan"))
        rows += (
            f"<tr><td><code>{row['feature']}</code></td>"
            f"<td>{'—' if np.isnan(drop) else f'{drop:.5f}'}</td>"
            f"<td>{'—' if np.isnan(shap) else f'{shap:.5f}'}</td>"
            f"<td>{'—' if np.isnan(good) else f'{good:.3f}'}</td>"
            f"<td style='color:{color};font-weight:600'>{row.get('status','—')}</td></tr>"
        )
    return f"""<table>
<tr><th>Feature</th><th>Perm AUC drop</th><th>Mean |SHAP|</th><th>Goodness</th><th>Status</th></tr>
{rows}</table>"""


def run(state: dict) -> dict:
    # Load data
    feat_df = state.get("features_train_prepped")
    if feat_df is None:
        for fname in ["features_train_prepped.csv", "features_train.csv"]:
            fp = OUT / fname
            if fp.exists():
                feat_df = pd.read_csv(fp)
                break
    if feat_df is None:
        raise FileNotFoundError("features_train_prepped.csv not found.")

    for col in feat_df.columns:
        if col not in ("tconst", "primaryTitle", "canonical_title"):
            feat_df[col] = pd.to_numeric(feat_df[col], errors="coerce")

    # Load models
    log_model = state.get("log_model")
    xgb_model = state.get("xgb_model")
    scaler    = state.get("scaler")
    feat_cols = state.get("feat_cols")
    best_model= state.get("best_model", "logistic")

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

    if log_model is None:
        raise FileNotFoundError("models.pkl not found — run theme_08 first.")

    if "label" not in feat_df.columns:
        raise ValueError("label column missing.")

    label_num = pd.to_numeric(feat_df["label"], errors="coerce")
    dropped = int(label_num.isna().sum())
    if dropped:
        print(f"[theme_09] Dropping {dropped} rows with missing/non-numeric label before split.")
    feat_df = feat_df.loc[label_num.notna()].copy()
    feat_df["label"] = label_num.loc[label_num.notna()].astype(int).to_numpy()

    feat_cols = [c for c in (feat_cols or []) if c in feat_df.columns]
    if not feat_cols:
        feat_cols = [c for c in feat_df.columns if c not in ("tconst", "label", "primaryTitle", "canonical_title")]

    for col in feat_cols:
        med = float(feat_df[col].median()) if feat_df[col].notna().sum() > 0 else 0.0
        feat_df[col] = feat_df[col].fillna(med)

    train_idx, val_idx = train_test_split(
        feat_df.index, test_size=0.20, random_state=SEED,
        stratify=feat_df["label"].astype(int)
    )
    train_X = feat_df.loc[train_idx, feat_cols]
    val_X   = feat_df.loc[val_idx,   feat_cols]
    train_y = feat_df.loc[train_idx, "label"].astype(int)
    val_y   = feat_df.loc[val_idx,   "label"].astype(int)

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(train_X)

    # Train AUC
    log_train_probs = log_model.predict_proba(scaler.transform(train_X))[:, 1]
    log_val_probs   = log_model.predict_proba(scaler.transform(val_X))[:, 1]
    log_train_auc   = float(roc_auc_score(train_y, log_train_probs))
    log_val_auc     = float(roc_auc_score(val_y,   log_val_probs))

    xgb_val_probs = None
    xgb_val_auc = None
    if xgb_model is not None:
        xgb_val_probs = xgb_model.predict_proba(val_X)[:, 1]
        xgb_train_probs = xgb_model.predict_proba(train_X)[:, 1]
        xgb_val_auc = float(roc_auc_score(val_y, xgb_val_probs))
        xgb_train_auc = float(roc_auc_score(train_y, xgb_train_probs))

    # Permutation drops
    active_name = best_model.lower()
    active_obj  = xgb_model if active_name == "xgboost" and xgb_model else log_model
    active_sclr = None if active_name == "xgboost" else scaler
    active_nm   = "xgboost" if active_name == "xgboost" and xgb_model else "logistic"
    print(f"[theme_09] Computing permutation drops for {active_nm} ({len(feat_cols)} features)...")
    perm_df = compute_permutation_drop(active_nm, active_obj, val_X, val_y, feat_cols, active_sclr)

    # SHAP
    shap_df = compute_shap(xgb_model, val_X, feat_cols)

    # Combined diagnostic
    diag_df = perm_df.copy()
    if not shap_df.empty:
        diag_df = diag_df.merge(shap_df[["feature","mean_abs_shap"]], on="feature", how="left")
    else:
        diag_df["mean_abs_shap"] = 0.0

    # Load goodness from theme_07 if available
    goodness_fp = OUT / "feature_goodness.csv"
    if goodness_fp.exists():
        good_df = pd.read_csv(goodness_fp)[["feature","goodness_score","status"]].copy()
        diag_df = diag_df.merge(good_df, on="feature", how="left")
    else:
        diag_df["goodness_score"] = 0.5
        diag_df["status"] = "review"

    # Re-classify status using diagnostics
    good_med = float(diag_df["goodness_score"].fillna(0.5).median())
    diag_df["status"] = "review"
    diag_df.loc[
        (diag_df["perm_auc_drop"] <= 0) &
        (diag_df["goodness_score"].fillna(0) < good_med),
        "status"
    ] = "drop_candidate"
    diag_df.loc[
        (diag_df["perm_auc_drop"] >= 0.002) |
        (diag_df["goodness_score"].fillna(0) >= 0.60),
        "status"
    ] = "keep"

    keeps = (diag_df["status"] == "keep").sum()
    drops = (diag_df["status"] == "drop_candidate").sum()
    top_perm = diag_df.iloc[0]["feature"] if len(diag_df) > 0 else "—"

    kpis = (
        _kpi(f"{log_val_auc:.4f}", "Logistic val AUC")
      + _kpi(f"{log_train_auc - log_val_auc:.4f}", "Train-val gap<br>(overfitting)")
      + _kpi(str(keeps),           "Features → keep")
      + _kpi(str(drops),           "Features →<br>drop_candidate")
    )

    sections = []

    sections.append(_card("1. Train vs validation AUC — overfitting check", f"""
<div class="note">A large gap (>0.05) between train and validation AUC indicates overfitting. XGBoost is more prone than logistic regression.</div>
<div class="grid2">
  <div>{_fig_auc_gap(log_train_auc, log_val_auc, "Logistic")}</div>
  <div>{''+_fig_auc_gap(xgb_train_auc, xgb_val_auc, "XGBoost") if xgb_model else "<p style='color:#888'>XGBoost not available.</p>"}</div>
</div>
"""))

    sections.append(_card("2. Calibration curve — how well probabilities are calibrated", f"""
<div class="note">A well-calibrated model has probability outputs that match empirical frequencies. The diagonal = perfect calibration.</div>
{_fig_calibration(log_val_probs, xgb_val_probs, val_y.to_numpy())}
"""))

    sections.append(_card(f"3. Permutation AUC drop — {active_nm}", f"""
<div class="note">Each feature is shuffled independently. The drop in AUC shows how much the model relies on that feature.
Negative drop = feature adds noise; drop &lt;0.002 = near-zero contribution.</div>
{_fig_perm_drop(perm_df, active_nm)}
"""))

    sections.append(_card("4. XGBoost SHAP values", f"""
<div class="note">SHAP quantifies each feature's contribution to individual predictions (not just importance rank).
Mean |SHAP| = average absolute impact on log-odds across the validation set.</div>
{_fig_shap(shap_df)}
"""))

    sections.append(_card("5. Feature diagnostic scatter + full table", f"""
<div class="note">
<strong>Status rules:</strong> keep if perm_drop≥0.002 or goodness≥0.60;
drop_candidate if perm_drop≤0 and goodness&lt;median; review otherwise.
</div>
{_fig_status_scatter(diag_df)}
{_diag_table(diag_df)}
"""))

    diag_df.to_csv(OUT / "feature_diagnostics.csv", index=False)
    state["feature_diagnostics"] = diag_df
    print(f"[theme_09] Saved feature_diagnostics.csv  ({len(diag_df)} features)")

    html = _page(
        "Theme 09 — Model Diagnostics",
        "permutation AUC drop · SHAP · calibration · overfitting check · feature status",
        kpis, sections,
    )
    (OUT / "theme_09_diagnostics.html").write_text(html, encoding="utf-8")
    print(f"[theme_09] Wrote {OUT}/theme_09_diagnostics.html")
    return state


if __name__ == "__main__":
    run({})
    print("[theme_09] Done.")
