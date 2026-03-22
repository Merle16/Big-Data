#!/usr/bin/env python3
"""
Theme 10 — Reduced Model Selection
=====================================
Shows:
  • Which features are dropped (keep-set selection from diagnostics)
  • Full model vs reduced model AUC comparison
  • Performance curve: AUC vs number of features kept (ablation)
  • Final keep-set features with justification
  • Retrain on keep-set and compare ROC curves

Reads  : outputs_restart/features_train_prepped.csv
         outputs_restart/feature_diagnostics.csv
         outputs_restart/models.pkl
Writes : outputs_restart/theme_10_reduced_model.html
         outputs_restart/keep_set_features.txt
         outputs_restart/reduced_model.pkl
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
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


def _fit_eval(X_tr, y_tr, X_vl, y_vl, feat_cols, model_type="logistic") -> float:
    X_tr_sub = X_tr[feat_cols]
    X_vl_sub = X_vl[feat_cols]
    if model_type == "xgboost" and HAS_XGB:
        m = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                          subsample=0.85, colsample_bytree=0.85, random_state=SEED,
                          objective="binary:logistic", eval_metric="logloss",
                          tree_method="hist", n_jobs=4, verbosity=0)
        m.fit(X_tr_sub, y_tr.astype(int), verbose=False)
        probs = m.predict_proba(X_vl_sub)[:, 1]
    else:
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr_sub)
        X_vl_sc = sc.transform(X_vl_sub)
        m = LogisticRegression(max_iter=2000, random_state=SEED)
        m.fit(X_tr_sc, y_tr.astype(int))
        probs = m.predict_proba(X_vl_sc)[:, 1]
    return float(roc_auc_score(y_vl.astype(int), probs))


def _fig_ablation(X_tr, y_tr, X_vl, y_vl, feat_cols_ranked, model_type) -> str:
    """AUC vs number of features kept (greedy top-down by diagnostic rank)."""
    steps = list(range(1, min(len(feat_cols_ranked) + 1, 26)))
    aucs  = []
    for n in steps:
        sub = feat_cols_ranked[:n]
        try:
            auc = _fit_eval(X_tr, y_tr, X_vl, y_vl, sub, model_type)
        except Exception:
            auc = float("nan")
        aucs.append(auc)
        if n % 5 == 0:
            print(f"[theme_10] ablation n={n}  AUC={auc:.4f}")

    fig, ax = plt.subplots(figsize=(9, 4), facecolor=BG)
    ax.set_facecolor(CRD)
    ax.plot(steps, aucs, color=Y, marker="o", markersize=5, linewidth=2)
    best_n = steps[int(np.nanargmax(aucs))]
    best_auc = float(np.nanmax(aucs))
    ax.axvline(best_n, color=GRN, linestyle="--", linewidth=1.5)
    ax.text(best_n + 0.3, min(aucs) + 0.001, f"Best: n={best_n}\nAUC={best_auc:.4f}",
            color=GRN, fontsize=9)
    ax.set_xlabel("Number of features (top-N by diagnostic rank)", color=TXT)
    ax.set_ylabel("Validation AUC", color=TXT)
    ax.set_title(f"Ablation curve — {model_type}: AUC vs feature count", color=TXT, fontsize=12)
    ax.tick_params(colors=TXT)
    for sp in ax.spines.values(): sp.set_color(MUT)
    fig.tight_layout()
    return _img(fig), best_n, best_auc, aucs


def _fig_roc_compare(full_probs, red_probs, y_val) -> str:
    y = y_val.astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
    ax.set_facecolor(CRD)
    ax.plot([0, 1], [0, 1], color=MUT, linestyle="--", linewidth=1)
    fpr1, tpr1, _ = roc_curve(y, full_probs)
    ax.plot(fpr1, tpr1, color=B, linewidth=2.5, label=f"Full model  AUC={roc_auc_score(y, full_probs):.4f}")
    fpr2, tpr2, _ = roc_curve(y, red_probs)
    ax.plot(fpr2, tpr2, color=Y, linewidth=2.5, label=f"Reduced model  AUC={roc_auc_score(y, red_probs):.4f}")
    ax.set_xlabel("FPR", color=TXT); ax.set_ylabel("TPR", color=TXT)
    ax.set_title("ROC: full vs reduced model", color=TXT, fontsize=12)
    ax.legend(facecolor=CRD, edgecolor=MUT, labelcolor=TXT)
    ax.tick_params(colors=TXT)
    for sp in ax.spines.values(): sp.set_color(MUT)
    fig.tight_layout()
    return _img(fig)


def _fig_dropped_vs_kept(diag_df: pd.DataFrame, keep_set: List[str]) -> str:
    all_feats = diag_df["feature"].tolist()
    is_kept = [f in keep_set for f in all_feats]
    diag_df = diag_df.copy()
    diag_df["in_keep_set"] = is_kept
    drop_df = diag_df[~diag_df["in_keep_set"]].sort_values("perm_auc_drop", ascending=True)
    keep_df = diag_df[ diag_df["in_keep_set"]].sort_values("perm_auc_drop", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, max(len(drop_df), len(keep_df)) * 0.35 + 1)), facecolor=BG)
    for ax, df, title, color in zip(axes,
                                     [drop_df, keep_df],
                                     ["Dropped features (perm AUC drop)", "Kept features (perm AUC drop)"],
                                     [RED, GRN]):
        ax.set_facecolor(CRD)
        if len(df) > 0:
            ax.barh(df["feature"], df["perm_auc_drop"].clip(lower=-0.005), color=color, alpha=0.8)
            ax.axvline(0, color=MUT, linewidth=1)
        ax.set_title(title, color=TXT, fontsize=10)
        ax.tick_params(colors=TXT, labelsize=7)
        for sp in ax.spines.values(): sp.set_color(MUT)
    fig.tight_layout()
    return _img(fig)


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

    # Load diagnostics
    diag_df = state.get("feature_diagnostics")
    if diag_df is None:
        fp = OUT / "feature_diagnostics.csv"
        if fp.exists():
            diag_df = pd.read_csv(fp)
    if diag_df is None:
        fp2 = OUT / "feature_goodness.csv"
        if fp2.exists():
            diag_df = pd.read_csv(fp2)

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

    if "label" not in feat_df.columns:
        raise ValueError("label column missing.")

    label_num = pd.to_numeric(feat_df["label"], errors="coerce")
    dropped = int(label_num.isna().sum())
    if dropped:
        print(f"[theme_10] Dropping {dropped} rows with missing/non-numeric label before split.")
    feat_df = feat_df.loc[label_num.notna()].copy()
    feat_df["label"] = label_num.loc[label_num.notna()].astype(int).to_numpy()

    feat_cols = [c for c in (feat_cols or []) if c in feat_df.columns]
    if not feat_cols:
        feat_cols = [c for c in feat_df.columns if c not in ("tconst","label","primaryTitle","canonical_title")]

    for col in feat_cols:
        med = float(feat_df[col].median()) if feat_df[col].notna().sum() > 0 else 0.0
        feat_df[col] = feat_df[col].fillna(med)

    train_idx, val_idx = train_test_split(
        feat_df.index, test_size=0.20, random_state=SEED, stratify=feat_df["label"].astype(int)
    )
    X_tr = feat_df.loc[train_idx, feat_cols]
    X_vl = feat_df.loc[val_idx,   feat_cols]
    y_tr = feat_df.loc[train_idx, "label"]
    y_vl = feat_df.loc[val_idx,   "label"]

    # Full model predictions
    active_type = "xgboost" if best_model.lower() == "xgboost" and xgb_model else "logistic"
    if scaler is None:
        scaler = StandardScaler(); scaler.fit(X_tr)
    if active_type == "xgboost":
        full_probs = xgb_model.predict_proba(X_vl)[:, 1]
    else:
        full_probs = log_model.predict_proba(scaler.transform(X_vl))[:, 1]
    full_auc = float(roc_auc_score(y_vl.astype(int), full_probs))

    # Determine keep-set from diagnostics
    if diag_df is not None and "feature" in diag_df.columns:
        # Sort by perm_auc_drop descending if available, else goodness_score
        sort_col = "perm_auc_drop" if "perm_auc_drop" in diag_df.columns else "goodness_score"
        diag_sorted = diag_df.sort_values(sort_col, ascending=False)
        feats_ranked = [f for f in diag_sorted["feature"].tolist() if f in feat_cols]
        # keep = status=="keep" OR (status!="drop_candidate" AND drop>0)
        if "status" in diag_df.columns:
            keep_set = [f for f in feats_ranked
                        if diag_df.loc[diag_df["feature"]==f, "status"].values[0] != "drop_candidate"]
        else:
            keep_set = feats_ranked[:max(5, len(feats_ranked)//2)]
    else:
        feats_ranked = feat_cols
        keep_set = feat_cols

    keep_set = [f for f in keep_set if f in feat_cols]
    if len(keep_set) < 3:
        keep_set = feat_cols  # safety

    # Ablation curve
    print(f"[theme_10] Running ablation ({len(feats_ranked)} features)...")
    abl_img, best_n, best_auc, abl_aucs = _fig_ablation(X_tr, y_tr, X_vl, y_vl, feats_ranked, active_type)

    # Use best_n from ablation as keep-set size (if better than status-based)
    ablation_set = feats_ranked[:best_n]
    final_keep = ablation_set if len(ablation_set) >= 2 else keep_set

    # Retrain reduced model
    print(f"[theme_10] Retraining reduced model on {len(final_keep)} features...")
    if active_type == "xgboost" and HAS_XGB:
        red_model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                   subsample=0.85, colsample_bytree=0.85, random_state=SEED,
                                   objective="binary:logistic", eval_metric="logloss",
                                   tree_method="hist", n_jobs=4, verbosity=0)
        red_model.fit(X_tr[final_keep], y_tr.astype(int), verbose=False)
        red_probs = red_model.predict_proba(X_vl[final_keep])[:, 1]
        red_scaler = None
    else:
        red_scaler = StandardScaler()
        X_tr_red = red_scaler.fit_transform(X_tr[final_keep])
        X_vl_red = red_scaler.transform(X_vl[final_keep])
        red_model = LogisticRegression(max_iter=2000, random_state=SEED)
        red_model.fit(X_tr_red, y_tr.astype(int))
        red_probs = red_model.predict_proba(X_vl_red)[:, 1]
    red_auc = float(roc_auc_score(y_vl.astype(int), red_probs))
    auc_delta = red_auc - full_auc

    kpis = (
        _kpi(str(len(feat_cols)), "Full feature<br>count")
      + _kpi(str(len(final_keep)),"Reduced feature<br>count")
      + _kpi(f"{full_auc:.4f}",  "Full model AUC")
      + _kpi(f"{red_auc:.4f}",   f"Reduced AUC<br>({'+'if auc_delta>=0 else ''}{auc_delta:.4f})")
    )

    sections = []

    sections.append(_card("1. Ablation curve — AUC vs number of features", f"""
<div class="note">
Features are added in diagnostic-score order (best first). The peak shows the optimal keep-set size
before additional weak features start to hurt or add noise.
</div>
{abl_img}
"""))

    sections.append(_card("2. Full vs reduced model ROC comparison", f"""
<div class="note">
<strong>Reduced model uses {len(final_keep)} features</strong> (selected at ablation peak n={best_n}).
AUC delta = {auc_delta:+.4f}.
A well-reduced model sacrifices minimal AUC for interpretability and inference speed.
</div>
{_fig_roc_compare(full_probs, red_probs, y_vl)}
"""))

    sections.append(_card("3. Dropped vs kept features", f"""
{_fig_dropped_vs_kept(diag_df if diag_df is not None else pd.DataFrame({"feature": feat_cols, "perm_auc_drop": 0}), final_keep)}
"""))

    # Keep-set table
    keep_rows = ""
    for feat in final_keep:
        perm = ""
        if diag_df is not None and "perm_auc_drop" in diag_df.columns:
            v = diag_df.loc[diag_df["feature"] == feat, "perm_auc_drop"]
            if len(v) > 0:
                perm = f"{float(v.iloc[0]):.5f}"
        status = ""
        if diag_df is not None and "status" in diag_df.columns:
            s = diag_df.loc[diag_df["feature"] == feat, "status"]
            if len(s) > 0:
                status = s.iloc[0]
        sc = {"keep": GRN, "review": ORG, "drop_candidate": RED}.get(status, TXT)
        keep_rows += f"<tr><td><code>{feat}</code></td><td>{perm}</td><td style='color:{sc}'>{status}</td></tr>"
    sections.append(_card("4. Final keep-set feature list", f"""
<table>
<tr><th>Feature</th><th>Perm AUC drop</th><th>Status</th></tr>
{keep_rows}
</table>
"""))

    # Save
    (OUT / "keep_set_features.txt").write_text("\n".join(final_keep), encoding="utf-8")
    with open(OUT / "reduced_model.pkl", "wb") as f:
        pickle.dump({"model": red_model, "scaler": red_scaler, "feat_cols": final_keep,
                     "model_type": active_type}, f)
    state["keep_set"] = final_keep
    state["red_model"] = red_model
    state["red_auc"]   = red_auc
    print(f"[theme_10] Full AUC={full_auc:.4f}  Reduced AUC={red_auc:.4f}  Keep={len(final_keep)} features")

    html = _page(
        "Theme 10 — Reduced Model Selection",
        "ablation curve · full vs reduced ROC · keep-set selection · AUC preservation",
        kpis, sections,
    )
    (OUT / "theme_10_reduced_model.html").write_text(html, encoding="utf-8")
    print(f"[theme_10] Wrote {OUT}/theme_10_reduced_model.html")
    return state


if __name__ == "__main__":
    run({})
    print("[theme_10] Done.")
