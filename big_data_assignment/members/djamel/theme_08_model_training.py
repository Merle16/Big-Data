#!/usr/bin/env python3
"""
Theme 08 — Model Training
==========================
Shows:
  • Logistic Regression vs XGBoost — ROC curves, confusion matrices
  • Validation AUC / accuracy comparison
  • XGBoost gain importance (top 20 features)
  • Logistic regression coefficients (top 20 by absolute value)
  • Score distribution by class (calibration proxy)
  • Model selection decision

Reads  : outputs_restart/features_train_prepped.csv
Writes : outputs_restart/theme_08_model_training.html
         outputs_restart/model_results.csv
         outputs_restart/threshold_analysis.csv
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
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              roc_auc_score, roc_curve)
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
FIG_DIR = OUT / "figures" / "theme_08"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
Y   = "#F5C518"; B = "#1848f5"; BG = "#0a0a0a"; CRD = "#141414"
TXT = "#ffffff"; MUT = "#888888"; GRN = "#2ecc71"; RED = "#e74c3c"; ORG = "#f39c12"
_FIG_COUNTER = 0

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
table{{width:100%;border-collapse:collapse;font-size:.85rem}}
th{{background:{Y};color:#111;padding:8px 10px;text-align:left}}
td{{padding:7px 10px;border-bottom:1px solid #222;vertical-align:top}}
tr:hover td{{background:#1e1e1e}}
img{{max-width:100%;border-radius:8px;margin-top:8px}}
.note{{background:#1a1a2e;border-left:4px solid {Y};padding:10px 14px;
       border-radius:4px;font-size:.85rem;margin:10px 0;color:#ccc}}
.green{{color:{GRN};font-weight:600}} .red{{color:{RED};font-weight:600}}
"""


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

def _img(fig): return f'<img src="data:image/png;base64,{_b64(fig)}">'
def _card(title, body, color=Y): return f'<div class="card"><h2 style="color:{color}">{title}</h2>{body}</div>'
def _kpi(val, lbl): return f'<div class="kpi"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>'
def _page(title, subtitle, kpis, sections):
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{title}</title><style>{CSS}</style></head><body>
<h1>{title}</h1><p class="subtitle">{subtitle}</p>
<div class="kpi-grid">{kpis}</div>{"".join(sections)}</body></html>"""


def _fig_roc(log_probs, xgb_probs, y_val) -> str:
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
    ax.set_facecolor(CRD)
    ax.plot([0, 1], [0, 1], color=MUT, linestyle="--", linewidth=1)

    fpr, tpr, _ = roc_curve(y_val, log_probs)
    log_auc = roc_auc_score(y_val, log_probs)
    ax.plot(fpr, tpr, color=B, linewidth=2.5, label=f"Logistic  AUC={log_auc:.4f}")

    if xgb_probs is not None:
        fpr2, tpr2, _ = roc_curve(y_val, xgb_probs)
        xgb_auc = roc_auc_score(y_val, xgb_probs)
        ax.plot(fpr2, tpr2, color=Y, linewidth=2.5, label=f"XGBoost   AUC={xgb_auc:.4f}")

    ax.set_xlabel("False Positive Rate", color=TXT)
    ax.set_ylabel("True Positive Rate", color=TXT)
    ax.set_title("ROC curves — validation set", color=TXT, fontsize=12)
    ax.legend(facecolor=CRD, edgecolor=MUT, labelcolor=TXT)
    ax.tick_params(colors=TXT)
    for sp in ax.spines.values(): sp.set_color(MUT)
    fig.tight_layout()
    return _img(fig)


def _fig_confusion(probs, y_val, threshold=0.5, title="Confusion matrix") -> str:
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_val, preds)
    fig, ax = plt.subplots(figsize=(4, 3.5), facecolor=BG)
    ax.set_facecolor(CRD)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"], color=TXT)
    ax.set_yticklabels(["True 0", "True 1"], color=TXT)
    ax.set_title(title, color=TXT, fontsize=10)
    for i in range(2):
        for j in range(2):
            # Force readable labels on both light and dark heatmap cells.
            cell_color = im.cmap(im.norm(cm[i, j]))
            luminance = 0.299 * cell_color[0] + 0.587 * cell_color[1] + 0.114 * cell_color[2]
            text_color = BG if luminance > 0.62 else TXT
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=text_color, fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return _img(fig)


def _fig_score_dist(log_probs, xgb_probs, y_val) -> str:
    fig, axes = plt.subplots(1, 2 if xgb_probs is not None else 1,
                              figsize=(12 if xgb_probs is not None else 6, 3.5), facecolor=BG)
    if xgb_probs is None:
        axes = [axes]
    pairs = [(log_probs, "Logistic", B)]
    if xgb_probs is not None:
        pairs.append((xgb_probs, "XGBoost", Y))
    for ax, (probs, name, color) in zip(axes, pairs):
        ax.set_facecolor(CRD)
        ax.hist(probs[y_val == 0], bins=40, color=RED, alpha=0.6, label="label=0", density=True)
        ax.hist(probs[y_val == 1], bins=40, color=GRN, alpha=0.6, label="label=1", density=True)
        ax.axvline(0.5, color=MUT, linestyle="--", linewidth=1)
        ax.set_title(f"{name} — score distribution by class", color=TXT, fontsize=10)
        ax.legend(facecolor=CRD, edgecolor=MUT, labelcolor=TXT, fontsize=8)
        ax.tick_params(colors=TXT, labelsize=8)
        for sp in ax.spines.values(): sp.set_color(MUT)
    fig.tight_layout()
    return _img(fig)


def _fig_xgb_importance(xgb_model, feat_cols: List[str]) -> str:
    booster = xgb_model.get_booster()
    score_map = booster.get_score(importance_type="gain")

    def _resolve(k):
        if k in feat_cols: return k
        if k.startswith("f") and k[1:].isdigit():
            idx = int(k[1:])
            if 0 <= idx < len(feat_cols): return feat_cols[idx]
        return k

    imp = pd.Series({_resolve(k): v for k, v in score_map.items()}).reindex(feat_cols, fill_value=0)
    imp = imp.sort_values(ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(8, max(4, len(imp) * 0.4)), facecolor=BG)
    ax.set_facecolor(CRD)
    ax.barh(imp.index.tolist(), imp.values, color=Y, alpha=0.85)
    ax.set_xlabel("Gain importance", color=TXT)
    ax.set_title("XGBoost feature importance — gain (top 20)", color=TXT, fontsize=12)
    ax.tick_params(colors=TXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_color(MUT)
    fig.tight_layout()
    return _img(fig)


def _fig_logistic_coefs(log_model: LogisticRegression, feat_cols: List[str]) -> str:
    coefs = pd.Series(log_model.coef_[0], index=feat_cols)
    top = coefs.abs().nlargest(20).index
    coefs_top = coefs[top].sort_values()
    colors = [GRN if v > 0 else RED for v in coefs_top.values]
    fig, ax = plt.subplots(figsize=(8, max(4, len(coefs_top) * 0.4)), facecolor=BG)
    ax.set_facecolor(CRD)
    ax.barh(coefs_top.index.tolist(), coefs_top.values, color=colors, alpha=0.85)
    ax.axvline(0, color=MUT, linewidth=1)
    ax.set_xlabel("Coefficient (standardized features)", color=TXT)
    ax.set_title("Logistic regression coefficients (top 20 by |coef|)", color=TXT, fontsize=12)
    ax.tick_params(colors=TXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_color(MUT)
    fig.tight_layout()
    return _img(fig)


def _fig_model_comparison(log_auc, xgb_auc, log_acc, xgb_acc) -> str:
    metrics = ["ROC-AUC", "Accuracy"]
    log_vals = [log_auc, log_acc]
    xgb_vals = [xgb_auc, xgb_acc] if xgb_auc is not None else None
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=BG)
    ax.set_facecolor(CRD)
    ax.bar(x - 0.2, log_vals, width=0.35, color=B, alpha=0.9, label="Logistic")
    if xgb_vals:
        ax.bar(x + 0.2, xgb_vals, width=0.35, color=Y, alpha=0.9, label="XGBoost")
    ax.set_xticks(x); ax.set_xticklabels(metrics, color=TXT)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", color=TXT)
    ax.set_title("Model comparison — validation set", color=TXT, fontsize=12)
    ax.legend(facecolor=CRD, edgecolor=MUT, labelcolor=TXT)
    ax.tick_params(colors=TXT)
    for sp in ax.spines.values(): sp.set_color(MUT)
    for i, v in enumerate(log_vals):
        ax.text(i - 0.2, v + 0.01, f"{v:.3f}", ha="center", color=TXT, fontsize=9)
    if xgb_vals:
        for i, v in enumerate(xgb_vals):
            ax.text(i + 0.2, v + 0.01, f"{v:.3f}", ha="center", color=TXT, fontsize=9)
    fig.tight_layout()
    return _img(fig)


def _youden_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j = tpr - fpr
    finite = np.isfinite(thresholds)
    if finite.any():
        idx = int(np.argmax(np.where(finite, j, -np.inf)))
    else:
        idx = int(np.argmax(j))
    threshold = thresholds[idx]
    if not np.isfinite(threshold):
        threshold = 0.5
    return float(np.clip(threshold, 0.0, 1.0)), float(j[idx])


def _threshold_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "youden_j": float(sensitivity + specificity - 1.0),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def _fig_threshold_sweep(y_true: np.ndarray, probs: np.ndarray, model_name: str,
                         youden_thr: float) -> str:
    thresholds = np.linspace(0.01, 0.99, 99)
    sensitivity = []
    specificity = []
    f1_vals = []
    youden_vals = []
    for thr in thresholds:
        m = _threshold_metrics(y_true, probs, float(thr))
        sensitivity.append(m["sensitivity"])
        specificity.append(m["specificity"])
        f1_vals.append(m["f1"])
        youden_vals.append(m["youden_j"])

    fig, ax = plt.subplots(figsize=(6.5, 3.8), facecolor=BG)
    ax.set_facecolor(CRD)
    ax.plot(thresholds, sensitivity, color=GRN, linewidth=2, label="Sensitivity")
    ax.plot(thresholds, specificity, color=ORG, linewidth=2, label="Specificity")
    ax.plot(thresholds, f1_vals, color=B, linewidth=2, label="F1")
    ax.plot(thresholds, youden_vals, color=Y, linewidth=2, label="Youden J")
    ax.axvline(0.5, color=MUT, linestyle="--", linewidth=1.4, label="Threshold 0.5")
    ax.axvline(youden_thr, color=RED, linestyle="--", linewidth=1.4,
               label=f"Youden {youden_thr:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Threshold", color=TXT)
    ax.set_ylabel("Metric value", color=TXT)
    ax.set_title(f"{model_name} — threshold effect on validation metrics", color=TXT, fontsize=11)
    ax.tick_params(colors=TXT)
    for sp in ax.spines.values():
        sp.set_color(MUT)
    ax.legend(facecolor=CRD, edgecolor=MUT, labelcolor=TXT, fontsize=8, ncol=2, loc="lower center")
    fig.tight_layout()
    return _img(fig)


def run(state: dict) -> dict:
    global _FIG_COUNTER
    _FIG_COUNTER = 0
    for p in FIG_DIR.glob("*.png"):
        p.unlink()

    feat_df = state.get("features_train_prepped")
    if feat_df is None:
        for fname in ["features_train_prepped.csv", "features_train.csv"]:
            fp = OUT / fname
            if fp.exists():
                feat_df = pd.read_csv(fp)
                break
    if feat_df is None:
        raise FileNotFoundError("features_train_prepped.csv not found — run theme_06 first.")

    for col in feat_df.columns:
        if col not in ("tconst", "primaryTitle", "canonical_title"):
            feat_df[col] = pd.to_numeric(feat_df[col], errors="coerce")

    if "label" not in feat_df.columns:
        raise ValueError("label column missing.")

    label_num = pd.to_numeric(feat_df["label"], errors="coerce")
    dropped = int(label_num.isna().sum())
    if dropped:
        print(f"[theme_08] Dropping {dropped} rows with missing/non-numeric label before split.")
    feat_df = feat_df.loc[label_num.notna()].copy()
    feat_df["label"] = label_num.loc[label_num.notna()].astype(int).to_numpy()

    feat_cols = [c for c in feat_df.columns if c not in ("tconst", "label", "primaryTitle", "canonical_title")]
    feat_cols = [c for c in feat_cols if feat_df[c].notna().sum() > 0]

    # Impute any remaining NaN with column median
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
    y_val_np = val_y.to_numpy()

    # Scale for logistic
    scaler = StandardScaler()
    train_Xs = scaler.fit_transform(train_X)
    val_Xs   = scaler.transform(val_X)

    # Logistic
    print("[theme_08] Fitting logistic regression...")
    log_model = LogisticRegression(max_iter=2000, random_state=SEED)
    log_model.fit(train_Xs, train_y)
    log_probs = log_model.predict_proba(val_Xs)[:, 1]
    log_auc   = float(roc_auc_score(val_y, log_probs))
    log_acc   = float(accuracy_score(val_y, (log_probs >= 0.5).astype(int)))

    # XGBoost
    xgb_model = None; xgb_probs = None; xgb_auc = None; xgb_acc = None
    if HAS_XGB:
        print("[theme_08] Fitting XGBoost...")
        xgb_model = XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", n_jobs=4, random_state=SEED,
            verbosity=0,
        )
        xgb_model.fit(train_X, train_y, verbose=False)
        xgb_probs = xgb_model.predict_proba(val_X)[:, 1]
        xgb_auc   = float(roc_auc_score(val_y, xgb_probs))
        xgb_acc   = float(accuracy_score(val_y, (xgb_probs >= 0.5).astype(int)))

    # Threshold analysis (0.5 vs Youden-J optimum on validation set)
    log_thr_youden, log_youden_j = _youden_threshold(y_val_np, log_probs)
    log_m_05 = _threshold_metrics(y_val_np, log_probs, 0.5)
    log_m_youden = _threshold_metrics(y_val_np, log_probs, log_thr_youden)

    xgb_thr_youden = None
    xgb_youden_j = None
    xgb_m_05 = None
    xgb_m_youden = None
    if xgb_probs is not None:
        xgb_thr_youden, xgb_youden_j = _youden_threshold(y_val_np, xgb_probs)
        xgb_m_05 = _threshold_metrics(y_val_np, xgb_probs, 0.5)
        xgb_m_youden = _threshold_metrics(y_val_np, xgb_probs, xgb_thr_youden)

    best = "XGBoost" if (xgb_auc is not None and xgb_auc >= log_auc) else "Logistic"
    best_auc = xgb_auc if best == "XGBoost" else log_auc

    kpis = (
        _kpi(f"{log_auc:.4f}",  "Logistic AUC<br>(validation)")
      + _kpi(f"{xgb_auc:.4f}" if xgb_auc else "N/A", "XGBoost AUC<br>(validation)")
      + _kpi(best,              "Selected model<br>(best AUC)")
      + _kpi(f"{best_auc:.4f}","Best model AUC")
    )

    sections = []

    # 1. ROC curves
    sections.append(_card("1. ROC curves — validation set", f"""
<div class="note">ROC-AUC measures ranking quality: probability that a random positive is ranked above a random negative.</div>
{_fig_roc(log_probs, xgb_probs, y_val_np)}
"""))

    # 2. Confusion matrices
    cm_html = f"""
<div class="grid2">
  <div><h3>Logistic (threshold=0.5)</h3>{_fig_confusion(log_probs, y_val_np, title=f"Logistic  AUC={log_auc:.4f}")}</div>
"""
    if xgb_probs is not None:
        cm_html += f'  <div><h3>XGBoost (threshold=0.5)</h3>{_fig_confusion(xgb_probs, y_val_np, title=f"XGBoost  AUC={xgb_auc:.4f}")}</div>'
    cm_html += "</div>"
    sections.append(_card("2. Confusion matrices (validation set, threshold=0.5)", cm_html))

    # 3. Model comparison bar
    sections.append(_card("3. Model comparison summary", f"""
<div class="grid2">
  <div>{_fig_model_comparison(log_auc, xgb_auc, log_acc, xgb_acc)}</div>
  <div>
    <table>
    <tr><th>Metric</th><th>Logistic</th><th>XGBoost</th></tr>
    <tr><td>ROC-AUC (val)</td><td>{log_auc:.4f}</td><td>{"—" if xgb_auc is None else f"{xgb_auc:.4f}"}</td></tr>
    <tr><td>Accuracy (val)</td><td>{log_acc:.4f}</td><td>{"—" if xgb_acc is None else f"{xgb_acc:.4f}"}</td></tr>
    <tr><td><strong>Selected</strong></td>
        <td style="color:{"#2ecc71" if best=="Logistic" else "#888"}">{"✅ YES" if best=="Logistic" else "—"}</td>
        <td style="color:{"#2ecc71" if best=="XGBoost"  else "#888"}">{"✅ YES" if best=="XGBoost"  else "—"}</td>
    </tr>
    </table>
    <div class="note" style="margin-top:12px">
    <strong>Selection rule:</strong> XGBoost selected if its AUC ≥ Logistic AUC.
    XGBoost typically wins due to ability to model non-linear interactions between features
    like director_hit_rate × numVotes_log1p.
    </div>
  </div>
</div>
"""))

    # 4. Score distributions
    sections.append(_card("4. Score distributions by class (calibration proxy)", f"""
<div class="note">Separation between classes in score space indicates good calibration. Overlap near 0.5 is the uncertain region.</div>
{_fig_score_dist(log_probs, xgb_probs, y_val_np)}
"""))

    # 5. Threshold effects: default 0.5 vs Youden-J
    threshold_rows = [
        {
            "model": "logistic",
            "rule": "fixed_0.5",
            **log_m_05,
            "youden_threshold": log_thr_youden,
            "youden_j_at_optimum": log_youden_j,
        },
        {
            "model": "logistic",
            "rule": "youden_opt",
            **log_m_youden,
            "youden_threshold": log_thr_youden,
            "youden_j_at_optimum": log_youden_j,
        },
    ]
    if xgb_m_05 is not None and xgb_m_youden is not None and xgb_thr_youden is not None and xgb_youden_j is not None:
        threshold_rows.extend([
            {
                "model": "xgboost",
                "rule": "fixed_0.5",
                **xgb_m_05,
                "youden_threshold": xgb_thr_youden,
                "youden_j_at_optimum": xgb_youden_j,
            },
            {
                "model": "xgboost",
                "rule": "youden_opt",
                **xgb_m_youden,
                "youden_threshold": xgb_thr_youden,
                "youden_j_at_optimum": xgb_youden_j,
            },
        ])
    threshold_df = pd.DataFrame(threshold_rows)
    threshold_show = threshold_df.copy()
    for c in ["threshold", "accuracy", "f1", "precision", "sensitivity", "specificity",
              "youden_j", "youden_threshold", "youden_j_at_optimum"]:
        threshold_show[c] = threshold_show[c].astype(float).round(4)

    threshold_fig_html = f"""
<div class="grid2">
  <div><h3>Logistic threshold sweep</h3>{_fig_threshold_sweep(y_val_np, log_probs, "Logistic", log_thr_youden)}</div>
"""
    if xgb_probs is not None and xgb_thr_youden is not None:
        threshold_fig_html += f'  <div><h3>XGBoost threshold sweep</h3>{_fig_threshold_sweep(y_val_np, xgb_probs, "XGBoost", xgb_thr_youden)}</div>'
    threshold_fig_html += "</div>"

    threshold_html = f"""
<div class="note">
<strong>Why not always 0.5?</strong> 0.5 is a default cut-off. Youden threshold maximizes
<code>sensitivity + specificity - 1</code> on validation and can improve operational decisions.
</div>
{threshold_fig_html}
<h3>Threshold effect table (validation)</h3>
{threshold_show.to_html(index=False, border=0)}
"""
    sections.append(_card("5. Threshold selection — fixed 0.5 vs Youden-J optimum", threshold_html))

    # 6. Feature importances
    imp_html = ""
    if xgb_model is not None:
        imp_html += f"<h3>XGBoost gain importance</h3>{_fig_xgb_importance(xgb_model, feat_cols)}"
    imp_html += f"<h3>Logistic regression coefficients</h3>{_fig_logistic_coefs(log_model, feat_cols)}"
    sections.append(_card("6. Feature importances", imp_html))

    # Save model artifacts
    model_results = pd.DataFrame([{
        "model": "logistic",
        "validation_auc": log_auc,
        "validation_accuracy": log_acc,
        "threshold_fixed": 0.5,
        "threshold_youden": log_thr_youden,
        "accuracy_youden": log_m_youden["accuracy"],
        "f1_fixed": log_m_05["f1"],
        "f1_youden": log_m_youden["f1"],
        "selected": best == "Logistic",
    }])
    if xgb_auc is not None:
        model_results = pd.concat([model_results, pd.DataFrame([{
            "model": "xgboost",
            "validation_auc": xgb_auc,
            "validation_accuracy": xgb_acc,
            "threshold_fixed": 0.5,
            "threshold_youden": xgb_thr_youden,
            "accuracy_youden": None if xgb_m_youden is None else xgb_m_youden["accuracy"],
            "f1_fixed": None if xgb_m_05 is None else xgb_m_05["f1"],
            "f1_youden": None if xgb_m_youden is None else xgb_m_youden["f1"],
            "selected": best == "XGBoost",
        }])], ignore_index=True)
    model_results.to_csv(OUT / "model_results.csv", index=False)
    threshold_df.to_csv(OUT / "threshold_analysis.csv", index=False)

    with open(OUT / "models.pkl", "wb") as f:
        pickle.dump({
            "log_model": log_model,
            "xgb_model": xgb_model,
            "scaler": scaler,
            "feat_cols": feat_cols,
            "best_model": best,
            "thresholds": {
                "logistic": {"fixed_0_5": 0.5, "youden": log_thr_youden},
                "xgboost": None if xgb_thr_youden is None else {"fixed_0_5": 0.5, "youden": xgb_thr_youden},
            },
            "train_idx": list(train_idx),
            "val_idx":   list(val_idx),
        }, f)

    state["log_model"] = log_model
    state["xgb_model"] = xgb_model
    state["scaler"]    = scaler
    state["feat_cols"] = feat_cols
    state["best_model"] = best
    state["threshold_analysis"] = threshold_df
    state["train_idx"] = list(train_idx)
    state["val_idx"]   = list(val_idx)
    state["log_auc"]   = log_auc
    state["xgb_auc"]   = xgb_auc
    print(f"[theme_08] Logistic AUC={log_auc:.4f}  XGB AUC={xgb_auc}  Best={best}")

    html = _page(
        "Theme 08 — Model Training",
        "logistic regression · XGBoost · ROC curves · confusion matrices · feature importance",
        kpis, sections,
    )
    (OUT / "theme_08_model_training.html").write_text(html, encoding="utf-8")
    print(f"[theme_08] Wrote {OUT}/theme_08_model_training.html")
    return state


if __name__ == "__main__":
    run({})
    print("[theme_08] Done.")
