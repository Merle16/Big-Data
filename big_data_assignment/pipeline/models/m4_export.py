#!/usr/bin/env python3
"""
Step m4: Artifact export — predictions, manifest, summary.

Figures saved to data/processed/model_figures/:
  18_artifact_sizes.png
  19_pipeline_funnel.png   (if funnel data available)

Artifacts saved to data/processed/:
  predictions_val.csv      — tconst, predicted_prob, predicted_label, true_label
  pipeline_manifest.csv
"""
from __future__ import annotations

import datetime
import pickle
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
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

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]   # big_data_assignment/
PIPELINE   = ROOT / "pipeline"
OUT_FEAT   = PIPELINE / "outputs" / "features"
OUT_MODELS = PIPELINE / "outputs" / "models"
MODEL_FIG_DIR  = OUT_MODELS
OUT_MODELS.mkdir(parents=True, exist_ok=True)

# ── config ─────────────────────────────────────────────────────────────────────
try:
    from ..config import CFG as _CFG
except ImportError:
    import yaml as _yaml
    _cfg_path = ROOT / "config.yaml"
    _CFG: dict = _yaml.safe_load(_cfg_path.read_text(encoding="utf-8")) if _cfg_path.exists() else {}

# ── constants ──────────────────────────────────────────────────────────────────
SEED = _CFG.get("global", {}).get("seed", 42)
BG  = "#0a0a0a"
CRD = "#111111"
BDR = "#252525"
TXT = "#e8e8e8"
MUT = "#666666"
Y   = "#F5C518"
GRN = "#2ecc71"
RED = "#e74c3c"
ORG = "#f39c12"
BLU = "#1848f5"
B   = BLU  # legacy alias

# ── matplotlib dark theme ──────────────────────────────────────────────────────
mpl.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    CRD,
    "axes.edgecolor":    BDR,
    "axes.labelcolor":   TXT,
    "text.color":        TXT,
    "xtick.color":       MUT,
    "ytick.color":       MUT,
    "grid.color":        BDR,
    "grid.linewidth":    0.4,
    "legend.facecolor":  CRD,
    "legend.edgecolor":  BDR,
    "legend.fontsize":   9,
    "legend.framealpha": 0.15,
    "font.family":       "sans-serif",
})


# ── data helpers ───────────────────────────────────────────────────────────────

def _load_feat_df(state: dict) -> Optional[pd.DataFrame]:
    feat_df = state.get("features_train_prepped")
    if feat_df is None:
        for fname in ["features_train_prepped.parquet", "features_train_prepped.csv"]:
            fp = OUT_FEAT / fname
            if fp.exists():
                feat_df = pd.read_parquet(fp) if fname.endswith(".parquet") else pd.read_csv(fp)
                break
    if feat_df is not None:
        for col in feat_df.columns:
            if col not in ("tconst", "primaryTitle", "canonical_title"):
                feat_df[col] = pd.to_numeric(feat_df[col], errors="coerce")
    return feat_df


# ── manifest ───────────────────────────────────────────────────────────────────

def _build_manifest(out_dir: Path) -> pd.DataFrame:
    """List all files in out_dir with size, modification time, and extension."""
    rows = []
    for fp in sorted(out_dir.iterdir()):
        if fp.is_file():
            stat = fp.stat()
            rows.append({
                "file":      fp.name,
                "size_kb":   round(stat.st_size / 1024, 1),
                "modified":  datetime.datetime.fromtimestamp(stat.st_mtime)
                             .strftime("%Y-%m-%d %H:%M"),
                "type":      fp.suffix,
            })
    return pd.DataFrame(rows).sort_values("size_kb", ascending=False)


# ── save helper ────────────────────────────────────────────────────────────────

def _save(fig, fname: str) -> None:
    fig.savefig(MODEL_FIG_DIR / fname, dpi=130, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)


# ── figure functions ───────────────────────────────────────────────────────────

def _fig_manifest_sizes(manifest: pd.DataFrame) -> None:
    """Horizontal bar chart of the top-20 artifact sizes."""
    top    = manifest.head(20).sort_values("size_kb", ascending=True)
    colors_map = {".html": Y, ".csv": BLU, ".pkl": ORG, ".txt": GRN, ".png": MUT}
    clrs   = [colors_map.get(t, MUT) for t in top["type"]]

    n_items = len(top)
    fig, ax = plt.subplots(figsize=(14, max(6, n_items * 0.55)))
    bars = ax.barh(top["file"], top["size_kb"], color=clrs, alpha=0.88, edgecolor="none")
    x_max = ax.get_xlim()[1]
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005 * x_max, bar.get_y() + bar.get_height() / 2,
                f"{w} KB", va="center", ha="left", fontsize=8, color=TXT)
    ax.set_xlabel("File size (KB)", fontsize=11, labelpad=8)
    ax.set_title("Output artifact sizes (top 20)",
                 fontsize=13, fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    ax.tick_params(axis="y", labelsize=9, labelcolor=MUT)
    ax.tick_params(axis="x", labelsize=9, labelcolor=MUT)
    # legend patches
    for ext, col in colors_map.items():
        ax.barh([], [], color=col, label=ext)
    ax.legend()
    fig.tight_layout(pad=2.5)
    _save(fig, "18_artifact_sizes.png")


def _fig_pipeline_funnel(stages: List[tuple]) -> None:
    """Funnel chart showing how row/feature counts shrink through pipeline stages."""
    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]
    max_v  = max(values) if values else 1

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(labels))
    for i, (lbl, val) in enumerate(zip(labels, values)):
        w = max(0.1, val / max_v) * 0.7
        ax.barh(i, val, height=w, color=Y, alpha=max(0.4, 0.88 - i * 0.05), edgecolor="none")
        ax.text(val * 1.01, i, f"{val:,}", va="center",
                color=TXT, fontsize=9, fontweight="bold")
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Count", fontsize=11, labelpad=8)
    ax.set_title("Pipeline data funnel — rows / features through stages",
                 fontsize=13, fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    ax.tick_params(axis="y", labelsize=9, labelcolor=MUT)
    ax.tick_params(axis="x", labelsize=9, labelcolor=MUT)
    fig.tight_layout(pad=2.5)
    _save(fig, "19_pipeline_funnel.png")


# ── main entry point ───────────────────────────────────────────────────────────

def run(state: dict) -> dict:
    """Export predictions, build manifest, save figures, print summary."""
    # ── pull model artifacts from state or disk ────────────────────────────────
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
        pkl = OUT_MODELS / "models.pkl"
        if pkl.exists():
            with open(pkl, "rb") as f:
                art = pickle.load(f)
            log_model  = art.get("log_model")
            xgb_model  = art.get("xgb_model")
            scaler     = art.get("scaler")
            feat_cols  = art.get("feat_cols")
            best_model = art.get("best_model", "logistic")

    # ── load Youden threshold ──────────────────────────────────────────────────
    youden_threshold = state.get(
        "youden_threshold",
        float(_CFG.get("classification", {}).get("fallback_threshold", 0.5)),
    )
    thr_fp = OUT_MODELS / "threshold_analysis.csv"
    if thr_fp.exists():
        try:
            thr_df = pd.read_csv(thr_fp)
            _model_key = "xgboost" if (best_model or "").lower() == "xgboost" else "logistic"
            _row = thr_df[(thr_df["model"] == _model_key) & (thr_df["rule"] == "youden_opt")]
            if not _row.empty:
                youden_threshold = float(_row["youden_threshold"].iloc[0])
                print(f"[m4_export] Using Youden threshold={youden_threshold:.4f} "
                      f"(model={_model_key})")
        except Exception as _e:
            print(f"[m4_export] Could not load threshold_analysis.csv: {_e} — using 0.5")

    if keep_set is None:
        ks_fp = OUT_MODELS / "keep_set_features.txt"
        if ks_fp.exists():
            keep_set = ks_fp.read_text(encoding="utf-8").strip().split("\n")

    # ── build predictions on validation set ────────────────────────────────────
    feat_df = _load_feat_df(state)
    val_auc = log_auc or 0.0

    if feat_df is not None and log_model is not None:
        if feat_cols is None:
            feat_cols = [c for c in feat_df.columns
                         if c not in ("tconst", "label", "primaryTitle", "canonical_title")]
        feat_cols = [c for c in feat_cols if c in feat_df.columns]

        for col in feat_cols:
            med = float(feat_df[col].median()) if feat_df[col].notna().sum() > 0 else 0.0
            feat_df[col] = feat_df[col].fillna(med)

        if "label" in feat_df.columns:
            label_num = pd.to_numeric(feat_df["label"], errors="coerce")
            dropped   = int(label_num.isna().sum())
            if dropped:
                print(f"[m4_export] Dropping {dropped} rows with non-numeric label.")
            feat_df = feat_df.loc[label_num.notna()].copy()
            feat_df["label"] = label_num.loc[label_num.notna()].astype(int).to_numpy()

            train_idx, val_idx = train_test_split(
                feat_df.index,
                test_size=float(_CFG.get("global", {}).get("test_size", 0.20)),
                random_state=SEED,
                stratify=feat_df["label"].astype(int),
            )
            X_vl = feat_df.loc[val_idx, feat_cols]
            y_vl = feat_df.loc[val_idx, "label"].astype(int)

            if scaler is None:
                X_tr   = feat_df.loc[train_idx, feat_cols]
                scaler = StandardScaler()
                scaler.fit(X_tr)

            active_type = (
                "xgboost" if best_model.lower() == "xgboost" and xgb_model else "logistic"
            )
            if active_type == "xgboost":
                val_probs = xgb_model.predict_proba(X_vl)[:, 1]
            else:
                val_probs = log_model.predict_proba(scaler.transform(X_vl))[:, 1]

            val_auc = float(roc_auc_score(y_vl, val_probs))
            if log_auc is None:
                log_auc = val_auc

            val_pred_df = feat_df.loc[val_idx, ["tconst"]].copy()
            val_pred_df["predicted_prob"]  = val_probs
            val_pred_df["predicted_label"] = (val_probs >= youden_threshold).astype(int)
            val_pred_df["true_label"]      = y_vl.values
            val_pred_df.to_csv(OUT_MODELS / "predictions_val.csv", index=False)
            print(f"[m4_export] Saved predictions_val.csv  "
                  f"({len(val_pred_df)} rows, AUC={val_auc:.4f})")

    # ── predictions on real val / test splits (submission files) ───────────────
    _active_model = None
    _active_type  = "logistic"
    if log_model is not None or xgb_model is not None:
        _active_type  = (
            "xgboost" if (best_model or "").lower() == "xgboost" and xgb_model else "logistic"
        )
        _active_model = xgb_model if _active_type == "xgboost" else log_model

    if _active_model is not None and feat_cols is not None:
        for _split_name, _out_name in [
            ("val",  "predictions_val_submission.csv"),
            ("test", "predictions_test.csv"),
        ]:
            _fp = OUT_FEAT / f"features_{_split_name}.parquet"
            if not _fp.exists():
                _fp = OUT_FEAT / f"features_{_split_name}.csv"
                if not _fp.exists():
                    print(f"[m4_export] features_{_split_name} not found — skipping {_out_name}.")
                    continue
            _df = pd.read_parquet(_fp) if str(_fp).endswith(".parquet") else pd.read_csv(_fp)
            for _c in _df.columns:
                if _c not in ("tconst", "label"):
                    _df[_c] = pd.to_numeric(_df[_c], errors="coerce")

            _cols = [c for c in feat_cols if c in _df.columns]
            _X    = _df[_cols].copy()
            # Fill any remaining NaNs with column medians (XGBoost handles NaN natively,
            # but LogisticRegression requires finite values)
            if _active_type != "xgboost":
                for _c in _cols:
                    _med = float(_X[_c].median()) if _X[_c].notna().sum() > 0 else 0.0
                    _X[_c] = _X[_c].fillna(_med)
                if scaler is not None:
                    _X = pd.DataFrame(scaler.transform(_X), columns=_cols, index=_X.index)

            _probs  = _active_model.predict_proba(_X)[:, 1]
            _out_df = _df[["tconst"]].copy()
            _out_df["predicted_prob"]  = _probs
            _out_df["predicted_label"] = (_probs >= youden_threshold).astype(int)
            _out_df.to_csv(OUT_MODELS / _out_name, index=False)
            print(f"[m4_export] Saved {_out_name}  ({len(_out_df)} rows)")

    # ── manifest ───────────────────────────────────────────────────────────────
    manifest = _build_manifest(OUT_MODELS)
    manifest.to_csv(OUT_MODELS / "pipeline_manifest.csv", index=False)

    # ── figures ────────────────────────────────────────────────────────────────
    print("[m4_export] Saving figures...")
    _fig_manifest_sizes(manifest)

    funnel_stages: List[tuple] = []
    funnel_candidates = [
        (OUT_FEAT    / "features_train_prepped.csv",     "Prepped matrix rows"),
        (OUT_FEAT    / "features_train_prepped.parquet", "Prepped matrix rows"),
        (OUT_MODELS  / "feature_diagnostics.csv",        "Diagnostic features"),
        (OUT_MODELS  / "model_results.csv",              "Models evaluated"),
        (OUT_MODELS  / "threshold_analysis.csv",         "Threshold rows"),
        (OUT_MODELS  / "predictions_val.csv",            "Val predictions"),
    ]
    seen_labels: set = set()
    for fp, label in funnel_candidates:
        if label in seen_labels:
            continue
        if fp.exists():
            try:
                n = sum(1 for _ in open(fp, encoding="utf-8")) - 1
                funnel_stages.append((label, max(n, 0)))
                seen_labels.add(label)
            except Exception:
                pass

    if funnel_stages:
        _fig_pipeline_funnel(funnel_stages)

    # ── print clean summary ────────────────────────────────────────────────────
    n_artifacts = len(manifest)
    print(
        f"\n[m4_export] Pipeline complete.\n"
        f"  Artifacts  : {n_artifacts} files in {OUT_MODELS}\n"
        f"  Best model : {best_model}\n"
        f"  Val AUC    : {val_auc:.4f}\n"
        f"  Threshold  : {youden_threshold:.4f} (Youden's J)\n"
        f"  Keep-set   : {len(keep_set) if keep_set else '—'} features"
    )

    state.update({
        "manifest": manifest,
        "val_auc":  val_auc,
    })
    return state


if __name__ == "__main__":
    run({})
    print("[m4_export] Done.")
