"""
members/ilesh/model.py
======================
Step 3 — sklearn: Model Training, Evaluation & Predictions.

WHY sklearn HERE?
  - Features are already StandardScaled by the PySpark pipeline — no extra
    preprocessing needed.
  - LogisticRegression is fast, interpretable, and well-suited to this
    8K-row feature matrix. No JVM overhead justified at this data size.
  - The leakage-free comparison re-runs fold-aware feature engineering
    purely in Pandas to give an honest internal score (~0.72 accuracy)
    that matches the Kaggle leaderboard (~0.73).

Can be imported and called independently:
    from members.ilesh.model import step3_model
    step3_model(train_pdf, val_pdf, test_pdf, raw_csv, feat_dir, sub_dir)
"""

from __future__ import annotations

import datetime
import glob as _glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from members.ilesh.features import FEATURE_COLS


def step3_model(
    train_pdf: pd.DataFrame | None,
    val_pdf:   pd.DataFrame | None,
    test_pdf:  pd.DataFrame | None,
    raw_csv:  Path,
    feat_dir: Path,
    sub_dir:  Path,
) -> None:
    """
    Train LogisticRegression, run leakage-free baseline comparison,
    and write submission files.

    Parameters
    ----------
    train_pdf / val_pdf / test_pdf : feature DataFrames from step2_spark()
                                     (pass None to load from feat_dir CSVs)
    raw_csv   : directory with original CSV files (for row-order alignment)
    feat_dir  : directory containing train/val/test_features.csv
    sub_dir   : output directory for submission .txt files
    """
    sub_dir.mkdir(parents=True, exist_ok=True)
    print("\n>> STEP 3 — sklearn: model training...")

    # Load CSVs if not passed in from Step 2
    if train_pdf is None:
        train_pdf = pd.read_csv(feat_dir / "train_features.csv")
        val_pdf   = pd.read_csv(feat_dir / "val_features.csv")
        test_pdf  = pd.read_csv(feat_dir / "test_features.csv")

    avail = [c for c in FEATURE_COLS if c in train_pdf.columns]
    X_all = train_pdf[avail].fillna(0).astype(float)
    y_all = (train_pdf["label"].astype(str).str.strip() == "True").astype(int)

    X_tr, _, y_tr, _ = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    print(f"  Train={len(X_tr):,}  features={len(avail)}")

    # Features already StandardScaled by PySpark pipeline — no extra scaler needed
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_tr, y_tr)

    # ── Leakage-free baseline comparison ──────────────────────────────────────
    _leakage_free_comparison(raw_csv)

    # ── Write submissions ─────────────────────────────────────────────────────
    # Spark's toPandas() does not preserve row order.
    # Re-align by merging on tconst with the original CSVs before writing.
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def _write_sub(X: pd.DataFrame, raw_csv_path: str, path: Path) -> None:
        raw_order = pd.read_csv(raw_csv_path)[["tconst"]]
        X_ordered = raw_order.merge(X, on="tconst", how="left")
        preds = model.predict(X_ordered[avail].fillna(0).astype(float))
        lines = ["True" if p == 1 else "False" for p in preds]
        path.write_text("\n".join(lines))
        print(f"  [OK] {path.name}: {len(lines)} predictions (order aligned to original CSV)")

    _write_sub(val_pdf,  str(raw_csv / "validation_hidden.csv"), sub_dir / f"ilesh_val_{ts}.txt")
    _write_sub(test_pdf, str(raw_csv / "test_hidden.csv"),       sub_dir / f"ilesh_test_{ts}.txt")
    print(f"[STEP 3 DONE] → {sub_dir}")


# ── Leakage-free comparison (internal helper) ─────────────────────────────────

def _leakage_free_comparison(raw_csv: Path) -> None:
    """
    Re-runs the full feature engineering fold-aware on raw CSVs:
      IQR fence, medians, success rates — all computed on the 80% fold only,
      then applied to the 20% holdout → no label leakage.
    Compares majority class / title-length naive / raw numerics / full pipeline.
    """
    print("\n  --- Leakage-free baseline comparison (fold-aware pipeline) ---")

    raw_frames = [pd.read_csv(p) for p in sorted(_glob.glob(str(raw_csv / "train-*.csv")))]
    raw_df = pd.concat(raw_frames, ignore_index=True)

    movies_raw = pd.DataFrame({
        "tconst":         raw_df["tconst"],
        "startYear":      pd.to_numeric(raw_df["startYear"].replace("\\N", float("nan")), errors="coerce"),
        "runtimeMinutes": pd.to_numeric(raw_df["runtimeMinutes"].replace("\\N", float("nan")), errors="coerce"),
        "numVotes":       pd.to_numeric(raw_df["numVotes"].replace("\\N", float("nan")), errors="coerce"),
        "label":          (raw_df["label"].astype(str).str.strip() == "True").astype(int),
        "primaryTitle":   raw_df["primaryTitle"].fillna(""),
    })

    dir_df = pd.read_csv(str(raw_csv / "movie_directors.csv"))
    wri_df = pd.read_csv(str(raw_csv / "movie_writers.csv"))
    dir_df = dir_df[~dir_df["director"].isin(["\\N", "\\\\N", ""])].dropna(subset=["director"])
    wri_df = wri_df[~wri_df["writer"].isin(["\\N", "\\\\N", ""])].dropna(subset=["writer"])
    dir_df.columns = ["tconst", "director_id"]
    wri_df.columns = ["tconst", "writer_id"]

    tr_idx, hv_idx = train_test_split(
        movies_raw.index, test_size=0.2, random_state=42, stratify=movies_raw["label"]
    )
    tr = movies_raw.loc[tr_idx].copy().reset_index(drop=True)
    hv = movies_raw.loc[hv_idx].copy().reset_index(drop=True)

    # (A) Naive: majority class
    maj_acc = accuracy_score(hv["label"], np.full(len(hv), int(tr["label"].mode()[0])))

    # (B) Naive: title length only (mirrors shared baseline in src/)
    tl_tr = tr["primaryTitle"].str.len().values.reshape(-1, 1).astype(float)
    tl_hv = hv["primaryTitle"].str.len().values.reshape(-1, 1).astype(float)
    naive_lr  = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(tl_tr, tr["label"])
    naive_acc = accuracy_score(hv["label"], naive_lr.predict(tl_hv))
    naive_auc = roc_auc_score(hv["label"], naive_lr.predict_proba(tl_hv)[:, 1])

    # (C) Naive: raw 3 numerics, no cap, no log
    rn_tr = tr[["numVotes", "runtimeMinutes", "startYear"]].fillna(0).astype(float)
    rn_hv = hv[["numVotes", "runtimeMinutes", "startYear"]].fillna(0).astype(float)
    raw_lr  = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(rn_tr, tr["label"])
    raw_acc = accuracy_score(hv["label"], raw_lr.predict(rn_hv))
    raw_auc = roc_auc_score(hv["label"], raw_lr.predict_proba(rn_hv)[:, 1])

    # (D) Full pipeline — fold-aware (no leakage)
    q1f, q3f = tr["numVotes"].quantile(0.25), tr["numVotes"].quantile(0.75)
    fence_f  = q3f + 1.5 * (q3f - q1f)
    meds = {c: tr[c].median() for c in ["startYear", "runtimeMinutes", "numVotes"]}
    for df in [tr, hv]:
        df["numVotes"] = df["numVotes"].clip(upper=fence_f)
        df["is_missing_numVotes"]       = df["numVotes"].isna().astype(float)
        df["is_missing_startYear"]      = df["startYear"].isna().astype(float)
        df["is_missing_runtimeMinutes"] = df["runtimeMinutes"].isna().astype(float)
        for c, v in meds.items():
            df[c] = df[c].fillna(v)
        df["log_numVotes"] = np.log1p(df["numVotes"])

    def _add_person_feats(tr_df, apply_df, edges, id_col, prefix):
        rates = (edges.merge(tr_df[["tconst", "label"]], on="tconst", how="inner")
                      .groupby(id_col)["label"].mean().rename("rate").reset_index())
        agg   = (edges.merge(rates, on=id_col, how="left")
                      .groupby("tconst")["rate"]
                      .agg(avg="mean", max="max").reset_index()
                      .rename(columns={"avg": f"avg_{prefix}_success_rate",
                                       "max": f"max_{prefix}_success_rate"}))
        cnt   = (edges.groupby("tconst")[id_col].nunique()
                      .rename(f"num_{prefix}s").reset_index())
        out = apply_df.merge(cnt, on="tconst", how="left").merge(agg, on="tconst", how="left")
        out[f"num_{prefix}s"]             = out[f"num_{prefix}s"].fillna(0)
        out[f"avg_{prefix}_success_rate"] = out[f"avg_{prefix}_success_rate"].fillna(0.5)
        out[f"max_{prefix}_success_rate"] = out[f"max_{prefix}_success_rate"].fillna(0.5)
        return out

    tr = _add_person_feats(tr, tr, dir_df, "director_id", "director")
    hv = _add_person_feats(tr, hv, dir_df, "director_id", "director")
    tr = _add_person_feats(tr, tr, wri_df, "writer_id",   "writer")
    hv = _add_person_feats(tr, hv, wri_df, "writer_id",   "writer")

    PIPE_FEAT = ["startYear", "runtimeMinutes", "log_numVotes",
                 "num_directors", "num_writers",
                 "avg_director_success_rate", "max_director_success_rate",
                 "avg_writer_success_rate",   "max_writer_success_rate",
                 "is_missing_numVotes", "is_missing_startYear", "is_missing_runtimeMinutes"]
    pipe_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    pipe_lr.fit(tr[PIPE_FEAT].fillna(0).astype(float), tr["label"])
    pipe_acc = accuracy_score(hv["label"], pipe_lr.predict(hv[PIPE_FEAT].fillna(0).astype(float)))
    pipe_auc = roc_auc_score(hv["label"], pipe_lr.predict_proba(hv[PIPE_FEAT].fillna(0).astype(float))[:, 1])

    print(f"  {'Model':<44} {'Accuracy':>9} {'ROC-AUC':>9}")
    print(f"  {'-'*62}")
    print(f"  {'Majority class (always predict True/False)':<44} {maj_acc:>9.4f} {'—':>9}")
    print(f"  {'Naive: title length only (shared baseline)':<44} {naive_acc:>9.4f} {naive_auc:>9.4f}")
    print(f"  {'Raw numerics (no cleaning, no engineering)':<44} {raw_acc:>9.4f} {raw_auc:>9.4f}")
    print(f"  {'Ilesh pipeline — leakage-free (12 features)':<44} {pipe_acc:>9.4f} {pipe_auc:>9.4f}")
    print(f"  {'-'*62}")
    print(f"  Pipeline gain over naive : +{pipe_acc-naive_acc:.4f} accuracy  +{pipe_auc-naive_auc:.4f} AUC")
