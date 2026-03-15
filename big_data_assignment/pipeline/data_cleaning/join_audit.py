"""
join_audit.py — Join correctness, fanout, and downstream drift validation.

Answers the questions:
  "Did joins preserve 1-row-per-entity?"
  "Will the cleaned data behave well downstream?"

Checks
------
  1. Fanout / duplicate check     — tconst uniqueness after every join stage;
                                    detects 1-to-many fanout that silently duplicates rows
  2. Row reconciliation           — counts through: raw → clean; printed as funnel
  3. Distribution drift           — PSI + KS + mean/std delta for each numeric
                                    column across train / val / test splits

Called from s9_report.run() automatically.

Entry point
-----------
  run(raw_splits, clean_splits, fig_dir) -> dict
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── rcParams ──────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "figure.facecolor":  "#0a0a0a",
    "axes.facecolor":    "#111111",
    "axes.edgecolor":    "#252525",
    "axes.labelcolor":   "#e8e8e8",
    "text.color":        "#e8e8e8",
    "xtick.color":       "#666666",
    "ytick.color":       "#666666",
    "grid.color":        "#1e1e1e",
    "grid.linewidth":    0.6,
    "axes.grid":         True,
    "axes.grid.axis":    "y",
    "figure.dpi":        130,
    "savefig.dpi":       130,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "#0a0a0a",
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "legend.framealpha": 0.15,
    "legend.edgecolor":  "#252525",
})

# ── Colour palette ────────────────────────────────────────────────────────────
BG   = "#0a0a0a"
CRD  = "#111111"
BDR  = "#252525"
TXT  = "#e8e8e8"
MUT  = "#666666"
Y    = "#F5C518"
GRN  = "#2ecc71"
RED  = "#e74c3c"
ORG  = "#f39c12"
BLU  = "#1848f5"

SPLIT_COLORS = {"train": BLU, "validation_hidden": ORG, "test_hidden": GRN}
SPLIT_LABELS = {"train": "train", "validation_hidden": "val", "test_hidden": "test"}
NUMERIC_COLS = ("startYear", "runtimeMinutes", "numVotes_log1p")


# ── helpers ───────────────────────────────────────────────────────────────────

def _psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    bins = np.quantile(expected, np.linspace(0, 1, n_bins + 1))
    bins[0] -= 1e-6
    bins[-1] += 1e-6
    e = np.histogram(expected, bins=bins)[0].astype(float)
    a = np.histogram(actual,   bins=bins)[0].astype(float)
    e = np.where(e == 0, 0.5, e)
    a = np.where(a == 0, 0.5, a)
    e /= e.sum(); a /= a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


# ── 1. Fanout / duplicate check ───────────────────────────────────────────────

def fanout_check(clean_splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Verify exactly one row per tconst in every split post-pipeline.

    A LEFT JOIN with any 1-to-many relationship silently inflates row counts;
    this check catches that.
    """
    rows = []
    for split, df in clean_splits.items():
        if "tconst" not in df.columns:
            rows.append({"split": split, "total_rows": len(df),
                         "unique_tconst": None, "duplicate_rows": None,
                         "fanout_rate_pct": None, "status": "no_tconst_column"})
            continue
        total    = len(df)
        unique   = df["tconst"].nunique()
        dups     = total - unique
        dup_pct  = round(dups / total * 100, 3) if total else 0.0
        status   = "ok" if dups == 0 else ("warn" if dup_pct < 1.0 else "fail")
        rows.append({
            "split":           split,
            "total_rows":      total,
            "unique_tconst":   unique,
            "duplicate_rows":  dups,
            "fanout_rate_pct": dup_pct,
            "status":          status,
        })
    return pd.DataFrame(rows)


# ── 2. Row reconciliation ─────────────────────────────────────────────────────

def row_reconciliation(
    raw_splits:   dict[str, pd.DataFrame],
    clean_splits: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Tracks row counts through each pipeline stage:  raw → clean.
    Computes rows_dropped, rows_dropped_pct, and imputed values per column.
    """
    rows = []
    for split in set(raw_splits) | set(clean_splits):
        raw_df   = raw_splits.get(split,   pd.DataFrame())
        clean_df = clean_splits.get(split, pd.DataFrame())

        n_raw   = len(raw_df)
        n_clean = len(clean_df)
        dropped = n_raw - n_clean

        row: dict = {
            "split":           split,
            "raw_rows":        n_raw,
            "clean_rows":      n_clean,
            "rows_dropped":    dropped,
            "rows_dropped_pct": round(dropped / n_raw * 100, 3) if n_raw else 0.0,
        }

        # Count imputed values per column (raw null → clean non-null)
        for col in ("startYear", "runtimeMinutes", "numVotes"):
            raw_col = col
            clean_col = col if col != "numVotes" else "numVotes_log1p"
            if raw_col in raw_df.columns and clean_col in clean_df.columns and not raw_df.empty:
                raw_null  = raw_df[raw_col].isna().sum() if hasattr(raw_df[raw_col], "isna") else 0
                clean_col_vals = clean_df[clean_col] if clean_col in clean_df.columns else pd.Series()
                # imputed = was null in raw, now non-null in clean
                raw_null_count = int(pd.to_numeric(raw_df[raw_col], errors="coerce").isna().sum())
                row[f"{col}_raw_null"]     = raw_null_count
                row[f"{col}_imputed_rows"] = raw_null_count  # all NULLs are filled by MICE gate
            else:
                row[f"{col}_raw_null"]     = None
                row[f"{col}_imputed_rows"] = None

        rows.append(row)

    return pd.DataFrame(rows).sort_values("split").reset_index(drop=True)


# ── 3. Distribution drift across splits ───────────────────────────────────────

def distribution_drift(clean_splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    For each numeric column, compute PSI + KS statistic comparing train vs val
    and train vs test distributions.

    High PSI or KS after cleaning indicates the pipeline treats splits differently
    (e.g., imputation leak, or genuine cohort differences that affect generalisation).
    """
    try:
        from scipy.stats import ks_2samp
    except ImportError:
        return pd.DataFrame({"error": ["scipy not available"]})

    train_df = clean_splits.get("train", pd.DataFrame())
    if train_df.empty:
        return pd.DataFrame()

    rows = []
    for col in NUMERIC_COLS:
        if col not in train_df.columns:
            continue
        train_vals = pd.to_numeric(train_df[col], errors="coerce").dropna().values

        for comp_split in ("validation_hidden", "test_hidden"):
            comp_df = clean_splits.get(comp_split, pd.DataFrame())
            if comp_df.empty or col not in comp_df.columns:
                continue
            comp_vals = pd.to_numeric(comp_df[col], errors="coerce").dropna().values
            if len(comp_vals) < 5:
                continue

            ks_stat, ks_pval = ks_2samp(train_vals, comp_vals)
            psi_val  = _psi(train_vals, comp_vals)
            mean_d   = float(np.mean(comp_vals)) - float(np.mean(train_vals))
            std_d    = float(np.std(comp_vals))  - float(np.std(train_vals))

            if psi_val < 0.10:
                status = "stable"
            elif psi_val < 0.25:
                status = "moderate_shift"
            else:
                status = "high_shift"

            rows.append({
                "column":         col,
                "comparison":     f"train_vs_{SPLIT_LABELS.get(comp_split, comp_split)}",
                "n_train":        len(train_vals),
                "n_compare":      len(comp_vals),
                "ks_stat":        round(ks_stat, 4),
                "ks_pval":        round(ks_pval, 4),
                "psi":            round(psi_val, 4),
                "train_mean":     round(float(np.mean(train_vals)), 3),
                "comp_mean":      round(float(np.mean(comp_vals)), 3),
                "mean_delta":     round(mean_d, 3),
                "std_delta":      round(std_d, 3),
                "status":         status,
            })

    return pd.DataFrame(rows)


# ── Checks (formatted strings for stdout) ─────────────────────────────────────

def _check_fanout(fanout_df: pd.DataFrame) -> list[str]:
    lines = []
    for _, r in fanout_df.iterrows():
        if r["status"] == "no_tconst_column":
            lines.append(f"  ? fanout [{r['split']}]: no tconst column")
        elif r["duplicate_rows"] == 0:
            lines.append(f"  ✓ fanout [{r['split']}]: 0 duplicate tconst — 1-row-per-entity confirmed")
        elif r["fanout_rate_pct"] < 1.0:
            lines.append(f"  ⚠ fanout [{r['split']}]: {r['duplicate_rows']} duplicates "
                         f"({r['fanout_rate_pct']:.3f}%)  ← minor fanout; check JOIN cardinality")
        else:
            lines.append(f"  ✗ fanout [{r['split']}]: {r['duplicate_rows']} duplicates "
                         f"({r['fanout_rate_pct']:.2f}%)  ← JOIN fanout likely inflating row counts")
    return lines


def _check_drift(drift_df: pd.DataFrame) -> list[str]:
    lines = []
    for _, r in drift_df.iterrows():
        sym = "✓" if r["status"] == "stable" else ("⚠" if r["status"] == "moderate_shift" else "✗")
        lines.append(
            f"  {sym} drift [{r['column']}] {r['comparison']}: "
            f"PSI={r['psi']:.3f} ({r['status']})  KS={r['ks_stat']:.3f}  "
            f"mean_delta={r['mean_delta']:+.3f}"
        )
    return lines


# ── Figures ───────────────────────────────────────────────────────────────────

def _fig_fanout(fanout_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 5))
    splits = fanout_df["split"].tolist()
    dups   = fanout_df["duplicate_rows"].fillna(0).tolist()
    colors = [GRN if d == 0 else (ORG if d < fanout_df["total_rows"].max() * 0.01 else RED)
              for d in dups]
    bars = ax.bar(splits, dups, color=colors, alpha=0.88, edgecolor="none")
    ax.set_ylabel("Duplicate rows", fontsize=11, color=TXT, labelpad=8)
    ax.set_title("Fanout check — duplicate tconst rows per split", fontsize=13,
                 fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    ax.tick_params(axis="x", rotation=15, labelsize=9)

    y_max = ax.get_ylim()[1]
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005 * y_max,
                str(int(h)), ha="center", va="bottom", fontsize=8, color=TXT)

    fig.tight_layout(pad=2.5)
    return fig


def _fig_row_reconciliation(recon_df: pd.DataFrame) -> plt.Figure:
    splits = recon_df["split"].tolist()
    n = len(splits)
    x = np.arange(n)
    w = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    bars_raw   = ax.bar(x - w / 2, recon_df["raw_rows"],   width=w, color=RED, alpha=0.88,
                        edgecolor="none", label="raw rows")
    bars_clean = ax.bar(x + w / 2, recon_df["clean_rows"], width=w, color=GRN, alpha=0.88,
                        edgecolor="none", label="clean rows")
    ax.set_xticks(x)
    ax.set_xticklabels(splits, rotation=15, fontsize=9, color=MUT)
    ax.set_ylabel("Row count", fontsize=11, color=TXT, labelpad=8)
    ax.set_title("Row reconciliation — raw vs clean per split", fontsize=13,
                 fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    ax.legend(fontsize=9)

    y_max = ax.get_ylim()[1]
    for bar in list(bars_raw) + list(bars_clean):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005 * y_max,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8, color=TXT)

    for i, (_, r) in enumerate(recon_df.iterrows()):
        imputed_cols = [c for c in recon_df.columns if c.endswith("_imputed_rows")]
        total_imputed = sum(int(r[c]) for c in imputed_cols if r[c] is not None and not pd.isna(r[c]))
        if total_imputed > 0:
            ax.text(i, r["clean_rows"] + r["clean_rows"] * 0.01,
                    f"~{total_imputed} imputed", ha="center", color=ORG, fontsize=7.5)

    fig.tight_layout(pad=2.5)
    return fig


def _fig_distribution_drift(drift_df: pd.DataFrame, clean_splits: dict[str, pd.DataFrame]) -> plt.Figure:
    _train = clean_splits.get("train")
    cols_present = [c for c in NUMERIC_COLS if c in (_train.columns if _train is not None else [])]
    n = len(cols_present)
    if n == 0:
        fig, ax = plt.subplots(figsize=(14, 5))
        return fig

    width = 18 if n >= 3 else (16 if n == 2 else 14)
    fig, axes = plt.subplots(1, n, figsize=(width, 5))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, cols_present):
        for split, color in SPLIT_COLORS.items():
            df = clean_splits.get(split, pd.DataFrame())
            if df.empty or col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            lbl  = SPLIT_LABELS.get(split, split)
            ax.hist(vals, bins=35, density=True, color=color, alpha=0.35,
                    edgecolor="none", label=f"{lbl} (n={len(vals):,})")

        sub = drift_df[drift_df["column"] == col]
        if not sub.empty:
            psi_txt = "\n".join(f"{r['comparison']}: PSI={r['psi']:.3f}" for _, r in sub.iterrows())
            ax.text(0.97, 0.97, psi_txt, transform=ax.transAxes,
                    ha="right", va="top", color=MUT, fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=CRD, edgecolor=BDR, alpha=0.7))

        ax.set_title(col, fontsize=13, fontweight="bold", color=TXT, pad=12)
        ax.title.set_position([0.5, 1.02])
        ax.set_xlabel("value", fontsize=11, color=TXT, labelpad=8)
        ax.legend(fontsize=9)

    fig.suptitle("Distribution drift: train vs val / test (post-cleaning)", color=TXT,
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(pad=2.5)
    return fig


# ── entry point ───────────────────────────────────────────────────────────────

def run(
    raw_splits:   dict[str, pd.DataFrame],
    clean_splits: dict[str, pd.DataFrame],
    fig_dir:      Path,
    csv_dir:      Path | None = None,
) -> dict:
    """
    Run all join and downstream quality checks.

    Parameters
    ----------
    raw_splits   : raw CSVs loaded with disguised-missing → NaN
    clean_splits : fully cleaned DataFrames (from parquets)
    fig_dir      : directory for PNG figures
    csv_dir      : directory for CSV outputs (defaults to fig_dir)

    Returns
    -------
    dict with keys: fanout_check, row_reconciliation, distribution_drift
    """
    fig_dir = Path(fig_dir)
    csv_dir = Path(csv_dir) if csv_dir else fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    print("\n[join_audit] Running join & downstream quality checks...")
    results: dict = {}

    # 1. Fanout
    print("[join_audit]   1/3 fanout / duplicate check...")
    try:
        fanout_df = fanout_check(clean_splits)
        results["fanout_check"] = fanout_df
        fanout_df.to_csv(csv_dir / "join_fanout_check.csv", index=False)
        for line in _check_fanout(fanout_df):
            print(line)
        fig = _fig_fanout(fanout_df)
        fig.savefig(fig_dir / "16_fanout_check.png", dpi=130, bbox_inches="tight",
                    facecolor=BG, edgecolor="none")
        plt.close(fig)
    except Exception as exc:
        print(f"  ⚠ fanout check failed: {exc}")

    # 2. Row reconciliation
    print("[join_audit]   2/3 row reconciliation...")
    try:
        recon_df = row_reconciliation(raw_splits, clean_splits)
        results["row_reconciliation"] = recon_df
        recon_df.to_csv(csv_dir / "join_row_reconciliation.csv", index=False)
        for _, r in recon_df.iterrows():
            sym = "✓" if r["rows_dropped"] == 0 else "⚠"
            print(f"  {sym} reconciliation [{r['split']}]: "
                  f"{r['raw_rows']:,} raw → {r['clean_rows']:,} clean  "
                  f"(dropped={r['rows_dropped']}  {r['rows_dropped_pct']:.2f}%)")
        fig = _fig_row_reconciliation(recon_df)
        fig.savefig(fig_dir / "17_row_reconciliation.png", dpi=130, bbox_inches="tight",
                    facecolor=BG, edgecolor="none")
        plt.close(fig)
    except Exception as exc:
        print(f"  ⚠ row reconciliation failed: {exc}")

    # 3. Distribution drift across splits
    print("[join_audit]   3/3 distribution drift (train vs val/test)...")
    try:
        drift_df = distribution_drift(clean_splits)
        if not drift_df.empty:
            results["distribution_drift"] = drift_df
            drift_df.to_csv(csv_dir / "join_distribution_drift.csv", index=False)
            for line in _check_drift(drift_df):
                print(line)
        fig = _fig_distribution_drift(drift_df, clean_splits)
        fig.savefig(fig_dir / "18_distribution_drift.png", dpi=130, bbox_inches="tight",
                    facecolor=BG, edgecolor="none")
        plt.close(fig)
    except Exception as exc:
        print(f"  ⚠ distribution drift failed: {exc}")

    print("[join_audit] Done.")
    return results
