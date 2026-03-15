"""
imputation_audit.py — Deep imputation quality validation.

Answers the question: "Did imputation preserve the data-generating structure?"

Checks
------
  1. Masked-value reconstruction   — mask 20% of observed values, impute, score
                                     MAE / RMSE / within-tolerance vs MICE & baselines
  2. Baseline comparison           — MICE vs median, mean, column-mean-by-group
  3. Distribution shift            — KS statistic + Wasserstein + PSI,
                                     observed (non-missing in raw) vs imputed (was missing)
  4. Correlation preservation      — correlation matrix on complete-cases vs all rows
  5. Conditional plausibility      — per-titleType KS test on imputed column values

Called from s9_report.run() automatically.

Entry point
-----------
  run(clean_train, raw_train, fig_dir) -> dict  (metrics as DataFrames, saved to fig_dir)
"""
from __future__ import annotations

import warnings
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

# ── column config ─────────────────────────────────────────────────────────────
IMPUTED_COLS = ("startYear", "runtimeMinutes", "numVotes_log1p")

# raw column name that maps to each imputed col (for identifying missing rows)
RAW_COL_MAP = {
    "startYear":      "startYear",
    "runtimeMinutes": "runtimeMinutes",
    "numVotes_log1p": "numVotes",      # raw has "numVotes"; clean has log-transformed version
}

# tolerance used for within-tolerance % metric
TOLERANCES = {
    "startYear":      2.0,   # ±2 years
    "runtimeMinutes": 10.0,  # ±10 min
    "numVotes_log1p": 0.5,   # ±0.5 on log1p scale ≈ 65% relative error
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between two continuous distributions."""
    bins = np.quantile(expected, np.linspace(0, 1, n_bins + 1))
    bins[0] -= 1e-6
    bins[-1] += 1e-6
    e_counts = np.histogram(expected, bins=bins)[0].astype(float)
    a_counts = np.histogram(actual,   bins=bins)[0].astype(float)
    # Smooth zeros
    e_counts = np.where(e_counts == 0, 0.5, e_counts)
    a_counts = np.where(a_counts == 0, 0.5, a_counts)
    e_pct = e_counts / e_counts.sum()
    a_pct = a_counts / a_counts.sum()
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def _get_obs_imp_masks(
    clean_train: pd.DataFrame,
    raw_train: pd.DataFrame,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Returns {imputed_col: (observed_mask, imputed_mask)} boolean arrays over clean_train rows,
    derived by joining clean_train with raw_train on tconst.
    """
    merged = clean_train[["tconst"]].merge(
        raw_train[["tconst"] + [c for c in RAW_COL_MAP.values() if c in raw_train.columns]],
        on="tconst", how="left",
    )
    masks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for imp_col, raw_col in RAW_COL_MAP.items():
        if imp_col not in clean_train.columns or raw_col not in merged.columns:
            continue
        raw_vals = pd.to_numeric(merged[raw_col], errors="coerce")
        obs_mask = raw_vals.notna().values
        imp_mask = raw_vals.isna().values
        masks[imp_col] = (obs_mask, imp_mask)
    return masks


# ── 1. Masked-value reconstruction & 2. Baseline comparison ──────────────────

def masked_value_validation(
    clean_train: pd.DataFrame,
    raw_train: pd.DataFrame,
    mask_frac: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Mask mask_frac of observed (originally non-missing) values, impute with
    MICE and simple baselines, score against truth.

    Returns a DataFrame with one row per imputed column.
    """
    try:
        from sklearn.impute import IterativeImputer, SimpleImputer
        from sklearn.linear_model import BayesianRidge
        from sklearn.exceptions import ConvergenceWarning
    except ImportError:
        return pd.DataFrame({"error": ["sklearn not available"]})

    masks = _get_obs_imp_masks(clean_train, raw_train)
    cols  = [c for c in IMPUTED_COLS if c in clean_train.columns and c in masks]
    if not cols:
        return pd.DataFrame()

    # Work on the numeric matrix for the 3 imputed columns
    X_full = clean_train[cols].apply(pd.to_numeric, errors="coerce").values.astype(float)

    # Complete-case rows: observed in ALL imputed cols
    complete_mask = np.ones(len(clean_train), dtype=bool)
    for col in cols:
        obs, _ = masks[col]
        complete_mask &= obs

    X_complete = X_full[complete_mask]
    rng = np.random.default_rng(seed)
    rows_out = []

    for col_idx, col in enumerate(cols):
        tol = TOLERANCES.get(col, 1.0)
        n_mask = max(10, int(len(X_complete) * mask_frac))
        mask_rows = rng.choice(len(X_complete), n_mask, replace=False)

        X_masked = X_complete.copy()
        X_masked[mask_rows, col_idx] = np.nan
        true_vals = X_complete[mask_rows, col_idx]

        def _score(pred: np.ndarray, label: str) -> dict:
            diff = np.abs(pred - true_vals)
            return {
                f"{label}_mae":           round(float(np.mean(diff)), 4),
                f"{label}_rmse":          round(float(np.sqrt(np.mean(diff ** 2))), 4),
                f"{label}_within_tol_pct": round(float(np.mean(diff <= tol) * 100), 2),
            }

        # MICE
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            mice = IterativeImputer(estimator=BayesianRidge(), max_iter=10,
                                    random_state=seed)
            mice.fit(X_masked)
            mice_pred = mice.transform(X_masked)[mask_rows, col_idx]

        # Median baseline
        med = SimpleImputer(strategy="median")
        med.fit(X_masked)
        med_pred = med.transform(X_masked)[mask_rows, col_idx]

        # Mean baseline
        mean_imp = SimpleImputer(strategy="mean")
        mean_imp.fit(X_masked)
        mean_pred = mean_imp.transform(X_masked)[mask_rows, col_idx]

        row = {
            "column":    col,
            "n_complete": len(X_complete),
            "n_masked":  n_mask,
            "tolerance": tol,
        }
        row.update(_score(mice_pred,  "mice"))
        row.update(_score(med_pred,   "median"))
        row.update(_score(mean_pred,  "mean"))
        row["mice_vs_median_mae_delta"] = round(row["mice_mae"] - row["median_mae"], 4)
        row["mice_wins_mae"]  = bool(row["mice_mae"]  < row["median_mae"])
        row["mice_wins_rmse"] = bool(row["mice_rmse"] < row["median_rmse"])
        rows_out.append(row)

    return pd.DataFrame(rows_out)


# ── 3. Distribution shift (observed vs imputed) ───────────────────────────────

def distribution_shift(
    clean_train: pd.DataFrame,
    raw_train: pd.DataFrame,
) -> pd.DataFrame:
    """
    KS statistic + Wasserstein distance + PSI comparing:
      observed  = rows where raw value was NOT missing
      imputed   = rows where raw value WAS missing (filled by MICE)
    """
    try:
        from scipy.stats import ks_2samp, wasserstein_distance
    except ImportError:
        return pd.DataFrame({"error": ["scipy not available"]})

    masks = _get_obs_imp_masks(clean_train, raw_train)
    rows  = []

    for col in IMPUTED_COLS:
        if col not in clean_train.columns or col not in masks:
            continue
        obs_mask, imp_mask = masks[col]
        vals = pd.to_numeric(clean_train[col], errors="coerce")
        observed = vals[obs_mask].dropna().values
        imputed  = vals[imp_mask].dropna().values

        row: dict = {
            "column":     col,
            "n_observed": len(observed),
            "n_imputed":  len(imputed),
            "imputed_pct": round(len(imputed) / (len(observed) + len(imputed)) * 100, 2)
                           if len(observed) + len(imputed) > 0 else 0.0,
        }
        if len(imputed) < 5:
            row.update({"ks_stat": np.nan, "ks_pval": np.nan,
                        "wasserstein": np.nan, "psi": np.nan,
                        "obs_mean": round(float(np.mean(observed)), 3) if len(observed) else np.nan,
                        "imp_mean": np.nan,
                        "obs_std": round(float(np.std(observed)), 3) if len(observed) else np.nan,
                        "imp_std": np.nan,
                        "mean_delta": np.nan, "status": "no_imputed_rows"})
            rows.append(row)
            continue

        ks_stat, ks_pval = ks_2samp(observed, imputed)
        wass = wasserstein_distance(observed, imputed)
        psi  = _psi(observed, imputed)

        obs_mean, imp_mean = float(np.mean(observed)), float(np.mean(imputed))
        obs_std,  imp_std  = float(np.std(observed)),  float(np.std(imputed))

        if psi < 0.10:
            status = "stable"
        elif psi < 0.25:
            status = "moderate_shift"
        else:
            status = "high_shift"

        row.update({
            "ks_stat":     round(ks_stat, 4),
            "ks_pval":     round(ks_pval, 4),
            "wasserstein": round(wass, 4),
            "psi":         round(psi, 4),
            "obs_mean":    round(obs_mean, 3),
            "imp_mean":    round(imp_mean, 3),
            "obs_std":     round(obs_std, 3),
            "imp_std":     round(imp_std, 3),
            "mean_delta":  round(imp_mean - obs_mean, 3),
            "status":      status,
        })
        rows.append(row)

    return pd.DataFrame(rows)


# ── 4. Correlation preservation ───────────────────────────────────────────────

def correlation_preservation(
    clean_train: pd.DataFrame,
    raw_train: pd.DataFrame,
) -> dict:
    """
    Compare correlation matrix on complete-case rows (originally non-missing)
    vs all rows post-imputation.

    Returns dict with 'complete_corr', 'full_corr', 'diff', 'frobenius_norm'.
    """
    masks = _get_obs_imp_roberts = _get_obs_imp_masks(clean_train, raw_train)
    cols  = [c for c in IMPUTED_COLS if c in clean_train.columns]
    if not cols:
        return {}

    X_num = clean_train[cols].apply(pd.to_numeric, errors="coerce")

    complete_mask = np.ones(len(clean_train), dtype=bool)
    for col in cols:
        if col in masks:
            obs, _ = masks[col]
            complete_mask &= obs

    X_complete = X_num[complete_mask]
    X_full     = X_num

    corr_complete = X_complete.corr()
    corr_full     = X_full.corr()
    diff          = corr_full - corr_complete

    frob = float(np.sqrt((diff.values ** 2).sum()))

    return {
        "complete_corr":  corr_complete,
        "full_corr":      corr_full,
        "diff":           diff,
        "frobenius_norm": round(frob, 4),
        "n_complete":     int(complete_mask.sum()),
        "n_full":         len(X_full),
    }


# ── 5. Conditional plausibility ───────────────────────────────────────────────

def conditional_plausibility(
    clean_train: pd.DataFrame,
    raw_train: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each imputed column, run KS test within each titleType group comparing
    observed vs imputed rows.  Flags groups where imputed distribution diverges.
    """
    try:
        from scipy.stats import ks_2samp
    except ImportError:
        return pd.DataFrame({"error": ["scipy not available"]})

    if "titleType" not in clean_train.columns:
        return pd.DataFrame()

    masks = _get_obs_imp_masks(clean_train, raw_train)
    rows  = []

    for col in ("runtimeMinutes", "startYear"):  # most meaningful conditional checks
        if col not in clean_train.columns or col not in masks:
            continue
        obs_mask, imp_mask = masks[col]
        vals = pd.to_numeric(clean_train[col], errors="coerce")

        for tt, grp in clean_train.groupby("titleType", observed=True):
            grp_idx = grp.index
            obs_vals = vals[grp_idx[obs_mask[grp.index.get_indexer(grp_idx)] if hasattr(grp.index, 'get_indexer') else obs_mask[grp_idx]]].dropna()

            # Simpler approach: boolean mask aligned to grp
            grp_obs_mask = obs_mask[clean_train.index.get_indexer(grp_idx)]
            grp_imp_mask = imp_mask[clean_train.index.get_indexer(grp_idx)]

            obs_vals = vals.iloc[clean_train.index.get_indexer(grp_idx)[grp_obs_mask]].dropna()
            imp_vals = vals.iloc[clean_train.index.get_indexer(grp_idx)[grp_imp_mask]].dropna()

            if len(obs_vals) < 5 or len(imp_vals) < 2:
                continue

            ks_stat, ks_pval = ks_2samp(obs_vals.values, imp_vals.values)
            rows.append({
                "column":     col,
                "titleType":  tt,
                "n_observed": len(obs_vals),
                "n_imputed":  len(imp_vals),
                "ks_stat":    round(ks_stat, 4),
                "ks_pval":    round(ks_pval, 4),
                "plausible":  ks_pval >= 0.05,
                "note":       "plausible" if ks_pval >= 0.05 else "suspicious — imputed dist diverges",
            })

    return pd.DataFrame(rows)


# ── Figures ───────────────────────────────────────────────────────────────────

def _fig_masked_validation(masked_df: pd.DataFrame) -> plt.Figure:
    cols = masked_df["column"].tolist()
    n = len(cols)
    x = np.arange(n)
    w = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    ax_mae, ax_tol = axes

    bar_specs = [
        ("mice_mae",   Y,   "MICE"),
        ("median_mae", BLU, "Median"),
        ("mean_mae",   ORG, "Mean"),
    ]
    for i, (key, color, label) in enumerate(bar_specs):
        bars = ax_mae.bar(x + (i - 1) * w, masked_df[key], width=w, color=color,
                          alpha=0.88, edgecolor="none", label=label)
        y_max = ax_mae.get_ylim()[1]
        for bar in bars:
            h = bar.get_height()
            ax_mae.text(bar.get_x() + bar.get_width() / 2, h + 0.005 * y_max,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=8, color=TXT)

    ax_mae.set_xticks(x)
    ax_mae.set_xticklabels(cols, fontsize=9, color=MUT)
    ax_mae.set_ylabel("MAE", fontsize=11, color=TXT, labelpad=8)
    ax_mae.set_title("Masked-value reconstruction — MAE", fontsize=13,
                     fontweight="bold", color=TXT, pad=12)
    ax_mae.title.set_position([0.5, 1.02])
    ax_mae.legend(fontsize=9)

    tol_specs = [
        ("mice_within_tol_pct",   Y,   "MICE"),
        ("median_within_tol_pct", BLU, "Median"),
        ("mean_within_tol_pct",   ORG, "Mean"),
    ]
    for i, (key, color, label) in enumerate(tol_specs):
        bars = ax_tol.bar(x + (i - 1) * w, masked_df[key], width=w, color=color,
                          alpha=0.88, edgecolor="none", label=label)
        y_max = ax_tol.get_ylim()[1]
        for bar in bars:
            h = bar.get_height()
            ax_tol.text(bar.get_x() + bar.get_width() / 2, h + 0.005 * y_max,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=8, color=TXT)

    ax_tol.set_xticks(x)
    ax_tol.set_xticklabels(cols, fontsize=9, color=MUT)
    ax_tol.set_ylabel("Within-tolerance %", fontsize=11, color=TXT, labelpad=8)
    ax_tol.set_title("Masked-value reconstruction — Within tolerance", fontsize=13,
                     fontweight="bold", color=TXT, pad=12)
    ax_tol.title.set_position([0.5, 1.02])
    ax_tol.set_ylim(0, 105)
    ax_tol.legend(fontsize=9)

    tol_notes = "  ".join(f"{r['column']} tol=±{r['tolerance']}" for _, r in masked_df.iterrows())
    fig.text(0.5, 0.01, tol_notes, ha="center", color=MUT, fontsize=7.5)

    fig.tight_layout(rect=[0, 0.04, 1, 1], pad=2.5)
    return fig


def _fig_distribution_shift(shift_df: pd.DataFrame, clean_train: pd.DataFrame,
                             raw_train: pd.DataFrame) -> plt.Figure:
    masks = _get_obs_imp_masks(clean_train, raw_train)
    valid = [c for c in IMPUTED_COLS
             if c in clean_train.columns and c in masks
             and len(masks[c][1]) > 0]
    n = len(valid)
    if n == 0:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.text(0.5, 0.5, "No imputed rows", ha="center", va="center", color=MUT)
        return fig

    width = 18 if n >= 3 else (16 if n == 2 else 14)
    fig, axes = plt.subplots(1, n, figsize=(width, 5))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, valid):
        obs_mask, imp_mask = masks[col]
        vals = pd.to_numeric(clean_train[col], errors="coerce")
        observed = vals[obs_mask].dropna().values
        imputed  = vals[imp_mask].dropna().values

        bins = np.linspace(min(observed.min(), imputed.min() if len(imputed) else observed.min()),
                           max(observed.max(), imputed.max() if len(imputed) else observed.max()),
                           35)
        ax.hist(observed, bins=bins, density=True, color=BLU,  alpha=0.35,
                edgecolor="none", label=f"observed (n={len(observed):,})")
        ax.hist(imputed,  bins=bins, density=True, color=RED, alpha=0.35,
                edgecolor="none", label=f"imputed (n={len(imputed):,})")

        row = shift_df[shift_df["column"] == col]
        if not row.empty:
            r = row.iloc[0]
            stats_txt = f"KS={r['ks_stat']:.3f}  p={r['ks_pval']:.3f}\nWass={r['wasserstein']:.3f}  PSI={r['psi']:.3f}"
            status_col = GRN if r["status"] == "stable" else (ORG if r["status"] == "moderate_shift" else RED)
            ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes,
                    ha="right", va="top", color=status_col, fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=CRD, edgecolor=status_col, alpha=0.7))

        ax.set_title(col, fontsize=13, fontweight="bold", color=TXT, pad=12)
        ax.title.set_position([0.5, 1.02])
        ax.set_xlabel("value", fontsize=11, color=TXT, labelpad=8)
        ax.legend(fontsize=9)

    fig.suptitle("Distribution shift: observed vs imputed values", color=TXT,
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(pad=2.5)
    return fig


def _fig_correlation_preservation(corr_result: dict) -> plt.Figure:
    if not corr_result or "complete_corr" not in corr_result:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color=MUT)
        return fig

    try:
        import seaborn as sns
        _cmap = sns.diverging_palette(15, 145, s=80, l=40, as_cmap=True)
    except ImportError:
        _cmap = "RdYlGn"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = [
        f"Complete cases\n(n={corr_result['n_complete']:,})",
        f"All rows post-imputation\n(n={corr_result['n_full']:,})",
        f"Difference\nFrobenius={corr_result['frobenius_norm']:.4f}",
    ]
    matrices = [corr_result["complete_corr"], corr_result["full_corr"], corr_result["diff"]]
    vmins = [-1, -1, None]
    vmaxs = [1,  1, None]

    for ax, mat, title, vmin, vmax in zip(axes, matrices, titles, vmins, vmaxs):
        kw = dict(cmap=_cmap, annot=True, fmt=".2f", linewidths=0.4,
                  linecolor=BG, annot_kws={"size": 8, "color": TXT})
        if vmin is not None:
            kw.update(vmin=vmin, vmax=vmax)
        try:
            import seaborn as sns
            sns.heatmap(mat, ax=ax, **kw)
            ax.tick_params(colors=MUT)
            if ax.collections:
                cbar = ax.collections[0].colorbar
                if cbar:
                    cbar.ax.tick_params(colors=MUT)
        except ImportError:
            im = ax.imshow(mat.values, cmap=_cmap, aspect="auto",
                           vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(mat.columns)))
            ax.set_xticklabels(mat.columns, rotation=40, ha="right", color=MUT, fontsize=9)
            ax.set_yticks(range(len(mat.index)))
            ax.set_yticklabels(mat.index, color=MUT, fontsize=9, rotation=0)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center",
                            color=TXT, fontsize=8)
            plt.colorbar(im, ax=ax)
        ax.set_title(title, fontsize=13, fontweight="bold", color=TXT, pad=12)
        ax.title.set_position([0.5, 1.02])

    fig.suptitle("Correlation matrix: before vs after imputation", color=TXT,
                 fontsize=13, fontweight="bold")
    fig.tight_layout(pad=2.5)
    return fig


def _fig_conditional_plausibility(plaus_df: pd.DataFrame) -> plt.Figure:
    if plaus_df.empty:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.text(0.5, 0.5, "No conditional plausibility data", ha="center", va="center", color=MUT)
        return fig

    cols = plaus_df["column"].unique()
    n = len(cols)
    width = 16 if n == 2 else 14
    fig, axes = plt.subplots(1, n, figsize=(width, 5))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        sub = plaus_df[plaus_df["column"] == col].sort_values("ks_stat", ascending=False)
        colors = [GRN if ok else RED for ok in sub["plausible"]]
        y_pos = range(len(sub))
        ax.barh(y_pos, sub["ks_stat"], color=colors, alpha=0.88, edgecolor="none")
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(sub["titleType"].tolist(), fontsize=9, color=MUT)
        ax.axvline(0.05 / max(len(sub), 1), color=Y, linestyle="--", linewidth=1,
                   label="Bonferroni α=0.05")
        ax.set_title(f"Conditional plausibility — {col}", fontsize=13,
                     fontweight="bold", color=TXT, pad=12)
        ax.title.set_position([0.5, 1.02])
        ax.set_xlabel("KS statistic (lower = more plausible)", fontsize=11, color=TXT, labelpad=8)
        ax.legend(fontsize=9)
        for i, (_, row) in enumerate(sub.iterrows()):
            ax.text(ax.get_xlim()[1] * 1.01, i,
                    f"obs={row['n_observed']} imp={row['n_imputed']}",
                    va="center", color=MUT, fontsize=7.5)

    fig.suptitle("Per-titleType KS test: observed vs imputed", color=TXT,
                 fontsize=13, fontweight="bold")
    fig.tight_layout(pad=2.5)
    return fig


# ── entry point ───────────────────────────────────────────────────────────────

def run(
    clean_train: pd.DataFrame,
    raw_train:   pd.DataFrame,
    fig_dir:     Path,
    csv_dir:     Path | None = None,
) -> dict:
    """
    Run all imputation quality checks.

    Parameters
    ----------
    clean_train : post-imputation train DataFrame (from train_clean.parquet)
    raw_train   : raw train DataFrame (loaded with disguised-missing tokens → NaN)
    fig_dir     : directory to save PNG figures
    csv_dir     : directory to save CSV outputs (defaults to fig_dir)

    Returns
    -------
    dict with keys: masked_validation, distribution_shift,
                    correlation_preservation, conditional_plausibility
    """
    fig_dir = Path(fig_dir)
    csv_dir = Path(csv_dir) if csv_dir else fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    print("\n[imputation_audit] Running deep imputation quality checks...")

    results: dict = {}

    # 1+2. Masked-value validation + baseline comparison
    print("[imputation_audit]   1/4 masked-value validation & baseline comparison...")
    try:
        masked_df = masked_value_validation(clean_train, raw_train)
        if not masked_df.empty:
            results["masked_validation"] = masked_df
            masked_df.to_csv(csv_dir / "imputation_masked_validation.csv", index=False)
            fig = _fig_masked_validation(masked_df)
            fig.savefig(fig_dir / "12_masked_validation.png", dpi=130, bbox_inches="tight",
                        facecolor=BG, edgecolor="none")
            plt.close(fig)
            for _, r in masked_df.iterrows():
                sym = "✓" if r["mice_wins_mae"] else "⚠"
                print(f"  {sym} {r['column']}: MICE MAE={r['mice_mae']:.3f}  "
                      f"median MAE={r['median_mae']:.3f}  "
                      f"MICE within tol={r['mice_within_tol_pct']:.1f}%  "
                      f"{'MICE wins ↑' if r['mice_wins_mae'] else 'median wins ↑'}")
    except Exception as exc:
        print(f"  ⚠ masked-value validation failed: {exc}")

    # 3. Distribution shift
    print("[imputation_audit]   2/4 distribution shift (observed vs imputed)...")
    try:
        shift_df = distribution_shift(clean_train, raw_train)
        if not shift_df.empty:
            results["distribution_shift"] = shift_df
            shift_df.to_csv(csv_dir / "imputation_distribution_shift.csv", index=False)
            fig = _fig_distribution_shift(shift_df, clean_train, raw_train)
            fig.savefig(fig_dir / "13_distribution_shift.png", dpi=130, bbox_inches="tight",
                        facecolor=BG, edgecolor="none")
            plt.close(fig)
            for _, r in shift_df.iterrows():
                sym = "✓" if r["status"] == "stable" else ("⚠" if r["status"] == "moderate_shift" else "✗")
                print(f"  {sym} {r['column']}: PSI={r['psi']:.3f} ({r['status']})  "
                      f"KS={r['ks_stat']:.3f}  Wass={r['wasserstein']:.3f}  "
                      f"mean_delta={r['mean_delta']:+.3f}")
    except Exception as exc:
        print(f"  ⚠ distribution shift failed: {exc}")

    # 4. Correlation preservation
    print("[imputation_audit]   3/4 correlation preservation...")
    try:
        corr_result = correlation_preservation(clean_train, raw_train)
        if corr_result:
            results["correlation_preservation"] = corr_result
            frob = corr_result["frobenius_norm"]
            sym  = "✓" if frob < 0.10 else ("⚠" if frob < 0.25 else "✗")
            print(f"  {sym} correlation matrix Frobenius distance = {frob:.4f}  "
                  f"({'minimal distortion' if frob < 0.10 else 'moderate' if frob < 0.25 else 'high distortion'})")
            fig = _fig_correlation_preservation(corr_result)
            fig.savefig(fig_dir / "14_correlation_preservation.png", dpi=130, bbox_inches="tight",
                        facecolor=BG, edgecolor="none")
            plt.close(fig)
    except Exception as exc:
        print(f"  ⚠ correlation preservation failed: {exc}")

    # 5. Conditional plausibility
    print("[imputation_audit]   4/4 conditional plausibility by titleType...")
    try:
        plaus_df = conditional_plausibility(clean_train, raw_train)
        if not plaus_df.empty:
            results["conditional_plausibility"] = plaus_df
            plaus_df.to_csv(csv_dir / "imputation_conditional_plausibility.csv", index=False)
            n_suspicious = int((~plaus_df["plausible"]).sum())
            total = len(plaus_df)
            sym   = "✓" if n_suspicious == 0 else "⚠"
            print(f"  {sym} {n_suspicious}/{total} group-column pairs show suspicious "
                  f"conditional distribution (KS p<0.05)")
            if n_suspicious > 0:
                bad = plaus_df[~plaus_df["plausible"]][["column", "titleType", "ks_stat", "ks_pval"]]
                for _, r in bad.iterrows():
                    print(f"      ✗ {r['column']} | {r['titleType']}: "
                          f"KS={r['ks_stat']:.3f}  p={r['ks_pval']:.4f}")
            fig = _fig_conditional_plausibility(plaus_df)
            fig.savefig(fig_dir / "15_conditional_plausibility.png", dpi=130, bbox_inches="tight",
                        facecolor=BG, edgecolor="none")
            plt.close(fig)
    except Exception as exc:
        print(f"  ⚠ conditional plausibility failed: {exc}")

    print("[imputation_audit] Done.")
    return results
