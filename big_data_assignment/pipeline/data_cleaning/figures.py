"""Pipeline figure utilities — no pipeline logic, only matplotlib.

Each public function accepts plain pandas DataFrames and returns a
matplotlib Figure.  Saving is handled by s9_report.
"""
from __future__ import annotations

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

# ── Shared constants ──────────────────────────────────────────────────────────

JOIN_COLS = [
    "genres", "titleType", "isAdult",
    "dir_count", "dir_avg_birth_year", "dir_min_birth_year", "dir_avg_death_year",
    "wri_count", "wri_avg_birth_year", "wri_min_birth_year", "wri_avg_death_year",
]

SPLIT_COLORS = {"train": BLU, "validation_hidden": ORG, "test_hidden": GRN}
SPLIT_LABELS = {"train": "train", "validation_hidden": "val", "test_hidden": "test"}

C_RAW   = RED
C_CLEAN = GRN
C_OK    = GRN
C_WARN  = ORG
C_FAIL  = RED

DOMAIN: dict[str, tuple[float, float]] = {
    "startYear":       (1880.0, 2025.0),
    "runtimeMinutes":  (1.0,    600.0),
    "numVotes_log1p":  (0.0,    15.0),
}

NUMERIC_COLS = ("startYear", "runtimeMinutes", "numVotes_log1p")

# ── Internal helpers ──────────────────────────────────────────────────────────

def _dark_table(tbl, n_headers: int, n_rows: int,
                highlight_rows: set[int] | None = None,
                highlight_cols: set[int] | None = None) -> None:
    """Apply dark-theme styling to a matplotlib table in-place."""
    for j in range(n_headers):
        cell = tbl[0, j]
        cell.set_facecolor("#1a1a1a")
        cell.set_text_props(color=Y, fontweight="bold")
        cell.set_edgecolor(BDR)

    for i in range(1, n_rows + 1):
        for j in range(n_headers):
            cell = tbl[i, j]
            highlighted = (
                (highlight_rows and (i - 1) in highlight_rows) or
                (highlight_cols and j in highlight_cols)
            )
            cell.set_facecolor("#1e2718" if highlighted else CRD)
            cell.set_text_props(color=TXT)
            cell.set_edgecolor(BDR)


def _no_data(msg: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            transform=ax.transAxes, color=MUT, fontsize=11)
    ax.set_axis_off()
    fig.tight_layout(pad=2.5)
    return fig


# ── 1. Missingness: raw vs cleaned ───────────────────────────────────────────

def missingness_comparison(
    raw_splits:   dict[str, pd.DataFrame],
    clean_splits: dict[str, pd.DataFrame],
    cols: tuple[str, ...] = ("startYear", "runtimeMinutes", "numVotes", "originalTitle"),
) -> plt.Figure:
    """Grouped bars: NULL % per column, raw vs cleaned, one panel per split."""
    splits = [s for s in ("train", "validation_hidden", "test_hidden") if s in raw_splits]
    n_splits = len(splits)
    width = 16 if n_splits == 2 else (18 if n_splits >= 3 else 14)
    fig, axes = plt.subplots(1, n_splits, figsize=(width, 5), sharey=True)
    if n_splits == 1:
        axes = [axes]

    x = np.arange(len(cols))
    w = 0.35

    for ax, split in zip(axes, splits):
        raw_df   = raw_splits[split]
        clean_df = clean_splits[split]

        raw_null   = [raw_df[c].isna().mean()   * 100 if c in raw_df.columns   else 0.0 for c in cols]
        clean_null = [clean_df[c].isna().mean() * 100 if c in clean_df.columns else 0.0 for c in cols]

        bars_raw   = ax.bar(x - w / 2, raw_null,   width=w, color=RED,   alpha=0.88,
                            edgecolor="none", label="raw")
        bars_clean = ax.bar(x + w / 2, clean_null, width=w, color=GRN, alpha=0.88,
                            edgecolor="none", label="cleaned")
        ax.set_xticks(x)
        labels_str = [c[:18] + "…" if len(c) > 18 else c for c in cols]
        ax.set_xticklabels(labels_str, rotation=35, ha="right", fontsize=9, color=MUT)
        ax.set_title(SPLIT_LABELS.get(split, split), fontsize=13, fontweight="bold",
                     color=TXT, pad=12)
        ax.title.set_position([0.5, 1.02])
        ax.set_ylim(0, 115)

        y_max = ax.get_ylim()[1]
        for bar in bars_raw:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005 * y_max,
                        f"{h:.0f}%", ha="center", va="bottom", fontsize=8, color=TXT)
        for bar in bars_clean:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005 * y_max,
                        f"{h:.0f}%", ha="center", va="bottom", fontsize=8, color=TXT)

    axes[0].set_ylabel("NULL %", fontsize=11, color=TXT, labelpad=8)
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Missingness — raw vs cleaned", fontsize=13, fontweight="bold", color=TXT)
    fig.tight_layout(pad=2.5)
    return fig


# ── 2. Numeric distributions (train / val / test overlaid) ───────────────────

def numeric_distributions(clean_splits: dict[str, pd.DataFrame]) -> plt.Figure:
    """Overlapping histograms for each numeric column across all three splits."""
    cols = [c for c in NUMERIC_COLS if any(c in df.columns for df in clean_splits.values())]
    n = len(cols)
    width = 18 if n >= 3 else (16 if n == 2 else 14)
    fig, axes = plt.subplots(1, n, figsize=(width, 5))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        for split, df in clean_splits.items():
            if col not in df.columns:
                continue
            vals  = pd.to_numeric(df[col], errors="coerce").dropna()
            label = f"{SPLIT_LABELS.get(split, split)} (n={len(vals):,})"
            ax.hist(vals, bins=40, alpha=0.35, edgecolor="none",
                    color=SPLIT_COLORS.get(split, MUT),
                    label=label, density=True)

        if col in DOMAIN:
            lo, hi = DOMAIN[col]
            ax.axvline(lo, color=Y, linestyle="--", linewidth=1)
            ax.axvline(hi, color=Y, linestyle="--", linewidth=1,
                       label=f"domain [{lo:.0f}, {hi:.0f}]")

        suffix = " [log LN(1+x)]" if col == "numVotes_log1p" else ""
        ax.set_title(f"{col}{suffix}", fontsize=13, fontweight="bold", color=TXT, pad=12)
        ax.title.set_position([0.5, 1.02])
        ax.set_xlabel(col, fontsize=11, color=TXT, labelpad=8)
        ax.set_ylabel("density", fontsize=11, color=TXT, labelpad=8)
        ax.legend(fontsize=9)

    fig.suptitle("Numeric distributions by split  (post-imputation)",
                 fontsize=13, fontweight="bold", color=TXT)
    fig.tight_layout(pad=2.5)
    return fig


# ── 3. Label balance ─────────────────────────────────────────────────────────

def label_balance(train_clean: pd.DataFrame) -> plt.Figure:
    """Class counts for the training split."""
    if "label" not in train_clean.columns or train_clean.empty:
        return _no_data("no label column in train")

    vc     = train_clean["label"].value_counts().sort_index()
    total  = vc.sum()
    palette = [GRN, RED, ORG, BLU]
    colors  = [palette[i % len(palette)] for i in range(len(vc))]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(vc.index.astype(str), vc.values, color=colors, alpha=0.88,
                  edgecolor="none", width=0.45)

    y_max = ax.get_ylim()[1]
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005 * y_max,
                f"{h:,}\n({h / total:.1%})", ha="center", va="bottom",
                fontsize=8.5, color=TXT)

    ax.set_ylabel("count", fontsize=11, color=TXT, labelpad=8)
    ax.set_title("Label class balance (train)", fontsize=13, fontweight="bold",
                 color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])
    fig.tight_layout(pad=2.5)
    return fig


# ── 4. Join coverage ─────────────────────────────────────────────────────────

def join_coverage(clean_splits: dict[str, pd.DataFrame]) -> plt.Figure:
    """Horizontal bars: % non-null for every column that originates from a JOIN."""
    train   = clean_splits.get("train", pd.DataFrame())
    present = [c for c in JOIN_COLS if c in train.columns]

    if not present:
        return _no_data("no joined columns found")

    coverage  = sorted([(c, (1 - train[c].isna().mean()) * 100) for c in present],
                       key=lambda x: x[1])
    col_names, vals = zip(*coverage)
    colors = [C_OK if v >= 90 else C_WARN if v >= 70 else C_FAIL for v in vals]

    fig, ax = plt.subplots(figsize=(14, max(5, len(present) * 0.52)))
    ax.barh(col_names, vals, color=colors, alpha=0.88, edgecolor="none")
    ax.axvline(90, color=MUT, linestyle="--", linewidth=1, label="90% threshold")
    ax.set_xlim(0, 112)
    ax.set_xlabel("% non-null (train)", fontsize=11, color=TXT, labelpad=8)
    ax.set_title("Join coverage — columns from LEFT JOINs", fontsize=13,
                 fontweight="bold", color=TXT, pad=12)
    ax.title.set_position([0.5, 1.02])

    y_max = ax.get_xlim()[1]
    for i, v in enumerate(vals):
        ax.text(v + 0.005 * y_max, i, f"{v:.1f}%", va="center",
                fontsize=8, color=TXT)

    ax.legend(fontsize=9)
    fig.tight_layout(pad=2.5)
    return fig


# ── 5. Domain bounds check ────────────────────────────────────────────────────

def domain_bounds(clean_splits: dict[str, pd.DataFrame]) -> plt.Figure:
    """Distributions with domain-valid range lines; violation counts in titles."""
    cols   = [c for c in NUMERIC_COLS if c in DOMAIN]
    splits = list(clean_splits.items())
    n_rows = len(cols)
    n_cols_grid = len(splits)

    fig, axes = plt.subplots(n_rows, n_cols_grid,
                             figsize=(16, 5 * n_rows),
                             sharey="row")

    if n_rows == 1 and n_cols_grid == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [list(axes)]
    elif n_cols_grid == 1:
        axes = [[ax] for ax in axes]

    for row, col in enumerate(cols):
        lo, hi = DOMAIN[col]
        for col_idx, (split, df) in enumerate(splits):
            ax = axes[row][col_idx]
            if col not in df.columns:
                ax.set_visible(False)
                continue

            vals    = pd.to_numeric(df[col], errors="coerce").dropna()
            n_bad   = int(((vals < lo) | (vals > hi)).sum())
            status  = "0 violations" if n_bad == 0 else f"{n_bad} violation{'s' if n_bad > 1 else ''}"
            ttl_col = C_OK if n_bad == 0 else C_FAIL

            ax.hist(vals, bins=40, alpha=0.35, edgecolor="none",
                    color=SPLIT_COLORS.get(split, MUT), density=True)
            ax.axvline(lo, color=Y, linestyle="--", linewidth=1, label=f"min={lo:.0f}")
            ax.axvline(hi, color=Y, linestyle="--", linewidth=1, label=f"max={hi:.0f}")
            ax.set_title(
                f"{SPLIT_LABELS.get(split, split)}  ·  {col}\n{status}",
                color=ttl_col, fontsize=9, fontweight="bold", pad=12,
            )
            ax.title.set_position([0.5, 1.02])
            ax.set_xlabel(col, fontsize=11, color=TXT, labelpad=8)
            if col_idx == 0:
                ax.legend(fontsize=7)

    fig.suptitle("Domain bounds  (gold dashed = valid range limits)",
                 fontsize=13, fontweight="bold", color=TXT)
    fig.tight_layout(pad=2.5)
    return fig


# ── 6. MICE imputation invariant ─────────────────────────────────────────────

def imputation_invariant(
    raw_splits:   dict[str, pd.DataFrame],
    clean_splits: dict[str, pd.DataFrame],
    cols: tuple[str, ...] = ("startYear", "runtimeMinutes"),
) -> plt.Figure:
    """Scatter raw vs cleaned for non-missing rows — points should lie on y = x."""
    plot_cols = [c for c in cols
                 if c in raw_splits.get("train", pd.DataFrame()).columns
                 and c in clean_splits.get("train", pd.DataFrame()).columns]

    if not plot_cols:
        return _no_data("no columns available for invariant check")

    n = len(plot_cols)
    width = 16 if n == 2 else (18 if n >= 3 else 14)
    fig, axes = plt.subplots(1, n, figsize=(width, 5))
    if n == 1:
        axes = [axes]

    raw_train   = raw_splits["train"]
    clean_train = clean_splits["train"]

    for ax, col in zip(axes, plot_cols):
        raw_col   = pd.to_numeric(raw_train[col],   errors="coerce")
        clean_col = pd.to_numeric(clean_train[col], errors="coerce")

        merged = (
            raw_train[["tconst"]].assign(raw=raw_col)
            .merge(clean_train[["tconst"]].assign(clean=clean_col), on="tconst", how="inner")
        )
        non_missing = merged[merged["raw"].notna()].dropna(subset=["raw", "clean"])

        if non_missing.empty:
            ax.text(0.5, 0.5, f"no non-missing rows for {col}",
                    ha="center", va="center", transform=ax.transAxes, color=MUT)
            continue

        lo = min(non_missing["raw"].min(), non_missing["clean"].min())
        hi = max(non_missing["raw"].max(), non_missing["clean"].max())
        ax.plot([lo, hi], [lo, hi], color="#444444", linestyle="--", linewidth=1,
                label="expected  y = x", zorder=5)

        sample = non_missing.sample(min(3000, len(non_missing)), random_state=42)
        ax.scatter(sample["raw"], sample["clean"],
                   alpha=0.7, s=40, linewidths=0, color=BLU, zorder=3)

        pct_ok = (non_missing["raw"].round() == non_missing["clean"]).mean()
        if pct_ok > 0.98:
            status, ttl_col = f"{pct_ok:.1%} preserved", C_OK
        else:
            status, ttl_col = f"only {pct_ok:.1%} preserved  (s7 alignment issue)", C_FAIL

        ax.set_xlabel(f"raw {col}  (non-missing rows only)", fontsize=11, color=TXT, labelpad=8)
        ax.set_ylabel(f"cleaned {col}", fontsize=11, color=TXT, labelpad=8)
        ax.set_title(f"MICE invariant — {col}\n{status}",
                     color=ttl_col, fontsize=9, fontweight="bold", pad=12)
        ax.title.set_position([0.5, 1.02])
        ax.legend(fontsize=9)
        ax.text(0.02, 0.97, f"n = {len(non_missing):,} non-missing rows",
                transform=ax.transAxes, fontsize=7.5, va="top", color=MUT)

    fig.suptitle(
        "MICE imputation invariant\n"
        "Non-missing raw values should be unchanged after imputation  →  points on y = x",
        fontsize=13, fontweight="bold", color=TXT,
    )
    fig.tight_layout(pad=2.5)
    return fig


# ── 7. Imputation justification table ────────────────────────────────────────

_IMPUTATION_MECHANISMS: dict[str, tuple[str, str]] = {
    "startYear":      ("MAR",  "Correlated with titleType & genres"),
    "runtimeMinutes": ("MAR",  "Correlated with titleType"),
    "numVotes":       ("MCAR", "Random source omission; log-transformed in s6 before imputation"),
    "originalTitle":  ("MCAR", "Rare — typically identical to primaryTitle"),
}

def imputation_summary(raw_splits: dict[str, pd.DataFrame]) -> plt.Figure:
    """Table: % missing per column per split, mechanism, and strategy."""
    check_cols = ("startYear", "runtimeMinutes", "numVotes", "originalTitle")
    splits     = [s for s in ("train", "validation_hidden", "test_hidden") if s in raw_splits]
    present    = [c for c in check_cols
                  if any(c in raw_splits[s].columns for s in splits)]

    rows: list[list[str]] = []
    highlight: set[int]   = set()
    for idx, col in enumerate(present):
        mech, rationale = _IMPUTATION_MECHANISMS.get(col, ("?", ""))
        pcts = []
        has_high = False
        for split in splits:
            df  = raw_splits[split]
            pct = df[col].isna().mean() * 100 if col in df.columns else float("nan")
            pcts.append(f"{pct:.1f}%")
            if pct > 5:
                has_high = True
        strategy = ("MICE (IterativeImputer, fit on train)"
                    if col in ("startYear", "runtimeMinutes", "numVotes") else "keep as-is")
        rows.append([col, mech] + pcts + [strategy, rationale])
        if has_high:
            highlight.add(idx)

    split_hdrs = [SPLIT_LABELS.get(s, s) + " miss%" for s in splits]
    headers    = ["column", "mechanism"] + split_hdrs + ["strategy", "rationale"]

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(14, max(5, n_rows * 0.45 + 1.8)))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    tbl = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#111111" if r > 0 else "#1a1a1a")
        cell.set_edgecolor(BDR)
        cell.set_text_props(color=TXT if r > 0 else Y)

    fig.suptitle("Imputation justification — per-column missingness & strategy",
                 fontsize=13, fontweight="bold", color=TXT, y=0.97)
    fig.tight_layout(pad=2.5)
    return fig


# ── 8. Missingness flag stability across splits ───────────────────────────────

def missingness_flags(
    raw_splits: dict[str, pd.DataFrame],
    cols: tuple[str, ...] = ("startYear", "runtimeMinutes", "numVotes"),
) -> plt.Figure:
    """Bar chart: % missing per column across splits — stable % suggests MCAR."""
    splits  = [s for s in ("train", "validation_hidden", "test_hidden") if s in raw_splits]
    present = [c for c in cols if any(c in raw_splits[s].columns for s in splits)]

    x = np.arange(len(present))
    w = 0.25
    fig, ax = plt.subplots(figsize=(14, 5))

    split_palette = [Y, ORG, BLU]
    for i, split in enumerate(splits):
        df   = raw_splits[split]
        pcts = [df[c].isna().mean() * 100 if c in df.columns else 0.0 for c in present]
        bars = ax.bar(x + i * w, pcts, width=w,
                      color=split_palette[i % len(split_palette)], alpha=0.88,
                      edgecolor="none", label=SPLIT_LABELS.get(split, split))
        y_max = ax.get_ylim()[1]
        for bar in bars:
            h = bar.get_height()
            if h > 0.5:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005 * y_max,
                        f"{h:.1f}%", ha="center", va="bottom", fontsize=8, color=TXT)

    ax.set_xticks(x + w)
    ax.set_xticklabels([f"{c}_was_missing" for c in present],
                       rotation=35, ha="right", fontsize=9, color=MUT)
    ax.set_ylabel("% missing in raw data", fontsize=11, color=TXT, labelpad=8)
    ax.set_title(
        "Missingness flag stability across splits\n"
        "(stable % → MCAR;  large drift → MAR/MNAR, check for data leakage)",
        fontsize=13, fontweight="bold", color=TXT, pad=12,
    )
    ax.title.set_position([0.5, 1.02])
    ax.legend(fontsize=9)
    fig.tight_layout(pad=2.5)
    return fig


# ── 9. Distributions by label class ──────────────────────────────────────────

_CLASS_COLORS = [GRN, RED, ORG, BLU]

def distributions_by_label(
    clean_train: pd.DataFrame,
    cols: tuple[str, ...] = NUMERIC_COLS,
) -> plt.Figure:
    """Overlapping histograms coloured by label — shows class separation per feature."""
    if "label" not in clean_train.columns or clean_train.empty:
        return _no_data("no label column in train")

    present = [c for c in cols if c in clean_train.columns]
    classes = sorted(clean_train["label"].unique())
    n = len(present)
    width = 18 if n >= 3 else (16 if n == 2 else 14)

    fig, axes = plt.subplots(1, n, figsize=(width, 5))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, present):
        for i, cls in enumerate(classes):
            vals = pd.to_numeric(
                clean_train.loc[clean_train["label"] == cls, col], errors="coerce"
            ).dropna()
            if vals.empty:
                continue
            color = _CLASS_COLORS[i % len(_CLASS_COLORS)]
            ax.hist(vals, bins=40, alpha=0.35, edgecolor="none", density=True,
                    color=color, label=f"label={cls}  (n={len(vals):,})")

        suffix = " [log LN(1+x)]" if col == "numVotes_log1p" else ""
        ax.set_title(f"{col}{suffix}", fontsize=13, fontweight="bold", color=TXT, pad=12)
        ax.title.set_position([0.5, 1.02])
        ax.set_xlabel(col, fontsize=11, color=TXT, labelpad=8)
        ax.set_ylabel("density", fontsize=11, color=TXT, labelpad=8)
        ax.legend(fontsize=9)

    fig.suptitle("Feature distributions by label class  (train only)",
                 fontsize=13, fontweight="bold", color=TXT)
    fig.tight_layout(pad=2.5)
    return fig


# ── 10. Hit / non-hit class separation scatter ───────────────────────────────

def class_separation(clean_train: pd.DataFrame) -> plt.Figure:
    """Scatter: numVotes_log1p × startYear and numVotes_log1p × runtimeMinutes."""
    if "label" not in clean_train.columns or clean_train.empty:
        return _no_data("no label column in train")

    pairs = [("numVotes_log1p", "startYear"), ("numVotes_log1p", "runtimeMinutes")]
    pairs = [(x, y) for x, y in pairs
             if x in clean_train.columns and y in clean_train.columns]

    if not pairs:
        return _no_data("required columns missing (numVotes_log1p, startYear, runtimeMinutes)")

    classes       = sorted(clean_train["label"].unique())
    class_palette = {cls: c for cls, c in zip(classes, [GRN, RED, ORG])}
    sample        = clean_train.sample(min(3000, len(clean_train)), random_state=42)

    n = len(pairs)
    width = 16 if n == 2 else 14
    fig, axes = plt.subplots(1, n, figsize=(width, 5))
    if n == 1:
        axes = [axes]

    for ax, (xcol, ycol) in zip(axes, pairs):
        for cls in classes:
            mask = sample["label"] == cls
            ax.scatter(
                pd.to_numeric(sample.loc[mask, xcol], errors="coerce"),
                pd.to_numeric(sample.loc[mask, ycol], errors="coerce"),
                alpha=0.7, s=40, linewidths=0,
                color=class_palette.get(cls, MUT),
                label=f"label={cls}",
            )
        ax.set_xlabel(f"{xcol} [log LN(1+x)]" if xcol == "numVotes_log1p" else xcol,
                      fontsize=11, color=TXT, labelpad=8)
        ax.set_ylabel(f"{ycol} [log LN(1+x)]" if ycol == "numVotes_log1p" else ycol,
                      fontsize=11, color=TXT, labelpad=8)
        ax.set_title(f"{xcol}  ×  {ycol}", fontsize=13, fontweight="bold", color=TXT, pad=12)
        ax.title.set_position([0.5, 1.02])
        ax.legend(fontsize=9, loc="upper left", framealpha=0.15, edgecolor=BDR, markerscale=3)
        ax.text(0.98, 0.02, f"n = {len(sample):,}", transform=ax.transAxes,
                ha="right", fontsize=7.5, color=MUT)

    fig.suptitle("Hit / non-hit class separation  (sample n=3 000 from train)",
                 fontsize=13, fontweight="bold", color=TXT)
    fig.tight_layout(pad=2.5)
    return fig


# ── 11. Outlier action table ──────────────────────────────────────────────────

_OUTLIER_DECISIONS: dict[str, str] = {
    "startYear":      "cap to domain [1880, 2025]",
    "runtimeMinutes": "keep (extremes are valid films)",
    "numVotes_log1p": "keep (log-transform already compresses heavy tail)",
}

def outlier_summary(clean_splits: dict[str, pd.DataFrame]) -> plt.Figure:
    """Table: 3×IQR bounds + extreme counts + decision for each numeric column (train)."""
    train = clean_splits.get("train", pd.DataFrame())

    rows: list[list[str]] = []
    highlight: set[int]   = set()
    for idx, col in enumerate(NUMERIC_COLS):
        if col not in train.columns:
            continue
        vals    = pd.to_numeric(train[col], errors="coerce").dropna()
        q1, q3  = float(vals.quantile(0.25)), float(vals.quantile(0.75))
        iqr     = q3 - q1
        lo, hi  = q1 - 3 * iqr, q3 + 3 * iqr
        n_below = int((vals < lo).sum())
        n_above = int((vals > hi).sum())
        n_ext   = n_below + n_above
        rows.append([
            col,
            f"{q1:.2f}", f"{q3:.2f}", f"{iqr:.2f}",
            f"{lo:.2f}", f"{hi:.2f}",
            str(n_below), str(n_above), str(n_ext),
            _OUTLIER_DECISIONS.get(col, "review"),
        ])
        if n_ext > 0:
            highlight.add(idx)

    headers = ["column", "Q1", "Q3", "IQR",
               "lower_3IQR", "upper_3IQR",
               "n_below", "n_above", "n_extreme", "decision"]

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(14, max(5, n_rows * 0.45 + 2.0)))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    tbl = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#111111" if r > 0 else "#1a1a1a")
        cell.set_edgecolor(BDR)
        cell.set_text_props(color=TXT if r > 0 else Y)

    fig.suptitle("Outlier action table — 3×IQR rule on training split",
                 fontsize=13, fontweight="bold", color=TXT, y=0.97)
    fig.tight_layout(pad=2.5)
    return fig
