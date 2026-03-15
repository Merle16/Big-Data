"""Step 9: Post-pipeline validity checks + figure output.

Loads raw CSVs and the output Parquet files, runs all checks, prints a
pass / warn / fail summary to stdout, and saves one PNG per diagnostic
to <out_dir>/pipeline_figures/.

Called automatically from __init__.run_pipeline() after all Parquets are
written.  Can also be called standalone:

    from pipeline.data_cleaning.s9_report import run
    run(out_paths, raw_csv_dir)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from . import figures as figs
from . import imputation_audit as imp_audit
from . import join_audit as jn_audit
from .s1_missing import DISGUISED_TOKENS

# ── Config ────────────────────────────────────────────────────────────────────

# Columns where MICE should preserve existing (non-missing) values.
# numVotes_log1p excluded: raw CSV has "numVotes" (integer), cleaned has
# "numVotes_log1p" (float after s6) — direct comparison would be meaningless.
_INVARIANT_COLS = ("startYear", "runtimeMinutes")

_DOMAIN      = figs.DOMAIN
_JOIN_COLS   = figs.JOIN_COLS
_IMPUTE_COLS = figs.NUMERIC_COLS
SPLIT_LABELS = figs.SPLIT_LABELS


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_raw(raw_csv_dir: Path) -> dict[str, pd.DataFrame]:
    """Read raw CSVs with disguised-missing tokens converted to NaN."""
    d = Path(raw_csv_dir)
    raw: dict[str, pd.DataFrame] = {}
    na_vals = list(DISGUISED_TOKENS)

    train_paths = sorted(d.glob("train-*.csv"))
    if train_paths:
        raw["train"] = pd.concat(
            [pd.read_csv(p, na_values=na_vals, keep_default_na=False) for p in train_paths],
            ignore_index=True,
        )

    for split, fname in [
        ("validation_hidden", "validation_hidden.csv"),
        ("test_hidden",       "test_hidden.csv"),
    ]:
        fp = d / fname
        if fp.exists():
            raw[split] = pd.read_csv(fp, na_values=na_vals, keep_default_na=False)

    return raw


# ── Individual checks (each returns a list of formatted strings) ──────────────

def _check_row_counts(
    raw_splits: dict[str, pd.DataFrame],
    clean_splits: dict[str, pd.DataFrame],
) -> list[str]:
    lines = []
    for split, clean_df in clean_splits.items():
        raw_n   = len(raw_splits.get(split, pd.DataFrame()))
        clean_n = len(clean_df)
        ok      = raw_n == clean_n
        sym     = "✓" if ok else "✗"
        note    = "" if ok else "  ← rows dropped during pipeline"
        lines.append(f"  {sym} row count [{split}]:  raw={raw_n:,}  →  cleaned={clean_n:,}{note}")
    return lines


def _check_remaining_nulls(clean_splits: dict[str, pd.DataFrame]) -> list[str]:
    lines = []
    for split, df in clean_splits.items():
        for col in _IMPUTE_COLS:
            if col not in df.columns:
                continue
            n   = int(df[col].isna().sum())
            sym = "✓" if n == 0 else "✗"
            note = "" if n == 0 else "  ← MICE did not fully impute"
            lines.append(f"  {sym} remaining NaN after imputation [{split}] {col}: {n}{note}")
    return lines


def _check_domain(clean_splits: dict[str, pd.DataFrame]) -> list[str]:
    lines = []
    for split, df in clean_splits.items():
        for col, (lo, hi) in _DOMAIN.items():
            if col not in df.columns:
                continue
            vals  = pd.to_numeric(df[col], errors="coerce").dropna()
            n_bad = int(((vals < lo) | (vals > hi)).sum())
            sym   = "✓" if n_bad == 0 else "✗"
            note  = "" if n_bad == 0 else f"  ← values outside [{lo:.0f}, {hi:.0f}]"
            lines.append(f"  {sym} domain bounds [{split}] {col}: {n_bad} violation(s){note}")
    return lines


def _check_invariant(
    raw_splits:   dict[str, pd.DataFrame],
    clean_splits: dict[str, pd.DataFrame],
) -> list[str]:
    """For rows where the raw value was NOT missing, verify MICE left it unchanged."""
    lines = []
    raw_train   = raw_splits.get("train")
    clean_train = clean_splits.get("train")
    if raw_train is None or clean_train is None:
        return ["  ? MICE invariant: training data unavailable"]

    for col in _INVARIANT_COLS:
        if col not in raw_train.columns or col not in clean_train.columns:
            continue

        raw_col = pd.to_numeric(raw_train[col], errors="coerce")
        merged  = (
            raw_train[["tconst"]].assign(raw=raw_col)
            .merge(
                clean_train[["tconst", col]].rename(columns={col: "clean"}),
                on="tconst", how="inner",
            )
        )
        non_missing = merged[merged["raw"].notna()]
        non_missing = non_missing.assign(clean=pd.to_numeric(non_missing["clean"], errors="coerce"))
        non_missing = non_missing.dropna(subset=["raw", "clean"])

        if non_missing.empty:
            lines.append(f"  ? MICE invariant [{col}]: no non-missing rows to check")
            continue

        pct = (non_missing["raw"].round() == non_missing["clean"]).mean()
        if pct > 0.98:
            lines.append(f"  ✓ MICE invariant [{col}]: {pct:.1%} of non-missing values preserved")
        else:
            lines.append(
                f"  ✗ MICE invariant [{col}]: only {pct:.1%} preserved"
                f"  ← row-alignment bug in s7 (ROW_NUMBER() OVER () without ORDER BY)"
            )
    return lines


def _check_join_coverage(clean_splits: dict[str, pd.DataFrame]) -> list[str]:
    lines = []
    train = clean_splits.get("train", pd.DataFrame())
    for col in _JOIN_COLS:
        if col not in train.columns:
            continue
        pct = (1 - train[col].isna().mean()) * 100
        if pct >= 90:
            sym  = "✓"
            note = ""
        elif pct >= 70:
            sym  = "⚠"
            note = "  ← coverage below 90%"
        else:
            sym  = "✗"
            note = "  ← low coverage; check JOIN key or source table"
        lines.append(f"  {sym} join coverage [{col}]: {pct:.1f}%{note}")
    return lines


def _check_missingness_stability(raw_splits: dict[str, pd.DataFrame]) -> list[str]:
    """Check that missingness rates are consistent across train/val/test.

    Large drift (>5 pp) suggests MAR/MNAR or a data-collection difference
    that could hurt generalisation.
    """
    lines  = []
    splits = [s for s in ("train", "validation_hidden", "test_hidden") if s in raw_splits]
    for col in _IMPUTE_COLS:
        pcts = {}
        for split in splits:
            df = raw_splits[split]
            if col in df.columns:
                pcts[split] = df[col].isna().mean() * 100
        if len(pcts) < 2:
            continue
        vals  = list(pcts.values())
        drift = max(vals) - min(vals)
        detail = "  ".join(
            f"{SPLIT_LABELS.get(s, s)}={v:.1f}%" for s, v in pcts.items()
        )
        if drift <= 5:
            lines.append(f"  ✓ missingness stability [{col}]: drift={drift:.1f}pp  ({detail})")
        else:
            lines.append(
                f"  ⚠ missingness stability [{col}]: drift={drift:.1f}pp  ({detail})"
                f"  ← consider MNAR / split-specific bias"
            )
    return lines


def _check_outliers(clean_splits: dict[str, pd.DataFrame]) -> list[str]:
    """3×IQR extreme counts per numeric column on the train split."""
    lines = []
    train = clean_splits.get("train", pd.DataFrame())
    for col in _IMPUTE_COLS:
        if col not in train.columns:
            continue
        vals    = pd.to_numeric(train[col], errors="coerce").dropna()
        q1, q3  = float(vals.quantile(0.25)), float(vals.quantile(0.75))
        iqr     = q3 - q1
        lo, hi  = q1 - 3 * iqr, q3 + 3 * iqr
        n_below = int((vals < lo).sum())
        n_above = int((vals > hi).sum())
        n_ext   = n_below + n_above
        pct_ext = n_ext / len(vals) * 100
        sym     = "✓" if n_ext == 0 else "⚠"
        lines.append(
            f"  {sym} outliers 3×IQR [train] {col}: "
            f"n_below={n_below}  n_above={n_above}  "
            f"total={n_ext} ({pct_ext:.2f}%)  "
            f"bounds=[{lo:.1f}, {hi:.1f}]"
        )
    return lines


def _check_label_balance(clean_splits: dict[str, pd.DataFrame]) -> list[str]:
    lines = []
    train = clean_splits.get("train", pd.DataFrame())
    if "label" not in train.columns or train.empty:
        return ["  ? label balance: no label column in train"]
    vc    = train["label"].value_counts(normalize=True)
    minor = float(vc.min())
    sym   = "✓" if minor >= 0.35 else "⚠"
    note  = "" if minor >= 0.35 else "  ← minority class below 35%"
    lines.append(f"  {sym} label balance (train): {vc.to_dict()}  minority={minor:.1%}{note}")
    return lines


# ── Figure saving helper ──────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[s9]   → {path.name}")


# ── Public entry point ────────────────────────────────────────────────────────

def run(
    out_paths:   dict[str, Path],
    raw_csv_dir: Path,
    fig_dir:     Path | None = None,
) -> None:
    """Run all validity checks, print results, and save PNGs to fig_dir.

    Parameters
    ----------
    out_paths   : dict returned by run_pipeline() — split → parquet path
    raw_csv_dir : directory containing train-*.csv, validation_hidden.csv, test_hidden.csv
    fig_dir     : output folder for PNGs (default: <processed_dir>/pipeline_figures/)
    """
    # Default figure dir next to the Parquet files
    if fig_dir is None:
        fig_dir = Path(next(iter(out_paths.values()))).parent / "pipeline_figures"
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    raw_splits   = _load_raw(raw_csv_dir)
    clean_splits = {split: pd.read_parquet(path) for split, path in out_paths.items()}

    # ── Validity checks ───────────────────────────────────────────────────────
    print("\n[s9] POST-PIPELINE VALIDITY CHECKS")
    print("─" * 64)

    all_lines = (
        _check_row_counts(raw_splits, clean_splits)
        + [""]
        + _check_remaining_nulls(clean_splits)
        + [""]
        + _check_domain(clean_splits)
        + [""]
        + _check_invariant(raw_splits, clean_splits)
        + [""]
        + _check_missingness_stability(raw_splits)
        + [""]
        + _check_outliers(clean_splits)
        + [""]
        + _check_join_coverage(clean_splits)
        + [""]
        + _check_label_balance(clean_splits)
    )
    for line in all_lines:
        print(line)

    # ── Figures ───────────────────────────────────────────────────────────────
    print(f"\n[s9] Writing figures to: {fig_dir}")

    train_clean = clean_splits.get("train", pd.DataFrame())

    _save(figs.missingness_comparison(raw_splits, clean_splits),   fig_dir / "01_missingness.png")
    _save(figs.numeric_distributions(clean_splits),                fig_dir / "02_distributions.png")
    _save(figs.label_balance(train_clean),                         fig_dir / "03_label_balance.png")
    _save(figs.join_coverage(clean_splits),                        fig_dir / "04_join_coverage.png")
    _save(figs.domain_bounds(clean_splits),                        fig_dir / "05_domain_bounds.png")
    _save(figs.imputation_invariant(raw_splits, clean_splits),     fig_dir / "06_imputation_invariant.png")
    _save(figs.imputation_summary(raw_splits),                     fig_dir / "07_imputation_summary.png")
    _save(figs.missingness_flags(raw_splits),                      fig_dir / "08_missingness_flags.png")
    _save(figs.distributions_by_label(train_clean),                fig_dir / "09_distributions_by_label.png")
    _save(figs.class_separation(train_clean),                      fig_dir / "10_class_separation.png")
    _save(figs.outlier_summary(clean_splits),                      fig_dir / "11_outlier_summary.png")

    # ── Imputation audit (figs 12–15) ─────────────────────────────────────────
    raw_train   = raw_splits.get("train")
    clean_train = clean_splits.get("train")
    if raw_train is not None and clean_train is not None:
        print("\n[s9] Running imputation audit …")
        imp_audit.run(
            clean_train=clean_train,
            raw_train=raw_train,
            fig_dir=fig_dir,
            csv_dir=fig_dir.parent,
        )

    # ── Join / fanout audit (figs 16–18) ──────────────────────────────────────
    print("\n[s9] Running join/fanout audit …")
    jn_audit.run(
        raw_splits=raw_splits,
        clean_splits=clean_splits,
        fig_dir=fig_dir,
        csv_dir=fig_dir.parent,
    )

    n = len(list(fig_dir.glob("*.png")))
    print(f"[s9] {n} figures saved.")
