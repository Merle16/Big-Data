"""Generate a self-contained HTML findings report from pipeline outputs.

Reads:
  - data/processed/*_clean.parquet
  - data/raw/csv/   (for raw-side checks)
  - data/processed/pipeline_figures/*.png

Writes:
  - data/processed/pipeline_report.html

Usage:
    python -m big_data_assignment.pipeline.data_cleaning.make_html_report
    python big_data_assignment/pipeline/data_cleaning/make_html_report.py
"""
from __future__ import annotations

import base64
import io
import sys
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT     = Path(__file__).resolve().parents[2]
_PIPELINE = Path(__file__).resolve().parents[1]
_PROC     = _ROOT / "data" / "processed"
_RAW_CSV  = _ROOT / "data" / "raw" / "csv"
_FIG_DIR  = _PIPELINE / "outputs" / "cleaning"
_OUT_HTML = _PIPELINE / "outputs" / "cleaning" / "pipeline_report.html"

_SPLITS = ("train", "validation_hidden", "test_hidden")

# ── Figure captions ───────────────────────────────────────────────────────────

_CAPTIONS: dict[str, str] = {
    "01_missingness.png":           "Missingness — raw vs cleaned NULL % per column",
    "02_distributions.png":         "Numeric distributions by split (post-imputation)",
    "03_label_balance.png":         "Label class balance (train)",
    "04_join_coverage.png":         "Join coverage — % non-null for LEFT JOIN columns",
    "05_domain_bounds.png":         "Domain bounds — valid-range violations per column",
    "06_imputation_invariant.png":  "MICE invariant — non-missing raw values vs cleaned",
    "07_imputation_summary.png":    "Imputation justification table",
    "08_missingness_flags.png":     "Missingness flag stability across splits",
    "09_distributions_by_label.png": "Feature distributions by label class (train)",
    "10_class_separation.png":      "Hit / non-hit class separation scatter",
    "11_outlier_summary.png":       "Outlier action table (3×IQR, train split)",
}

# ── Data loading ──────────────────────────────────────────────────────────────

def _load_outputs() -> tuple[dict, dict]:
    out_paths = {
        s: _PROC / f"{s}_clean.parquet"
        for s in _SPLITS
        if (_PROC / f"{s}_clean.parquet").exists()
    }
    clean_splits = {s: pd.read_parquet(p) for s, p in out_paths.items()}
    return out_paths, clean_splits


# ── Capture check output ──────────────────────────────────────────────────────

def _run_checks(out_paths: dict, clean_splits: dict) -> str:
    from .s9_report import _load_raw, _check_row_counts, _check_remaining_nulls
    from .s9_report import _check_domain, _check_invariant, _check_missingness_stability
    from .s9_report import _check_outliers, _check_join_coverage, _check_label_balance

    raw_splits = _load_raw(_RAW_CSV)

    buf = io.StringIO()
    with redirect_stdout(buf):
        sections = [
            ("Row count conservation",       _check_row_counts(raw_splits, clean_splits)),
            ("Remaining NaN after imputation", _check_remaining_nulls(clean_splits)),
            ("Post-imputation domain bounds", _check_domain(clean_splits)),
            ("MICE invariant",                _check_invariant(raw_splits, clean_splits)),
            ("Missingness stability",          _check_missingness_stability(raw_splits)),
            ("Outlier counts (3×IQR, train)", _check_outliers(clean_splits)),
            ("Join coverage",                  _check_join_coverage(clean_splits)),
            ("Label balance",                  _check_label_balance(clean_splits)),
        ]
        for title, lines in sections:
            print(f"\n{title}")
            print("─" * len(title))
            for line in lines:
                print(line)

    return buf.getvalue()


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _png_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _colorise_line(line: str) -> str:
    """Wrap check-result lines in a coloured <span>."""
    if line.strip().startswith("✓"):
        return f'<span class="ok">{line}</span>'
    if line.strip().startswith("✗"):
        return f'<span class="fail">{line}</span>'
    if line.strip().startswith("⚠"):
        return f'<span class="warn">{line}</span>'
    if line.strip().startswith("?"):
        return f'<span class="info">{line}</span>'
    return line


_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: "Segoe UI", system-ui, sans-serif;
    font-size: 14px;
    background: #f8f9fa;
    color: #212529;
    padding: 2rem;
    max-width: 1400px;
    margin: auto;
}
h1 { font-size: 1.8rem; margin-bottom: 0.25rem; }
h2 {
    font-size: 1.15rem;
    margin: 2rem 0 0.75rem;
    padding-bottom: 0.3rem;
    border-bottom: 2px solid #dee2e6;
    color: #495057;
}
.meta { color: #6c757d; font-size: 0.85rem; margin-bottom: 1.5rem; }
pre {
    background: #1e1e1e;
    color: #d4d4d4;
    padding: 1rem 1.25rem;
    border-radius: 6px;
    font-size: 12.5px;
    line-height: 1.55;
    overflow-x: auto;
    white-space: pre-wrap;
}
.ok   { color: #4ec9b0; }
.fail { color: #f48771; font-weight: 600; }
.warn { color: #dcdcaa; }
.info { color: #9cdcfe; }
.fig-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(560px, 1fr));
    gap: 1.5rem;
    margin-top: 1rem;
}
figure {
    background: #fff;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
figure img { width: 100%; display: block; }
figcaption {
    padding: 0.5rem 0.75rem;
    font-size: 12px;
    color: #6c757d;
    border-top: 1px solid #dee2e6;
    background: #f8f9fa;
}
"""


def _build_html(check_text: str) -> str:
    today = date.today().isoformat()

    # ── Checks section ────────────────────────────────────────────────────────
    coloured_lines = "\n".join(_colorise_line(ln) for ln in check_text.splitlines())
    checks_html = f"<pre>{coloured_lines}</pre>"

    # ── Figures section ───────────────────────────────────────────────────────
    fig_items: list[str] = []
    for fname, caption in _CAPTIONS.items():
        fpath = _FIG_DIR / fname
        if not fpath.exists():
            continue
        b64 = _png_to_b64(fpath)
        fig_items.append(
            f'<figure>'
            f'<img src="data:image/png;base64,{b64}" alt="{caption}">'
            f'<figcaption>{caption}</figcaption>'
            f'</figure>'
        )
    figures_html = f'<div class="fig-grid">{"".join(fig_items)}</div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pipeline Data Cleaning Report</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Pipeline Data Cleaning Report</h1>
<p class="meta">Generated: {today} &nbsp;·&nbsp; {len(fig_items)} figures &nbsp;·&nbsp; pipeline/data_cleaning</p>
<h2>Validity Checks</h2>
{checks_html}
<h2>Figures</h2>
{figures_html}
</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────

def run(out_html: Path = _OUT_HTML) -> Path:
    out_paths, clean_splits = _load_outputs()
    if not out_paths:
        print("[report] No cleaned Parquet files found — run the pipeline first.")
        sys.exit(1)

    check_text  = _run_checks(out_paths, clean_splits)
    html        = _build_html(check_text)

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    print(f"[report] Written: {out_html}  ({len(html) // 1024} KB)")
    return out_html


if __name__ == "__main__":
    run()
