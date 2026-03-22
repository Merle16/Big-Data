#!/usr/bin/env python3
"""
Theme 01 — Repository Audit
============================
Shows:
  • Which required files exist / are missing
  • Train shard count and sizes
  • Known broken paths in shared repository code
  • Self-contained HTML report

Reads  : filesystem only
Writes : outputs/theme_01_repository_audit.html
"""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT   = Path(__file__).resolve().parents[2]
MEMBER = Path(__file__).resolve().parent
OUT    = MEMBER / "outputs_restart"
OUT.mkdir(exist_ok=True)
STATE  = OUT / "state.pkl"

Y="#F5C518"; B="#1848f5"; BG="#0a0a0a"; CRD="#141414"
TXT="#ffffff"; MUT="#888"; GRN="#2ecc71"; RED="#e74c3c"; ORG="#f39c12"

CSS = f"""
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:{BG};color:{TXT};font-family:'Segoe UI',sans-serif;padding:24px;line-height:1.6}}
h1{{color:{Y};font-size:2rem;margin-bottom:6px}}
h2{{color:{Y};font-size:1.3rem;margin:0 0 12px}}
h3{{color:{ORG};font-size:1rem;margin:12px 0 6px}}
.subtitle{{color:{MUT};margin-bottom:28px;font-size:.95rem}}
.card{{background:{CRD};border:1px solid #222;border-radius:12px;padding:20px;margin-bottom:24px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px}}
.kpi{{background:{CRD};border:1px solid #2a2a2a;border-radius:10px;padding:16px;text-align:center}}
.kpi .val{{font-size:2rem;font-weight:700;color:{Y}}}
.kpi .lbl{{color:{MUT};font-size:.8rem;margin-top:4px}}
table{{width:100%;border-collapse:collapse;font-size:.82rem}}
th{{background:{Y};color:#111;padding:8px 10px;text-align:left}}
td{{padding:7px 10px;border-bottom:1px solid #222}}
tr:hover td{{background:#1e1e1e}}
img{{max-width:100%;border-radius:8px;margin-top:8px}}
.note{{background:#1a1a2e;border-left:4px solid {Y};padding:10px 14px;
       border-radius:4px;font-size:.85rem;margin:10px 0;color:#ccc}}
.ok{{color:{GRN};font-weight:600}} .fail{{color:{RED};font-weight:600}}
.warn{{color:{ORG};font-weight:600}}
.badge{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.75rem;font-weight:600}}
.badge-ok{{background:{GRN};color:#111}} .badge-fail{{background:{RED};color:#fff}}
.badge-warn{{background:{ORG};color:#111}}
"""

def _b64(fig):
    buf=BytesIO(); fig.savefig(buf,format="png",dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(fig); return base64.b64encode(buf.getvalue()).decode()
def _img(fig): return f'<img src="data:image/png;base64,{_b64(fig)}">'
def _card(title,body,color=Y): return f'<div class="card"><h2 style="color:{color}">{title}</h2>{body}</div>'
def _kpi(val,lbl): return f'<div class="kpi"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>'
def _page(title,subtitle,kpis,sections):
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{title}</title><style>{CSS}</style></head><body>
<h1>{title}</h1><p class="subtitle">{subtitle}</p>
<div class="kpi-grid">{kpis}</div>{"".join(sections)}</body></html>"""


ISSUE_FIX_LOG = [
    {"Issue": "Missing src.models package",
     "Why it breaks": "python -m src.pipeline.baseline → ModuleNotFoundError",
     "Fix applied": "Independent pipeline in theme_*.py files — no shared code dependency"},
    {"Issue": "config.yaml points to imdb_train.csv / imdb_validation.csv (absent)",
     "Why it breaks": "train.py fails with FileNotFoundError before any model step",
     "Fix applied": "Loads shard files from data/raw/csv/train-*.csv directly"},
    {"Issue": "predict.py raises NotImplementedError by design",
     "Why it breaks": "Shared predict path is non-functional",
     "Fix applied": "Full train→validate→predict flow in theme_08 + theme_11"},
    {"Issue": "JSON util expects review/label schema",
     "Why it breaks": "IMDB JSON has movie-person many-to-many, not review rows",
     "Fix applied": "DuckDB json_each() reconstruction in theme_03"},
    {"Issue": "movie_directors.csv has double-escaped \\\\N tokens",
     "Why it breaks": "Simple '\\N' replace misses them; dirty IDs survive",
     "Fix applied": "Normalize full MISSING_TOKENS set including \\\\N in theme_04"},
    {"Issue": "many_to_many.ipynb is empty (0 bytes)",
     "Why it breaks": "No reproducible implementation in that file",
     "Fix applied": "Reproducible DuckDB reconstruction in theme_03"},
]


def _check(p: Path) -> dict:
    exists = p.exists()
    size   = p.stat().st_size if exists else 0
    return {"path": str(p.relative_to(ROOT)), "exists": exists, "size_bytes": size}


def _fig_shard_sizes(shard_paths: List[Path]) -> str:
    sizes_kb = [p.stat().st_size / 1024 for p in shard_paths]
    names    = [p.name for p in shard_paths]
    fig, ax  = plt.subplots(figsize=(10, 4), facecolor=BG)
    ax.set_facecolor(CRD)
    ax.bar(names, sizes_kb, color=Y)
    ax.set_title("Train shard file sizes (KB)", color=TXT, fontsize=12)
    ax.set_xlabel("Shard file", color=TXT)
    ax.set_ylabel("KB", color=TXT)
    ax.tick_params(colors=TXT, axis="x", rotation=30, labelsize=8)
    ax.tick_params(colors=TXT, axis="y")
    for spine in ax.spines.values(): spine.set_color(MUT)
    ax.grid(axis="y", alpha=0.15, color=MUT)
    fig.tight_layout()
    return _img(fig)


def _fig_file_status(checks: dict) -> str:
    labels = list(checks.keys())
    colors_bar = [GRN if v["exists"] else RED for v in checks.values()]
    sizes  = [max(v["size_bytes"]/1024, 0.5) if v["exists"] else 0 for v in checks.values()]
    fig, ax = plt.subplots(figsize=(10, max(4, len(labels)*0.45)), facecolor=BG)
    ax.set_facecolor(CRD)
    ax.barh(labels, sizes, color=colors_bar)
    ax.set_title("Required files — size (KB) if present, red if missing", color=TXT, fontsize=12)
    ax.set_xlabel("KB", color=TXT)
    ax.tick_params(colors=TXT, labelsize=8)
    for spine in ax.spines.values(): spine.set_color(MUT)
    ax.grid(axis="x", alpha=0.15, color=MUT)
    fig.tight_layout()
    return _img(fig)


def run(state: dict) -> dict:
    raw_csv  = ROOT / "data" / "raw" / "csv"
    raw_json = ROOT / "data" / "raw" / "json"

    shard_paths = sorted(raw_csv.glob("train-*.csv"))

    checks = {
        "train shards (×{})".format(len(shard_paths)):
            {"exists": len(shard_paths) > 0, "size_bytes": sum(p.stat().st_size for p in shard_paths)},
        "validation_hidden.csv":   _check(raw_csv  / "validation_hidden.csv"),
        "test_hidden.csv":         _check(raw_csv  / "test_hidden.csv"),
        "movie_directors.csv":     _check(raw_csv  / "movie_directors.csv"),
        "movie_writers.csv":       _check(raw_csv  / "movie_writers.csv"),
        "directing.json":          _check(raw_json / "directing.json"),
        "writing.json":            _check(raw_json / "writing.json"),
        "shared/src/pipeline/train.py":   _check(ROOT / "src" / "pipeline" / "train.py"),
        "shared/src/pipeline/predict.py": _check(ROOT / "src" / "pipeline" / "predict.py"),
        "shared/config/config.yaml":      _check(ROOT / "config" / "config.yaml"),
    }

    # Member folder structure (djamel)
    theme_files  = sorted(MEMBER.glob("theme_*.py"))
    nb_files     = sorted((MEMBER / "notebooks").glob("*.ipynb")) if (MEMBER / "notebooks").exists() else []

    n_ok      = sum(1 for v in checks.values() if v["exists"])
    n_missing = len(checks) - n_ok

    kpis = (
        _kpi(len(shard_paths),      "Train shards found")
      + _kpi(f"{n_ok}/{len(checks)}", "Required files<br>present")
      + _kpi(n_missing,             "Files missing<br>or broken")
      + _kpi(len(ISSUE_FIX_LOG),    "Known issues<br>documented & fixed")
    )

    # table of file checks
    tbl_rows = ""
    for name, v in checks.items():
        badge = f'<span class="badge badge-ok">EXISTS</span>' if v["exists"] \
                else f'<span class="badge badge-fail">MISSING</span>'
        size = f'{v["size_bytes"]/1024:.1f} KB' if v["exists"] else "—"
        tbl_rows += f"<tr><td>{name}</td><td>{badge}</td><td>{size}</td></tr>"
    file_table = f"<table><tr><th>File</th><th>Status</th><th>Size</th></tr>{tbl_rows}</table>"

    # issue-fix table
    issue_rows = ""
    for row in ISSUE_FIX_LOG:
        issue_rows += (f"<tr><td>{row['Issue']}</td><td style='color:{RED}'>{row['Why it breaks']}</td>"
                       f"<td style='color:{GRN}'>{row['Fix applied']}</td></tr>")
    issue_table = (f"<table><tr><th>Issue found</th><th>Why it breaks</th><th>How fixed</th></tr>"
                   f"{issue_rows}</table>")

    # Member folder listing
    _run_fp = MEMBER / "run_all_themes.py"
    _run_size = f"{_run_fp.stat().st_size/1024:.1f} KB" if _run_fp.exists() else "—"
    theme_rows = "".join(
        f"<tr><td><code>{f.name}</code></td><td>{f.stat().st_size/1024:.1f} KB</td></tr>"
        for f in theme_files
    )
    nb_rows = "".join(
        f"<tr><td><code>notebooks/{f.name}</code></td><td>{f.stat().st_size/1024:.1f} KB</td></tr>"
        for f in nb_files
    )
    member_table = f"""<table>
<tr><th>File</th><th>Size</th></tr>
{theme_rows}{nb_rows}
<tr><td><code>run_all_themes.py</code></td>
    <td>{_run_size}</td></tr>
</table>"""

    sections = []
    sections.append(_card("1. Member folder — pipeline files", f"""
<div class="note">
All pipeline logic lives in self-contained <code>theme_*.py</code> files. No shared code dependency.
Run with: <code>python run_all_themes.py</code>
</div>
{member_table}
"""))
    sections.append(_card("2. Shared data file checks", f"""
{_fig_file_status(checks)}
{file_table}
"""))
    sections.append(_card("3. Train shard breakdown", f"""
{_fig_shard_sizes(shard_paths)}
<ul style="color:#ccc;margin:10px 0 0 16px">
  <li>Shards: <strong>{len(shard_paths)}</strong></li>
  <li>Total size: <strong>{sum(p.stat().st_size for p in shard_paths)/1024/1024:.2f} MB</strong></li>
</ul>
"""))
    sections.append(_card("4. Known shared repository issues → how we fixed them", f"""
<div class="note">Issues found in the shared repo starter code. Each is addressed in the corresponding theme file.</div>
{issue_table}
"""))

    html = _page("Theme 01 — Repository Audit",
                 "File existence · broken paths · known issues and fixes",
                 kpis, sections)
    out_path = OUT / "theme_01_repository_audit.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[theme_01] Wrote {out_path}")
    return state


if __name__ == "__main__":
    run({})
    print("[theme_01] Done.")
