#!/usr/bin/env python3
"""
Theme 02 — Engine Choice Justification
========================================
Shows:
  • DuckDB vs PySpark vs MapReduce — scored comparison
  • Why DuckDB is right for this pipeline
  • When you WOULD switch to PySpark or MapReduce
  • Self-contained HTML with radar chart + comparison table

Reads  : nothing (static analysis)
Writes : outputs/theme_02_engine_choice.html
"""
from __future__ import annotations

import base64
import pickle
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT   = Path(__file__).resolve().parents[2]
MEMBER = Path(__file__).resolve().parent
OUT    = MEMBER / "outputs_restart"
OUT.mkdir(exist_ok=True)

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
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start}}
.kpi-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:24px}}
.kpi{{background:{CRD};border:1px solid #2a2a2a;border-radius:10px;padding:16px;text-align:center}}
.kpi .val{{font-size:2rem;font-weight:700;color:{Y}}}
.kpi .lbl{{color:{MUT};font-size:.8rem;margin-top:4px}}
table{{width:100%;border-collapse:collapse;font-size:.82rem}}
th{{background:{Y};color:#111;padding:8px 10px;text-align:left}}
td{{padding:7px 10px;border-bottom:1px solid #222;vertical-align:top}}
tr:hover td{{background:#1e1e1e}}
img{{max-width:100%;border-radius:8px;margin-top:8px}}
.note{{background:#1a1a2e;border-left:4px solid {Y};padding:10px 14px;
       border-radius:4px;font-size:.85rem;margin:10px 0;color:#ccc}}
.green{{color:{GRN};font-weight:600}} .red{{color:{RED};font-weight:600}}
.score-bar{{display:inline-block;height:12px;border-radius:6px;vertical-align:middle;margin-left:6px}}
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


# Scores 1-5 for the radar
CRITERIA = ["JSON/many-to-many\nhandling", "Setup\noverhead (inv)",
            "Local\nreproducibility", "Interactive\nanalysis",
            "SQL / dataframe\nflexibility", "Single-machine\nsuitability"]

SCORES = {
    "DuckDB":    [5, 5, 5, 5, 5, 5],
    "PySpark":   [3, 2, 3, 3, 4, 2],
    "MapReduce": [1, 1, 1, 1, 2, 1],
}
COLORS = {"DuckDB": Y, "PySpark": B, "MapReduce": RED}

DETAIL_TABLE = pd.DataFrame([
    {"Criterion": "JSON / many-to-many",
     "DuckDB": "✅ Native json_each() in one SQL stmt",
     "PySpark": "⚠️ from_json + explode — verbose",
     "MapReduce": "❌ Manual mapper/reducer logic required"},
    {"Criterion": "Setup overhead",
     "DuckDB": "✅ Zero — embedded in-process library",
     "PySpark": "❌ JVM + cluster config + port mapping",
     "MapReduce": "❌ Hadoop/YARN/HDFS cluster required"},
    {"Criterion": "Local reproducibility",
     "DuckDB": "✅ Same SQL → same result every time",
     "PySpark": "⚠️ Local[*] mode works but non-trivial",
     "MapReduce": "❌ Not designed for single-machine use"},
    {"Criterion": "Interactive analysis",
     "DuckDB": "✅ Instant; notebook/REPL friendly",
     "PySpark": "⚠️ DAG planning adds latency for small frames",
     "MapReduce": "❌ Batch-only; no interactive mode"},
    {"Criterion": "OOF / iterative ML",
     "DuckDB": "✅ Pandas/sklearn after DuckDB ingest",
     "PySpark": "⚠️ Requires collect() calls → defeats purpose",
     "MapReduce": "❌ Cannot express cross-validation loops"},
    {"Criterion": "Scalability",
     "DuckDB": "⚠️ Single machine; fails > RAM",
     "PySpark": "✅ Horizontal scaling to 100s GB on cluster",
     "MapReduce": "✅ Petabyte-scale batch aggregation"},
    {"Criterion": "THIS pipeline — selected?",
     "DuckDB": "✅ YES",
     "PySpark": "❌ NO",
     "MapReduce": "❌ NO"},
])


def _fig_radar() -> str:
    N = len(CRITERIA)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True}, facecolor=BG)
    ax.set_facecolor(BG)
    for engine, vals in SCORES.items():
        v = vals + vals[:1]
        ax.plot(angles, v, color=COLORS[engine], linewidth=2.5, label=engine)
        ax.fill(angles, v, color=COLORS[engine], alpha=0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(CRITERIA, color=TXT, fontsize=9)
    ax.set_yticklabels([]); ax.set_ylim(0, 5)
    ax.grid(color=MUT, alpha=0.25)
    ax.spines["polar"].set_color(MUT)
    ax.set_title("Engine suitability (1=poor, 5=excellent)", color=TXT, pad=20, fontsize=13)
    leg = ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15),
                    facecolor=CRD, edgecolor=MUT)
    for t in leg.get_texts(): t.set_color(TXT)
    fig.tight_layout()
    return _img(fig)


def _fig_score_bars() -> str:
    totals = {eng: sum(vals) for eng, vals in SCORES.items()}
    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=BG)
    ax.set_facecolor(CRD)
    bars = ax.barh(list(totals.keys()), list(totals.values()),
                   color=[COLORS[e] for e in totals])
    ax.set_xlim(0, 30)
    ax.set_xlabel("Total score (max 30)", color=TXT)
    ax.set_title("Overall engine score for this workload", color=TXT, fontsize=12)
    ax.tick_params(colors=TXT)
    for spine in ax.spines.values(): spine.set_color(MUT)
    ax.grid(axis="x", alpha=0.15, color=MUT)
    for bar, (eng, v) in zip(bars, totals.items()):
        ax.text(v + 0.3, bar.get_y() + bar.get_height()/2,
                f"{v}/30", va="center", color=TXT, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _img(fig)


def _fig_when_to_switch() -> str:
    """Timeline / threshold chart: when would you switch from DuckDB to PySpark?"""
    sizes = [0.01, 0.1, 1, 10, 100, 1000]  # GB
    duckdb_ok  = [1 if s <= 32 else 0 for s in sizes]
    pyspark_ok = [1 if s >= 1  else 0 for s in sizes]
    x = np.arange(len(sizes))
    labels = ["10 MB", "100 MB", "1 GB", "10 GB", "100 GB", "1 TB"]

    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor=BG)
    ax.set_facecolor(CRD)
    ax.bar(x - 0.2, duckdb_ok,  width=0.4, color=Y,  label="DuckDB viable")
    ax.bar(x + 0.2, pyspark_ok, width=0.4, color=B,  label="PySpark advantageous")
    ax.set_xticks(x); ax.set_xticklabels(labels, color=TXT)
    ax.set_yticks([0,1]); ax.set_yticklabels(["No","Yes"], color=TXT)
    ax.set_title("When to switch engines (data size on one machine)", color=TXT, fontsize=12)
    ax.legend(facecolor=CRD, edgecolor=MUT, labelcolor=TXT)
    ax.grid(axis="y", alpha=0.15, color=MUT)
    for spine in ax.spines.values(): spine.set_color(MUT)
    # Mark current dataset
    ax.axvline(x=0.5, color=GRN, linestyle="--", linewidth=1.5)
    ax.text(0.55, 0.85, "← This dataset\n  (~10K rows)", color=GRN, fontsize=8,
            transform=ax.get_xaxis_transform())
    fig.tight_layout()
    return _img(fig)


def run(state: dict) -> dict:
    totals = {eng: sum(vals) for eng, vals in SCORES.items()}

    kpis = (
        _kpi("DuckDB",                   "Selected engine")
      + _kpi(f"{totals['DuckDB']}/30",   "DuckDB suitability<br>score")
      + _kpi("0",                         "Setup steps<br>required")
    )

    def _tbl(df):
        def _cell(v):
            sv = str(v)
            color = GRN if "✅" in sv else RED if "❌" in sv else TXT
            return f"<td style='color:{color}'>{v}</td>"
        rows = "".join(
            "<tr>" + "".join(_cell(v) for v in r) + "</tr>"
            for _, r in df.iterrows()
        )
        hdr = "".join(f"<th>{c}</th>" for c in df.columns)
        return f"<table><tr>{hdr}</tr>{rows}</table>"

    sections = []

    sections.append(_card("1. Suitability radar + overall scores", f"""
<div class="grid2">
  <div>{_fig_radar()}</div>
  <div>{_fig_score_bars()}</div>
</div>
"""))

    sections.append(_card("2. Detailed criterion-by-criterion comparison", f"""
{_tbl(DETAIL_TABLE)}
"""))

    sections.append(_card("3. When would you switch to PySpark?", f"""
{_fig_when_to_switch()}
<div class="note">
<strong>Switch to PySpark when:</strong>
<ul style="margin:6px 0 0 16px">
  <li>Training data exceeds single-machine RAM (typically > 32–64 GB)</li>
  <li>Pipeline runs as part of a distributed data-lake workflow on a cluster</li>
  <li>Feature generation requires joins across multiple very large tables that don't fit in memory</li>
</ul>
<strong>Never use MapReduce here:</strong> the pipeline is iterative (OOF cross-validation, SHAP,
permutation importance), which is fundamentally incompatible with MapReduce's rigid batch model.
</div>
"""))

    sections.append(_card("4. DuckDB SQL used for JSON many-to-many", f"""
<p>This is the exact query that reconstructs (tconst, director_id) pairs from directing.json:</p>
<div style="background:#111;border:1px solid #333;border-radius:6px;padding:12px 16px;
font-family:monospace;font-size:.8rem;color:{Y};white-space:pre;overflow-x:auto;margin:8px 0">
WITH src AS (
  SELECT json(movie) AS movie_obj, json(director) AS director_obj
  FROM read_json_auto('directing.json')
),
movies AS (
  SELECT je.key AS k,
         trim(both '&quot;' from CAST(je.value AS VARCHAR)) AS tconst
  FROM src, json_each(movie_obj) je
),
directors AS (
  SELECT je.key AS k,
         trim(both '&quot;' from CAST(je.value AS VARCHAR)) AS director_id
  FROM src, json_each(director_obj) je
)
SELECT m.tconst, d.director_id
FROM movies m JOIN directors d USING (k)
</div>
<div class="note">
The <code>json_each()</code> function explodes both parallel dicts and the JOIN on shared key
correctly aligns movie↔director. No Python dict iteration, no row ordering assumptions.
</div>
"""))

    html = _page(
        "Theme 02 — Engine Choice Justification",
        "DuckDB vs PySpark vs MapReduce · scored comparison · when to switch",
        kpis, sections,
    )
    (OUT / "theme_02_engine_choice.html").write_text(html, encoding="utf-8")
    print(f"[theme_02] Wrote {OUT}/theme_02_engine_choice.html")
    return state


if __name__ == "__main__":
    run({})
    print("[theme_02] Done.")
