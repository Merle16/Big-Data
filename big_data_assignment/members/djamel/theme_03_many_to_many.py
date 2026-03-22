#!/usr/bin/env python3
"""
Theme 03 — Many-to-Many Rebuild (DuckDB)
=========================================
Shows:
  • Why DuckDB is used (vs PySpark / MapReduce)
  • How directing.json and writing.json are parsed
  • JSON schema explained with examples
  • Edge counts and consistency check vs pre-converted CSVs
  • Self-contained HTML with embedded figures

Reads  : data/raw/json/directing.json, writing.json
         data/raw/csv/movie_directors.csv, movie_writers.csv
Writes : state.pkl (directors_raw, writers_raw)
         outputs/theme_03_many_to_many.html
"""
from __future__ import annotations

import base64
import json
import pickle
from io import BytesIO
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import duckdb

ROOT   = Path(__file__).resolve().parents[2]
MEMBER = Path(__file__).resolve().parent
OUT    = MEMBER / "outputs_restart"
OUT.mkdir(exist_ok=True)
STATE  = OUT / "state.pkl"

Y=  "#F5C518"; B="#1848f5"; BG="#0a0a0a"; CRD="#141414"
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
.code{{background:#111;border:1px solid #333;border-radius:6px;padding:10px 14px;
       font-family:monospace;font-size:.78rem;color:{Y};overflow-x:auto;margin:8px 0;white-space:pre}}
.tag{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.75rem;font-weight:600}}
.green{{background:{GRN};color:#111}}.red{{background:{RED};color:#fff}}
.blue{{background:{B};color:#fff}}.orange{{background:{ORG};color:#111}}
"""

def _b64(fig):
    buf=BytesIO(); fig.savefig(buf,format="png",dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(fig); return base64.b64encode(buf.getvalue()).decode()
def _img(fig): return f'<img src="data:image/png;base64,{_b64(fig)}">'
def _card(title,body,color=Y): return f'<div class="card"><h2 style="color:{color}">{title}</h2>{body}</div>'
def _kpi(val,lbl): return f'<div class="kpi"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>'
def _df_html(df):
    rows="".join("<tr>"+"".join(f"<td>{v}</td>" for v in r)+"</tr>" for r in df.values)
    hdr="".join(f"<th>{c}</th>" for c in df.columns)
    return f"<table><tr>{hdr}</tr>{rows}</table>"
def _page(title,subtitle,kpis,sections):
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{title}</title><style>{CSS}</style></head><body>
<h1>{title}</h1><p class="subtitle">{subtitle}</p>
<div class="kpi-grid">{kpis}</div>{"".join(sections)}</body></html>"""


def _build_edges_duckdb(json_dir: Path):
    directing_json = json_dir / "directing.json"
    writing_json   = json_dir / "writing.json"
    con = duckdb.connect()
    con.execute("""
        CREATE OR REPLACE TEMP TABLE directing_edges_raw AS
        WITH src AS (
          SELECT json(movie) AS movie_obj, json(director) AS director_obj
          FROM read_json_auto(?)
        ),
        movies AS (
          SELECT je.key AS k, trim(both '"' from CAST(je.value AS VARCHAR)) AS tconst
          FROM src, json_each(movie_obj) je
        ),
        directors AS (
          SELECT je.key AS k, trim(both '"' from CAST(je.value AS VARCHAR)) AS director_id
          FROM src, json_each(director_obj) je
        )
        SELECT m.tconst, d.director_id FROM movies m JOIN directors d USING (k)
    """, [str(directing_json)])
    con.execute("""
        CREATE OR REPLACE TEMP TABLE writing_edges_raw AS
        SELECT CAST(movie AS VARCHAR) AS tconst, CAST(writer AS VARCHAR) AS writer_id
        FROM read_json_auto(?)
    """, [str(writing_json)])
    directors = con.execute("SELECT tconst, director_id FROM directing_edges_raw").fetchdf()
    writers   = con.execute("SELECT tconst, writer_id FROM writing_edges_raw").fetchdf()
    con.close()
    return directors, writers


def _consistency_check(csv_dir, json_dir):
    with open(json_dir/"directing.json", encoding="utf-8") as f:
        dj = json.load(f)
    json_dir_pairs = {(str(t), str(d))
                      for t, d in zip(dj["movie"].values(), dj["director"].values())}
    dir_csv = pd.read_csv(csv_dir/"movie_directors.csv")
    csv_dir_pairs = {(str(r["tconst"]), str(r["director"]))
                     for _, r in dir_csv.iterrows()}
    return {
        "json_pairs": len(json_dir_pairs),
        "csv_pairs":  len(csv_dir_pairs),
        "in_json_not_csv": len(json_dir_pairs - csv_dir_pairs),
        "in_csv_not_json": len(csv_dir_pairs - json_dir_pairs),
    }


def _fig_engine_comparison() -> str:
    engines = ["DuckDB", "PySpark", "MapReduce"]
    categories = ["JSON handling", "Setup cost\n(inverted)", "Interactivity",
                  "SQL flexibility", "Single machine\nfit"]
    scores = {
        "DuckDB":    [5, 5, 5, 5, 5],
        "PySpark":   [3, 2, 3, 4, 2],
        "MapReduce": [1, 1, 1, 2, 1],
    }
    colors = {Y: "DuckDB", B: "PySpark", RED: "MapReduce"}
    angle = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angle += angle[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True}, facecolor=BG)
    ax.set_facecolor(BG)
    clrs = [Y, B, RED]
    for (eng, vals), clr in zip(scores.items(), clrs):
        vals_plot = vals + vals[:1]
        ax.plot(angle, vals_plot, color=clr, linewidth=2, label=eng)
        ax.fill(angle, vals_plot, color=clr, alpha=0.15)
    ax.set_xticks(angle[:-1])
    ax.set_xticklabels(categories, color=TXT, fontsize=9)
    ax.set_yticklabels([]); ax.set_ylim(0, 5)
    ax.tick_params(colors=TXT)
    ax.grid(color=MUT, alpha=0.3)
    ax.spines["polar"].set_color(MUT)
    ax.set_title("Engine suitability for this workload", color=TXT, pad=20, fontsize=12)
    leg = ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
                    facecolor=CRD, edgecolor=MUT)
    for t in leg.get_texts(): t.set_color(TXT)
    fig.tight_layout()
    return _img(fig)


def _fig_edge_counts(dir_df, wr_df) -> str:
    # movies per director count distribution
    dir_per_movie = dir_df.groupby("tconst")["director_id"].nunique()
    wr_per_movie  = wr_df.groupby("tconst")["writer_id"].nunique()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG)
    for ax, data, label, color in [
        (axes[0], dir_per_movie, "Directors per movie", Y),
        (axes[1], wr_per_movie,  "Writers per movie",  B),
    ]:
        ax.set_facecolor(CRD)
        vc = data.value_counts().sort_index().head(12)
        ax.bar(vc.index.astype(str), vc.values, color=color)
        ax.set_title(label, color=TXT, fontsize=11)
        ax.set_xlabel("Count", color=TXT); ax.set_ylabel("Movies", color=TXT)
        ax.tick_params(colors=TXT)
        for spine in ax.spines.values(): spine.set_color(MUT)
        ax.grid(axis="y", alpha=0.15, color=MUT)
    fig.patch.set_facecolor(BG)
    fig.tight_layout()
    return _img(fig)


def _fig_consistency(chk: dict) -> str:
    labels = ["Pairs in JSON\n(ground truth)", "Pairs in CSV",
              "In JSON\nnot in CSV", "In CSV\nnot in JSON"]
    vals   = [chk["json_pairs"], chk["csv_pairs"],
              chk["in_json_not_csv"], chk["in_csv_not_json"]]
    cols   = [Y, B, RED, RED]
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
    ax.set_facecolor(CRD)
    bars = ax.bar(labels, vals, color=cols)
    ax.set_title("JSON vs CSV directing edge consistency", color=TXT, fontsize=12)
    ax.tick_params(colors=TXT)
    ax.set_ylabel("Pairs", color=TXT)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+100,
                f"{v:,}", ha="center", color=TXT, fontsize=9)
    for spine in ax.spines.values(): spine.set_color(MUT)
    ax.grid(axis="y", alpha=0.15, color=MUT)
    fig.tight_layout()
    return _img(fig)


ENGINE_TABLE = pd.DataFrame([
    {"Engine":"DuckDB",    "JSON handling":"Native json_each()",  "Setup":"Zero (embedded)",
     "Local repro":"Perfect","Interactive":"Yes","Scales to":"Single machine (~64 GB RAM)",
     "Selected":"✅ YES"},
    {"Engine":"PySpark",   "JSON handling":"from_json / explode", "Setup":"High (JVM + cluster)",
     "Local repro":"Moderate","Interactive":"Slow","Scales to":"Hundreds of GB / cluster",
     "Selected":"❌ NO"},
    {"Engine":"MapReduce", "JSON handling":"Manual mapper/reducer","Setup":"Very high (Hadoop/YARN)",
     "Local repro":"Poor","Interactive":"None","Scales to":"Petabyte batch jobs",
     "Selected":"❌ NO"},
])


def run(state: dict) -> dict:
    json_dir = ROOT / "data" / "raw" / "json"
    csv_dir  = ROOT / "data" / "raw" / "csv"

    directors_raw, writers_raw = _build_edges_duckdb(json_dir)
    chk = _consistency_check(csv_dir, json_dir)

    # sample JSON schema for display
    with open(json_dir/"directing.json", encoding="utf-8") as f:
        dj = json.load(f)
    sample_keys = list(dj["movie"].keys())[:3]
    schema_example = {
        "movie":    {k: dj["movie"][k]    for k in sample_keys},
        "director": {k: dj["director"][k] for k in sample_keys},
    }

    kpis = (
        _kpi(f"{len(directors_raw):,}", "Director edges<br>rebuilt from JSON")
      + _kpi(f"{len(writers_raw):,}",   "Writer edges<br>rebuilt from JSON")
      + _kpi(chk["in_json_not_csv"], "Pairs in JSON<br>missing from CSV")
      + _kpi(chk["in_csv_not_json"], "Extra pairs in CSV<br>not in JSON")
    )

    dir_movies = directors_raw["tconst"].nunique()
    wr_movies  = writers_raw["tconst"].nunique()

    sections = []

    sections.append(_card("1. Why DuckDB?", f"""
{_fig_engine_comparison()}
{_df_html(ENGINE_TABLE)}
<div class="note">
<strong>Decision:</strong> DuckDB is chosen because this pipeline runs on a single machine with
~10 k movie rows and two JSON edge files. DuckDB parses <code>json_each()</code> natively in
a single SQL statement with zero setup overhead and full determinism. PySpark would add JVM +
cluster overhead with no benefit; MapReduce cannot express the iterative analytics this pipeline needs.
</div>
"""))

    sections.append(_card("2. JSON schema explained", f"""
<h3>directing.json structure</h3>
<p>Two parallel dicts with the same integer keys — <code>movie</code> maps key→tconst,
<code>director</code> maps the same key→nconst.</p>
<div class="code">{json.dumps(schema_example, indent=2)}</div>
<div class="note">We JOIN on the shared key to rebuild (tconst, director_id) pairs.
The pre-converted CSV loses this alignment information if rows are reordered.</div>

<h3>DuckDB SQL used</h3>
<div class="code">WITH src AS (
  SELECT json(movie) AS movie_obj, json(director) AS director_obj
  FROM read_json_auto('directing.json')
),
movies AS (
  SELECT je.key AS k,
         trim(both '\"' from CAST(je.value AS VARCHAR)) AS tconst
  FROM src, json_each(movie_obj) je
),
directors AS (
  SELECT je.key AS k,
         trim(both '\"' from CAST(je.value AS VARCHAR)) AS director_id
  FROM src, json_each(director_obj) je
)
SELECT m.tconst, d.director_id
FROM movies m JOIN directors d USING (k)</div>
"""))

    sections.append(_card("3. Edge counts & distribution", f"""
{_fig_edge_counts(directors_raw, writers_raw)}
<div class="grid2">
<div>
<h3>Directors</h3>
<ul style="color:#ccc;margin-left:16px">
  <li>Total edge rows: <strong>{len(directors_raw):,}</strong></li>
  <li>Unique movies with directors: <strong>{dir_movies:,}</strong></li>
  <li>Unique directors: <strong>{directors_raw['director_id'].nunique():,}</strong></li>
  <li>Avg directors/movie: <strong>{len(directors_raw)/max(dir_movies,1):.2f}</strong></li>
</ul>
</div>
<div>
<h3>Writers</h3>
<ul style="color:#ccc;margin-left:16px">
  <li>Total edge rows: <strong>{len(writers_raw):,}</strong></li>
  <li>Unique movies with writers: <strong>{wr_movies:,}</strong></li>
  <li>Unique writers: <strong>{writers_raw['writer_id'].nunique():,}</strong></li>
  <li>Avg writers/movie: <strong>{len(writers_raw)/max(wr_movies,1):.2f}</strong></li>
</ul>
</div>
</div>
"""))

    sections.append(_card("4. JSON vs pre-converted CSV consistency", f"""
{_fig_consistency(chk)}
{_df_html(pd.DataFrame([chk]))}
<div class="note">
Any non-zero value in <em>In JSON not in CSV</em> means edges were lost during CSV conversion.
We use JSON as the single source of truth to avoid this.
</div>
"""))

    html = _page(
        "Theme 03 — Many-to-Many Rebuild (DuckDB)",
        "JSON edge reconstruction · engine justification · consistency checks",
        kpis, sections,
    )
    (OUT / "theme_03_many_to_many.html").write_text(html, encoding="utf-8")
    print(f"[theme_03] Wrote {OUT}/theme_03_many_to_many.html")

    directors_raw.to_csv(OUT / "directors_raw.csv", index=False)
    writers_raw.to_csv(OUT / "writers_raw.csv", index=False)

    state["directors_raw"] = directors_raw
    state["writers_raw"]   = writers_raw
    return state


if __name__ == "__main__":
    st = run({})
    with open(STATE, "wb") as f:
        pickle.dump({}, f)   # DataFrames saved as CSV
    print("[theme_03] Done.")
