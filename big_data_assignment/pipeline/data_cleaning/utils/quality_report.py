"""
Schema-agnostic data quality audit.

Accepts any CSV or Parquet files as arguments. Auto-detects column types,
keys, missingness, normalization issues, outliers, and cross-table linkage.
Detects only -- never fixes.

Disguised-missing: hardcoded tokens = confirmed missing; entropy yields
CANDIDATE tokens for data-specific heuristics (review/filter per column).

Includes MAD, IQR, and k-trimmed-mean outlier analysis.
Includes fingerprint keying for textual deduplication candidates.

Usage (CLI):
    python quality_report.py data/raw/csv/train-1.csv
    python quality_report.py data/raw/csv/*.csv data/raw/IMDB_external_csv/*.csv
    python quality_report.py data/raw/csv/train-1.csv --output /tmp/report.txt

    Report is printed to the terminal and written to quality_audit_report.txt
    in the script's folder (or --output path if given).

Usage (Python):
    from pipeline.data_cleaning.quality_report import run_quality_audit
    run_quality_audit(["data/raw/csv/train-1.csv"])
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import duckdb
import numpy as np

# ---------------------------------------------------------------------------
# Hardcoded disguised-missing tokens: treated as confirmed missing.
# Entropy-based detection yields CANDIDATE tokens for data-specific heuristics
# (review/filter per column; do not auto-treat as missing).
# ---------------------------------------------------------------------------
DISGUISED_TOKENS: Tuple[str, ...] = (
    "\\N", "\\\\N", "", "NA", "N/A", "null", "None", "nan", "NaN",
)

_STRIP_PUNCT = re.compile(r"[^\w\s]")
_COLLAPSE_WS = re.compile(r"\s+")
_ID_LIKE_RE = re.compile(r"^[a-zA-Z]{1,4}\d+$")

# config in % to be dataset-agnostic
TRIM_K_VALUES = (0.05, 0.10, 0.25)


# ── Helpers: entropy ──────────────────────────────────────────────────────────

def char_entropy(token: str) -> float:
    """Shannon entropy of characters in *token* (0 for empty/non-string)."""
    if not isinstance(token, str) or len(token) == 0:
        return 0.0
    counts = Counter(token)
    n = len(token)
    probs = np.array(list(counts.values()), dtype=float) / n
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def _looks_numeric(tok: str) -> bool:
    """True if *tok* looks like a plain number (int or float)."""
    try:
        float(tok)
        return True
    except (ValueError, TypeError):
        return False


def detect_disguised_tokens_entropy(
    values: Any,
    min_freq_ratio: float = 0.01,
    min_count: int = 10,
    max_entropy: float = 1.5,
    max_len: int = 10,
) -> Set[str]:
    """Return candidate disguised-missing tokens (low-entropy, frequent, short).
    These are CANDIDATES for data-specific heuristics: review per column and
    filter (e.g. exclude valid categories like 'Horror') before treating as missing."""
    import pandas as pd
    s = pd.Series(values).dropna().astype(str).str.strip()
    total = len(s)
    if total == 0:
        return set()
    freq = s.value_counts()
    ratio = freq / total
    out: Set[str] = set()
    for tok, cnt in freq.items():
        if cnt < min_count or ratio[tok] < min_freq_ratio:
            continue
        if _looks_numeric(tok):
            continue
        if char_entropy(tok) <= max_entropy and len(tok) <= max_len:
            out.add(tok)
    return out


# ── Helpers: normalization & fingerprint ──────────────────────────────────────

def _normalize_text(s: str) -> str:
    """NFKD + ASCII + strip punctuation + collapse whitespace.
    Inspired by members/ilesh/dq.py normalize_title()."""
    if not s or s != s:
        return s
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = _STRIP_PUNCT.sub(" ", s)
    s = _COLLAPSE_WS.sub(" ", s).strip()
    return s


def _fingerprint(s: str) -> str:
    """Fingerprint keying: strip, lower, remove punct, NFKD→ASCII, tokenise,
    sort and deduplicate tokens."""
    if not s or s != s:
        return ""
    s = s.strip().lower()
    s = re.sub(r"[\p{P}\p{C}]" if False else r"[^\w\s]", " ", s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    tokens = sorted(set(s.split()))
    return " ".join(tokens)


# ── Helpers: SQL / misc ───────────────────────────────────────────────────────

def _sql_token_list(tokens: Set[str]) -> str:
    escaped = [str(t).replace("'", "''") for t in tokens if t is not None]
    return ", ".join(f"'{t}'" for t in escaped) if escaped else "'__NONE__'"


def _resolve_project_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    if (root / "config" / "config.yaml").exists():
        return root
    for p in Path(__file__).resolve().parents:
        if (p / "config" / "config.yaml").exists():
            return p
    return root


def _safe_table_name(stem: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]", "_", stem)
    if name[0:1].isdigit():
        name = "t_" + name
    return name


# ── Report accumulator ────────────────────────────────────────────────────────

@dataclass
class _Report:
    lines: List[str] = field(default_factory=list)
    structured: List[Dict[str, Any]] = field(default_factory=list)

    def section(self, title: str) -> None:
        self.lines.append("")
        self.lines.append("=" * 72)
        self.lines.append(title)
        self.lines.append("=" * 72)

    def subsection(self, title: str) -> None:
        self.lines.append("")
        self.lines.append(f"--- {title} ---")

    def line(self, msg: str = "") -> None:
        self.lines.append(msg)

    def record(self, table: str, column: str, check: str, **kw: Any) -> None:
        self.structured.append({"table": table, "column": column, "check": check, **kw})

    def text(self) -> str:
        return "\n".join(self.lines)


# ── Column classification ─────────────────────────────────────────────────────

def _classify_columns(
    con: duckdb.DuckDBPyConnection,
    tbl: str,
    cols: List[str],
    n_rows: int,
) -> Dict[str, str]:
    """Classify each column as 'numeric', 'id', or 'text'."""
    result: Dict[str, str] = {}
    for col in cols:
        if n_rows == 0:
            result[col] = "text"
            continue

        non_null = con.execute(
            f"SELECT COUNT(*) FROM {tbl} WHERE {col} IS NOT NULL AND TRIM(CAST({col} AS VARCHAR)) != ''"
        ).fetchone()[0]
        if non_null == 0:
            result[col] = "text"
            continue

        n_numeric = con.execute(f"""
            SELECT COUNT(*) FROM {tbl}
            WHERE TRY_CAST(TRIM(CAST({col} AS VARCHAR)) AS DOUBLE) IS NOT NULL
              AND {col} IS NOT NULL AND TRIM(CAST({col} AS VARCHAR)) != ''
        """).fetchone()[0]

        if non_null > 0 and n_numeric / non_null > 0.5:
            n_distinct = con.execute(
                f"SELECT COUNT(DISTINCT TRIM(CAST({col} AS VARCHAR))) FROM {tbl} WHERE {col} IS NOT NULL"
            ).fetchone()[0]
            uniqueness = n_distinct / non_null if non_null > 0 else 0
            if uniqueness > 0.9:
                sample = con.execute(f"""
                    SELECT TRIM(CAST({col} AS VARCHAR)) AS v FROM {tbl}
                    WHERE {col} IS NOT NULL AND TRIM(CAST({col} AS VARCHAR)) != ''
                    LIMIT 100
                """).fetchdf()
                id_matches = sum(1 for v in sample["v"] if _ID_LIKE_RE.match(str(v)))
                if id_matches / len(sample) > 0.8:
                    result[col] = "id"
                    continue
            result[col] = "numeric"
            continue

        n_distinct = con.execute(
            f"SELECT COUNT(DISTINCT TRIM(CAST({col} AS VARCHAR))) FROM {tbl} WHERE {col} IS NOT NULL"
        ).fetchone()[0]
        uniqueness = n_distinct / non_null if non_null > 0 else 0
        if uniqueness > 0.9:
            sample = con.execute(f"""
                SELECT TRIM(CAST({col} AS VARCHAR)) AS v FROM {tbl}
                WHERE {col} IS NOT NULL AND TRIM(CAST({col} AS VARCHAR)) != ''
                LIMIT 100
            """).fetchdf()
            id_matches = sum(1 for v in sample["v"] if _ID_LIKE_RE.match(str(v)))
            if id_matches / len(sample) > 0.8:
                result[col] = "id"
                continue

        result[col] = "text"
    return result


# ── Ingestion ─────────────────────────────────────────────────────────────────

def _default_file_paths() -> List[Path]:
    root = _resolve_project_root()
    paths: List[Path] = []
    csv_dir = root / "data" / "raw" / "csv"
    if csv_dir.exists():
        paths.extend(sorted(csv_dir.glob("*.csv")))
    ext_dir = root / "data" / "raw" / "IMDB_external_csv"
    if ext_dir.exists():
        paths.extend(sorted(ext_dir.glob("*.csv")))
    return paths


def _ingest_files(
    con: duckdb.DuckDBPyConnection,
    file_paths: List[Path],
) -> Tuple[Dict[str, str], Dict[str, Path]]:
    """Load each file as a DuckDB table.

    Returns:
        tables: {table_name: table_name}
        table_paths: {table_name: source_path}  — for dtype inference
    """
    tables: Dict[str, str] = {}
    table_paths: Dict[str, Path] = {}
    seen_names: Dict[str, int] = {}
    for p in file_paths:
        base = _safe_table_name(p.stem)
        if base in seen_names:
            seen_names[base] += 1
            tbl_name = f"{base}_{seen_names[base]}"
        else:
            seen_names[base] = 0
            tbl_name = base

        try:
            if p.suffix.lower() in (".parquet", ".pq"):
                con.execute(f"CREATE OR REPLACE TABLE {tbl_name} AS SELECT * FROM read_parquet('{p}')")
            else:
                con.execute(f"""
                    CREATE OR REPLACE TABLE {tbl_name} AS
                    SELECT * FROM read_csv_auto('{p}', header=True, all_varchar=True, ignore_errors=True)
                """)
            # Drop column0 if present (index column from CSV; not used in the report)
            try:
                desc = con.execute(f"DESCRIBE {tbl_name}").fetchdf()
                if "column0" in desc["column_name"].tolist():
                    con.execute(f"""
                        CREATE OR REPLACE TABLE {tbl_name} AS
                        SELECT * EXCLUDE (column0) FROM {tbl_name}
                    """)
            except Exception:
                pass
            tables[tbl_name] = tbl_name
            table_paths[tbl_name] = p
        except Exception as exc:
            print(f"[quality_report] WARNING: could not load {p}: {exc}")
    return tables, table_paths


def _infer_native_types(
    con: duckdb.DuckDBPyConnection,
    tbl_name: str,
    path: Path,
) -> Dict[str, str]:
    """Return DuckDB's natively-inferred SQL type for each column.

    Re-reads the source file with type inference enabled (no all_varchar) so we
    can compare against actual content and flag mismatches.
    """
    try:
        if path.suffix.lower() in (".parquet", ".pq"):
            desc = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}') LIMIT 0").fetchdf()
        else:
            desc = con.execute(
                f"DESCRIBE SELECT * FROM read_csv_auto('{path}', header=True, ignore_errors=True) LIMIT 0"
            ).fetchdf()
        return dict(zip(desc["column_name"], desc["column_type"]))
    except Exception:
        return {}


# ── Section 1: Schema Discovery ──────────────────────────────────────────────

def _section_schema(
    con: duckdb.DuckDBPyConnection,
    tables: Dict[str, str],
    table_paths: Dict[str, Path],
    rpt: _Report,
) -> Dict[str, Dict[str, str]]:
    """Returns {table: {col: classification}} for downstream use."""
    rpt.section("1. SCHEMA DISCOVERY")
    rpt.line(f"  {'table':<25} {'rows':>8}  {'cols':>4}  numeric  text  id  columns")
    rpt.line(f"  {'-'*25} {'-'*8}  {'-'*4}  {'-'*7}  {'-'*4}  {'-'*2}  {'-'*40}")
    all_classes: Dict[str, Dict[str, str]] = {}
    # For dtype conflict detection: {tbl: {col: native_duck_type}}
    native_types: Dict[str, Dict[str, str]] = {}

    for tbl_name in tables:
        desc = con.execute(f"DESCRIBE {tbl_name}").fetchdf()
        cols = desc["column_name"].tolist()
        n_rows = con.execute(f"SELECT COUNT(*) FROM {tbl_name}").fetchone()[0]

        classifications = _classify_columns(con, tbl_name, cols, n_rows)
        all_classes[tbl_name] = classifications

        # Infer native DuckDB types from source file
        if tbl_name in table_paths:
            native_types[tbl_name] = _infer_native_types(con, tbl_name, table_paths[tbl_name])
        else:
            native_types[tbl_name] = {}

        n_num = sum(1 for t in classifications.values() if t == "numeric")
        n_txt = sum(1 for t in classifications.values() if t == "text")
        n_id  = sum(1 for t in classifications.values() if t == "id")
        col_summary = ", ".join(
            f"{c}({classifications[c][0]})" for c in cols
        )
        rpt.line(f"  {tbl_name:<25} {n_rows:>8,}  {len(cols):>4}  {n_num:>7}  {n_txt:>4}  {n_id:>2}  {col_summary}")
        for col in cols:
            rpt.record(tbl_name, col, "schema", detected=classifications[col])
        rpt.record(tbl_name, "*", "row_count", value=n_rows)

    # ── Dtype conflict sub-section ──────────────────────────────────────────
    # Flag columns where DuckDB infers VARCHAR/TEXT but content is numeric, or
    # where DuckDB infers a numeric/bool type that doesn't match logical class.
    _NUMERIC_DUCK = {"INTEGER", "BIGINT", "HUGEINT", "SMALLINT", "TINYINT",
                     "FLOAT", "DOUBLE", "DECIMAL", "UBIGINT", "UINTEGER",
                     "USMALLINT", "UTINYINT", "INT4", "INT8", "INT2", "INT1"}
    _STRING_DUCK  = {"VARCHAR", "TEXT", "STRING", "BLOB", "CHAR"}

    rpt.line("")
    rpt.line("  Expected dtype vs inferred dtype  (conflicts only):")
    rpt.line(f"  {'table':<25} {'column':<22} {'inferred':>12}  {'content-class':<14}  issue")
    rpt.line(f"  {'-'*25} {'-'*22} {'-'*12}  {'-'*14}  {'-'*40}")
    any_conflict = False
    for tbl_name, col_types in native_types.items():
        for col, duck_type in col_types.items():
            if col not in all_classes[tbl_name]:
                continue
            logical = all_classes[tbl_name][col]
            base_duck = duck_type.split("(")[0].upper()  # strip precision, e.g. DECIMAL(10,2) → DECIMAL

            issue = ""
            if base_duck in _STRING_DUCK and logical == "numeric":
                issue = "stored VARCHAR but content is numeric → dirty values (e.g. \\N) prevent type inference"
            elif base_duck in _NUMERIC_DUCK and logical == "text":
                issue = f"DuckDB inferred {duck_type} but classified as text → may be misclassified (low cardinality numeric?)"
            elif base_duck == "BOOLEAN" and logical == "text":
                issue = f"DuckDB inferred BOOLEAN but classified as text → check encoding"

            if issue:
                any_conflict = True
                rpt.line(f"  {tbl_name:<25} {col:<22} {duck_type:>12}  {logical:<14}  {issue}")
                rpt.record(tbl_name, col, "dtype_conflict", inferred=duck_type, logical=logical, issue=issue)

    if not any_conflict:
        rpt.line("  (no dtype conflicts detected)")

    return all_classes


# ── Section 2: Key / UUID Detection ──────────────────────────────────────────

def _section_keys(
    con: duckdb.DuckDBPyConnection,
    tables: Dict[str, str],
    classes: Dict[str, Dict[str, str]],
    rpt: _Report,
) -> None:
    rpt.section("2. KEY / UUID DETECTION")
    rpt.line(f"  {'table':<25} {'column':<20} {'kind':<14} {'prefix':<8} {'unique%':>7}  candidate_key")
    rpt.line(f"  {'-'*25} {'-'*20} {'-'*14} {'-'*8} {'-'*7}  {'-'*13}")

    for tbl_name in tables:
        n_rows = con.execute(f"SELECT COUNT(*) FROM {tbl_name}").fetchone()[0]
        if n_rows == 0:
            continue
        cols = list(classes[tbl_name].keys())

        for col in cols:
            n_distinct = con.execute(
                f"SELECT COUNT(DISTINCT TRIM(CAST({col} AS VARCHAR))) FROM {tbl_name} WHERE {col} IS NOT NULL"
            ).fetchone()[0]
            n_non_null = con.execute(
                f"SELECT COUNT(*) FROM {tbl_name} WHERE {col} IS NOT NULL"
            ).fetchone()[0]
            uniqueness = n_distinct / n_non_null if n_non_null > 0 else 0

            if classes[tbl_name][col] == "id":
                sample_vals = con.execute(f"""
                    SELECT TRIM(CAST({col} AS VARCHAR)) AS v FROM {tbl_name}
                    WHERE {col} IS NOT NULL AND TRIM(CAST({col} AS VARCHAR)) != ''
                    LIMIT 5
                """).fetchdf()["v"].tolist()
                prefix = ""
                if sample_vals:
                    m = re.match(r"^([a-zA-Z]+)", str(sample_vals[0]))
                    if m:
                        prefix = m.group(1)
                is_ck = "YES" if uniqueness >= 0.999 else "no"
                rpt.line(f"  {tbl_name:<25} {col:<20} {'id-like':<14} {prefix:<8} {uniqueness:>6.1%}  {is_ck}")
                rpt.record(tbl_name, col, "key_detection",
                           kind="id", prefix=prefix, distinct=n_distinct,
                           non_null=n_non_null, uniqueness=round(uniqueness, 4),
                           candidate_key=is_ck)
            elif uniqueness >= 0.999 and n_non_null > 10:
                rpt.line(f"  {tbl_name:<25} {col:<20} {'candidate_key':<14} {'':<8} {uniqueness:>6.1%}  YES")
                rpt.record(tbl_name, col, "key_detection",
                           kind="candidate_key", distinct=n_distinct,
                           non_null=n_non_null, uniqueness=round(uniqueness, 4))


# ── Section 3: Key Dependencies ──────────────────────────────────────────────

def _section_key_deps(
    con: duckdb.DuckDBPyConnection,
    tables: Dict[str, str],
    classes: Dict[str, Dict[str, str]],
    rpt: _Report,
) -> None:
    rpt.section("3. KEY DEPENDENCIES (cross-table linkage)")

    tbl_names = list(tables.keys())
    if len(tbl_names) < 2:
        rpt.line("  (only one table loaded; skipping cross-table analysis)")
        return

    col_map: Dict[str, List[str]] = {}
    for tbl_name in tbl_names:
        for col in classes[tbl_name]:
            col_map.setdefault(col, []).append(tbl_name)

    shared_cols = {col: tbls for col, tbls in col_map.items() if len(tbls) > 1}
    if not shared_cols:
        rpt.line("  No column names shared across tables.")
        return

    for col, tbls in shared_cols.items():
        for i in range(len(tbls)):
            for j in range(i + 1, len(tbls)):
                a, b = tbls[i], tbls[j]
                try:
                    overlap = con.execute(f"""
                        SELECT COUNT(*) FROM (
                            SELECT DISTINCT TRIM(CAST({col} AS VARCHAR)) AS v
                            FROM {a} WHERE {col} IS NOT NULL
                            INTERSECT
                            SELECT DISTINCT TRIM(CAST({col} AS VARCHAR)) AS v
                            FROM {b} WHERE {col} IS NOT NULL
                        )
                    """).fetchone()[0]
                    da = con.execute(
                        f"SELECT COUNT(DISTINCT TRIM(CAST({col} AS VARCHAR))) FROM {a} WHERE {col} IS NOT NULL"
                    ).fetchone()[0]
                    db = con.execute(
                        f"SELECT COUNT(DISTINCT TRIM(CAST({col} AS VARCHAR))) FROM {b} WHERE {col} IS NOT NULL"
                    ).fetchone()[0]
                    denom = min(da, db) if min(da, db) > 0 else 1
                    pct = 100.0 * overlap / denom

                    if overlap > 0:
                        rpt.line(
                            f"  {col:<20} {a} <-> {b}:  "
                            f"{overlap:,} shared values  ({pct:.1f}% of smaller side)"
                        )
                        rpt.record(a, col, "key_dependency",
                                   linked_table=b, shared=overlap, pct=round(pct, 2))
                except Exception:
                    pass


# ── Section 3: Missingness ────────────────────────────────────────────────────

def _section_missingness(
    con: duckdb.DuckDBPyConnection,
    tables: Dict[str, str],
    classes: Dict[str, Dict[str, str]],
    rpt: _Report,
) -> None:
    rpt.section("3. MISSINGNESS")

    hardcoded_set = set(DISGUISED_TOKENS)

    # ── Step 1: NULL rate — only show columns with any NULLs ──────────────────
    rpt.line("  Step 1: NULL rate  (columns with >0 NULLs only)")
    rpt.line(f"  {'table':<25} {'column':<25} {'null_n':>8}  {'null%':>6}")
    any_null = False
    for tbl_name in tables:
        n_rows = con.execute(f"SELECT COUNT(*) FROM {tbl_name}").fetchone()[0]
        if n_rows == 0:
            continue
        cols = [r[0] for r in con.execute(f"DESCRIBE {tbl_name}").fetchall()]
        for col in cols:
            null_n = con.execute(f"SELECT COUNT(*) FROM {tbl_name} WHERE {col} IS NULL").fetchone()[0]
            if null_n > 0:
                any_null = True
                null_pct = 100.0 * null_n / n_rows
                rpt.line(f"  {tbl_name:<25} {col:<25} {null_n:>8,}  {null_pct:>5.1f}%")
            rpt.record(tbl_name, col, "null_count", value=null_n,
                       pct=round(100.0 * null_n / n_rows, 2) if n_rows else 0)
    if not any_null:
        rpt.line("  (no NULLs found)")

    # ── Step 2: Disguised missing — candidates first, then confirmed ─────────
    rpt.line("")
    rpt.line("  Step 2: Disguised missing  (exact encoding; CANDIDATE tokens before hardcoded confirmed)")

    # Collect appearances across all tables/columns
    confirmed_appearances: Dict[str, List[tuple]] = defaultdict(list)
    candidate_appearances: Dict[str, List[tuple]] = defaultdict(list)
    all_candidate_tokens: Set[str] = set()
    any_disguised = False

    for tbl_name in tables:
        n_rows = con.execute(f"SELECT COUNT(*) FROM {tbl_name}").fetchone()[0]
        if n_rows == 0:
            continue
        cols = [r[0] for r in con.execute(f"DESCRIBE {tbl_name}").fetchall()]
        for col in cols:
            try:
                df = con.execute(
                    f"SELECT TRIM(CAST({col} AS VARCHAR)) AS v FROM {tbl_name} WHERE {col} IS NOT NULL"
                ).fetchdf()
            except Exception:
                continue
            if df.empty:
                continue

            value_counts = df["v"].value_counts()
            # All entropy-detected tokens (including those in hardcoded list)
            candidate_tokens = detect_disguised_tokens_entropy(df["v"])

            hc_count = 0
            for tok in hardcoded_set:
                cnt = int(value_counts.get(tok, 0))
                if cnt > 0:
                    hc_count += cnt
                    confirmed_appearances[tok].append((tbl_name, col, cnt, 100.0 * cnt / n_rows))
            for tok in candidate_tokens:
                cnt = int(value_counts.get(tok, 0))
                if cnt > 0:
                    all_candidate_tokens.add(tok)
                    candidate_appearances[tok].append((tbl_name, col, cnt, 100.0 * cnt / n_rows))

            null_n = con.execute(f"SELECT COUNT(*) FROM {tbl_name} WHERE {col} IS NULL").fetchone()[0]
            total_confirmed = null_n + hc_count
            total_conf_pct = 100.0 * total_confirmed / n_rows
            if hc_count > 0 or candidate_tokens:
                any_disguised = True

            rpt.record(tbl_name, col, "disguised_confirmed", value=hc_count)
            rpt.record(tbl_name, col, "total_missing_confirmed", value=total_confirmed, pct=round(total_conf_pct, 2))

    # Candidates first (entropy-detected; includes those also in hardcoded list)
    rpt.line(f"  CANDIDATE TOKENS (entropy-detected; review per column) = {sorted(repr(t) for t in all_candidate_tokens)}")
    if any(t == "\\N" for t in all_candidate_tokens):
        rpt.line("    (Note: '\\N' = backslash-N, common disguised-missing encoding e.g. IMDB; shown in Python repr)")
    if candidate_appearances:
        rpt.line("  Candidate token appearances:")
        for tok in sorted(candidate_appearances.keys(), key=lambda t: (-sum(x[2] for x in candidate_appearances[t]), t)):
            rows = candidate_appearances[tok]
            n_cols = len(set((t, c) for t, c, _, _ in rows))
            also_confirmed = " [also in confirmed list]" if tok in hardcoded_set else ""
            parts = [f"{t}.{c} ({n:,}; {p:.1f}%)" for t, c, n, p in rows]
            rpt.line(f"    {repr(tok):20} in {n_cols} column(s){also_confirmed}: {', '.join(parts)}")
    else:
        rpt.line("  (no entropy-detected candidates found)")

    # Then confirmed hardcoded tokens
    confirmed_list = sorted(set(DISGUISED_TOKENS))
    rpt.line(f"  CONFIRMED TOKENS (hardcoded; treated as missing) = {[repr(t) for t in confirmed_list]}")
    # Explicit note so readers recognize the common encoding (repr shows '\\N' for backslash-N)
    if any(t == "\\N" for t in confirmed_list):
        rpt.line("    (Note: '\\N' in the list above = the two-character token backslash-N, common disguised-missing encoding e.g. IMDB)")
    if confirmed_appearances:
        rpt.line("  Confirmed token appearances:")
        for tok in sorted(confirmed_appearances.keys(), key=lambda t: (-sum(x[2] for x in confirmed_appearances[t]), t)):
            rows = confirmed_appearances[tok]
            n_cols = len(set((t, c) for t, c, _, _ in rows))
            parts = [f"{t}.{c} ({n:,}; {p:.1f}%)" for t, c, n, p in rows]
            rpt.line(f"    {repr(tok):20} in {n_cols} column(s): {', '.join(parts)}")

    if not any_disguised:
        rpt.line("  (no confirmed or candidate disguised tokens found)")

    # ── Step 3: Type-mismatch in numeric columns ──────────────────────────────
    rpt.line("")
    rpt.line("  Step 3: Type-mismatch in numeric cols  (non-numeric strings in a numeric column)")
    rpt.line(f"  {'table':<25} {'column':<25} {'n':>8}  {'%':>6}  examples")
    any_mismatch = False
    for tbl_name in tables:
        n_rows = con.execute(f"SELECT COUNT(*) FROM {tbl_name}").fetchone()[0]
        if n_rows == 0:
            continue
        for col, kind in classes[tbl_name].items():
            if kind != "numeric":
                continue
            n_mm = con.execute(f"""
                SELECT COUNT(*) FROM {tbl_name}
                WHERE {col} IS NOT NULL
                  AND TRY_CAST(TRIM(CAST({col} AS VARCHAR)) AS DOUBLE) IS NULL
            """).fetchone()[0]
            if n_mm > 0:
                any_mismatch = True
                mm_pct = 100.0 * n_mm / n_rows
                ex = con.execute(f"""
                    SELECT DISTINCT TRIM(CAST({col} AS VARCHAR)) AS v FROM {tbl_name}
                    WHERE {col} IS NOT NULL
                      AND TRY_CAST(TRIM(CAST({col} AS VARCHAR)) AS DOUBLE) IS NULL
                    LIMIT 5
                """).fetchdf()["v"].tolist()
                rpt.line(f"  {tbl_name:<25} {col:<25} {n_mm:>8,}  {mm_pct:>5.1f}%  {ex}")
                rpt.record(tbl_name, col, "type_mismatch", value=n_mm,
                           pct=round(mm_pct, 2), examples=str(ex))
    if not any_mismatch:
        rpt.line("  (no type mismatches found)")


# ── Section 4: Normalization Detection ────────────────────────────────────────

def _section_normalization(
    con: duckdb.DuckDBPyConnection,
    tables: Dict[str, str],
    classes: Dict[str, Dict[str, str]],
    rpt: _Report,
) -> None:
    rpt.section("4. NORMALIZATION DETECTION  (NFKD+ASCII; only columns with unnormalized values)")
    rpt.line(f"  {'table':<25} {'column':<25} {'unnorm':>8}  {'%':>6}  {'distinct':>8}  strlen")

    try:
        con.create_function("_nfkd_norm", _normalize_text, [str], str)
    except Exception:
        pass

    any_found = False
    for tbl_name in tables:
        for col, kind in classes[tbl_name].items():
            if kind != "text":
                continue
            try:
                non_null = con.execute(f"""
                    SELECT COUNT(*) FROM {tbl_name}
                    WHERE {col} IS NOT NULL AND TRIM(CAST({col} AS VARCHAR)) != ''
                """).fetchone()[0]
                if non_null == 0:
                    continue
                n_changed = con.execute(f"""
                    SELECT COUNT(*) FROM {tbl_name}
                    WHERE {col} IS NOT NULL
                      AND TRIM(CAST({col} AS VARCHAR)) != ''
                      AND _nfkd_norm(CAST({col} AS VARCHAR)) != CAST({col} AS VARCHAR)
                """).fetchone()[0]
                if n_changed == 0:
                    continue
                any_found = True
                pct = 100.0 * n_changed / non_null
                stats = con.execute(f"""
                    SELECT COUNT(DISTINCT CAST({col} AS VARCHAR)),
                           MIN(LENGTH(CAST({col} AS VARCHAR))),
                           MAX(LENGTH(CAST({col} AS VARCHAR)))
                    FROM {tbl_name}
                    WHERE {col} IS NOT NULL AND TRIM(CAST({col} AS VARCHAR)) != ''
                """).fetchone()
                n_distinct, min_len, max_len = stats
                rpt.line(f"  {tbl_name:<25} {col:<25} {n_changed:>8,}  {pct:>5.1f}%  {n_distinct:>8,}  [{min_len},{max_len}]")
                rpt.record(tbl_name, col, "normalization",
                           unnormalized=n_changed, pct=round(pct, 2),
                           distinct=n_distinct, min_len=min_len, max_len=max_len)
            except Exception:
                continue
    if not any_found:
        rpt.line("  (all text columns are already normalized)")


# ── Section 6: Robust Statistics ──────────────────────────────────────────────

def _val_subquery(tbl: str, col: str) -> str:
    return (f"SELECT TRY_CAST(TRIM(CAST({col} AS VARCHAR)) AS DOUBLE) AS v "
            f"FROM {tbl} "
            f"WHERE TRY_CAST(TRIM(CAST({col} AS VARCHAR)) AS DOUBLE) IS NOT NULL")


def _section_robust_stats(
    con: duckdb.DuckDBPyConnection,
    tables: Dict[str, str],
    classes: Dict[str, Dict[str, str]],
    rpt: _Report,
) -> None:
    rpt.section("5. ROBUST STATISTICS  (numeric columns; median/MAD resist corrupted values)")
    hdr = (f"  {'table':<20} {'col':<18} {'n':>7}  {'min':>9}  {'max':>9}  "
           f"{'mean':>9}  {'p50':>9}  {'MAD':>7}  "
           f"{'out_IQR%':>8}  {'out_MAD%':>8}  "
           f"{'k5%_Δ':>8}  {'k10%_Δ':>8}  {'k25%_Δ':>8}")
    rpt.line(hdr)

    for tbl_name in tables:
        num_cols = [c for c, t in classes[tbl_name].items() if t == "numeric"]
        for col in num_cols:
            vq = _val_subquery(tbl_name, col)
            stats = con.execute(f"""
                SELECT COUNT(v), MIN(v), MAX(v),
                       ROUND(AVG(v),2), ROUND(STDDEV_POP(v),2),
                       ROUND(QUANTILE_CONT(v,0.25),2),
                       ROUND(QUANTILE_CONT(v,0.50),2),
                       ROUND(QUANTILE_CONT(v,0.75),2)
                FROM ({vq})
            """).fetchone()
            cnt, vmin, vmax, vmean, vstd, p25, p50, p75 = stats
            if not cnt:
                continue

            iqr = p75 - p25
            iqr_lo, iqr_hi = p25 - 1.5 * iqr, p75 + 1.5 * iqr

            mad_val = con.execute(f"""
                SELECT ROUND(QUANTILE_CONT(ABS(v - med), 0.50), 4)
                FROM (SELECT v, QUANTILE_CONT(v, 0.50) OVER () AS med FROM ({vq}))
            """).fetchone()[0] or 0.0
            mad_lo, mad_hi = p50 - 3 * mad_val, p50 + 3 * mad_val

            n_iqr = con.execute(f"SELECT COUNT(*) FROM ({vq}) WHERE v < {iqr_lo} OR v > {iqr_hi}").fetchone()[0]
            n_mad = con.execute(f"SELECT COUNT(*) FROM ({vq}) WHERE v < {mad_lo} OR v > {mad_hi}").fetchone()[0]

            deltas = []
            for k in TRIM_K_VALUES:
                lo_q = con.execute(f"SELECT QUANTILE_CONT(v, {k}) FROM ({vq})").fetchone()[0]
                hi_q = con.execute(f"SELECT QUANTILE_CONT(v, {1-k}) FROM ({vq})").fetchone()[0]
                tr = con.execute(f"SELECT ROUND(AVG(v),2) FROM ({vq}) WHERE v >= {lo_q} AND v <= {hi_q}").fetchone()[0]
                deltas.append(round(vmean - tr, 2) if tr is not None else None)
                rpt.record(tbl_name, col, "trimmed_mean", k=k, trimmed_mean=tr, full_mean=vmean,
                           delta=deltas[-1])

            d5, d10, d25 = (f"{d:+.2f}" if d is not None else "n/a" for d in deltas)
            rpt.line(
                f"  {tbl_name:<20} {col:<18} {cnt:>7,}  {vmin:>9}  {vmax:>9}  "
                f"{vmean:>9}  {p50:>9}  {mad_val:>7}  "
                f"{100*n_iqr/cnt:>7.1f}%  {100*n_mad/cnt:>7.1f}%  "
                f"{d5:>8}  {d10:>8}  {d25:>8}"
            )
            rpt.record(tbl_name, col, "distribution",
                       count=cnt, min=vmin, max=vmax, mean=vmean, std=vstd,
                       p25=p25, p50=p50, p75=p75,
                       iqr=round(iqr, 4), iqr_outliers=n_iqr,
                       mad=mad_val, mad_outliers=n_mad)

            # Suspicious outlier values: most extreme distinct values outside MAD fence
            if n_mad > 0:
                outlier_vals = con.execute(f"""
                    SELECT v, COUNT(*) AS n
                    FROM ({vq})
                    WHERE v < {mad_lo} OR v > {mad_hi}
                    GROUP BY v ORDER BY ABS(v - {p50}) DESC LIMIT 5
                """).fetchall()
                if outlier_vals:
                    parts = [f"{v} (×{n:,})" for v, n in outlier_vals]
                    rpt.line(f"    suspicious: {', '.join(parts)}")


# ── Section 7: Fingerprint Keying ─────────────────────────────────────────────

def _section_fingerprint(
    con: duckdb.DuckDBPyConnection,
    tables: Dict[str, str],
    classes: Dict[str, Dict[str, str]],
    rpt: _Report,
) -> None:
    rpt.section("6. FINGERPRINT KEYING (textual deduplication candidates)")

    try:
        con.create_function("_fprint", _fingerprint, [str], str)
    except Exception:
        pass

    rpt.line(f"  {'table':<25} {'column':<25} {'groups':>6}  top example pairs")
    any_found = False
    for tbl_name in tables:
        for col, kind in classes[tbl_name].items():
            if kind != "text":
                continue
            try:
                n_non_null = con.execute(f"""
                    SELECT COUNT(*) FROM {tbl_name}
                    WHERE {col} IS NOT NULL AND TRIM(CAST({col} AS VARCHAR)) != ''
                """).fetchone()[0]
                if n_non_null < 10:
                    continue

                collisions = con.execute(f"""
                    SELECT fp, COUNT(DISTINCT raw_val) AS n_raw
                    FROM (
                        SELECT _fprint(CAST({col} AS VARCHAR)) AS fp,
                               CAST({col} AS VARCHAR) AS raw_val
                        FROM {tbl_name}
                        WHERE {col} IS NOT NULL AND TRIM(CAST({col} AS VARCHAR)) != ''
                    )
                    GROUP BY fp HAVING n_raw > 1
                    ORDER BY n_raw DESC
                    LIMIT 3
                """).fetchdf()
                if collisions.empty:
                    continue

                any_found = True
                n_groups = con.execute(f"""
                    SELECT COUNT(*) FROM (
                        SELECT _fprint(CAST({col} AS VARCHAR)) AS fp
                        FROM {tbl_name}
                        WHERE {col} IS NOT NULL AND TRIM(CAST({col} AS VARCHAR)) != ''
                        GROUP BY fp HAVING COUNT(DISTINCT CAST({col} AS VARCHAR)) > 1
                    )
                """).fetchone()[0]

                examples_str = ""
                for _, row in collisions.iterrows():
                    ex = con.execute(f"""
                        SELECT DISTINCT CAST({col} AS VARCHAR) AS v FROM {tbl_name}
                        WHERE _fprint(CAST({col} AS VARCHAR)) = '{str(row["fp"]).replace("'","''")}'
                        LIMIT 2
                    """).fetchdf()["v"].tolist()
                    examples_str += f"{ex}  "

                rpt.line(f"  {tbl_name:<25} {col:<25} {n_groups:>6}  {examples_str.strip()}")
                rpt.record(tbl_name, col, "fingerprint_collisions", value=n_groups)
            except Exception:
                continue

    if not any_found:
        rpt.line("  (no fingerprint collisions found)")


# ── Section 8: Duplicates & Candidate Keys ────────────────────────────────────

def _section_duplicates(
    con: duckdb.DuckDBPyConnection,
    tables: Dict[str, str],
    classes: Dict[str, Dict[str, str]],
    rpt: _Report,
) -> None:
    rpt.section("7. DUPLICATES & CANDIDATE KEYS")
    rpt.line(f"  {'table':<25} {'dup_groups':>10}  candidate_key_columns")
    rpt.line(f"  {'-'*25} {'-'*10}  {'-'*40}")

    for tbl_name in tables:
        cols = list(classes[tbl_name].keys())
        col_list = ", ".join(cols)
        n_rows = con.execute(f"SELECT COUNT(*) FROM {tbl_name}").fetchone()[0]

        n_exact_dup = con.execute(f"""
            SELECT COUNT(*) FROM (
                SELECT {col_list}, COUNT(*) AS cnt FROM {tbl_name}
                GROUP BY {col_list} HAVING cnt > 1
            )
        """).fetchone()[0]
        rpt.record(tbl_name, "*", "exact_duplicate_groups", value=n_exact_dup)

        ck_cols = []
        if n_rows > 0:
            for col in cols:
                n_non_null = con.execute(
                    f"SELECT COUNT(*) FROM {tbl_name} WHERE {col} IS NOT NULL"
                ).fetchone()[0]
                if n_non_null < 10:
                    continue
                n_distinct = con.execute(
                    f"SELECT COUNT(DISTINCT TRIM(CAST({col} AS VARCHAR))) FROM {tbl_name} WHERE {col} IS NOT NULL"
                ).fetchone()[0]
                if n_distinct / n_non_null >= 0.999:
                    ck_cols.append(col)
                    rpt.record(tbl_name, col, "candidate_key",
                               distinct=n_distinct, non_null=n_non_null,
                               uniqueness=round(n_distinct / n_non_null, 4))

        dup_str = f"{n_exact_dup} (WARN)" if n_exact_dup > 0 else f"{n_exact_dup} (OK)"
        rpt.line(f"  {tbl_name:<25} {dup_str:>10}  {', '.join(ck_cols) or '—'}")


# ── Section 9: Label / Class Balance ──────────────────────────────────────────

def _section_label(
    con: duckdb.DuckDBPyConnection,
    tables: Dict[str, str],
    classes: Dict[str, Dict[str, str]],
    rpt: _Report,
) -> None:
    rpt.section("8. LABEL / CLASS BALANCE")

    found_any = False
    label_names = {"label", "target", "class", "y"}

    for tbl_name in tables:
        cols = list(classes[tbl_name].keys())
        n_rows = con.execute(f"SELECT COUNT(*) FROM {tbl_name}").fetchone()[0]
        if n_rows == 0:
            continue

        for col in cols:
            is_label_name = col.lower() in label_names
            n_distinct = con.execute(f"""
                SELECT COUNT(DISTINCT TRIM(CAST({col} AS VARCHAR)))
                FROM {tbl_name} WHERE {col} IS NOT NULL
            """).fetchone()[0]

            if not (is_label_name or n_distinct == 2):
                continue

            balance = con.execute(f"""
                SELECT TRIM(CAST({col} AS VARCHAR)) AS val,
                       COUNT(*) AS n,
                       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
                FROM {tbl_name} WHERE {col} IS NOT NULL
                GROUP BY val ORDER BY val
            """).fetchdf()

            if not found_any:
                rpt.line(f"  {'table.column':<35} {'value':<15} {'n':>7}  {'%':>6}")
                found_any = True

            for _, row in balance.iterrows():
                label_col = f"{tbl_name}.{col}"
                rpt.line(f"  {label_col:<35} {str(row['val']):<15} {int(row['n']):>7,}  {row['pct']:>5.2f}%")
                rpt.record(tbl_name, col, "label_balance",
                           label=row["val"], count=int(row["n"]), pct=float(row["pct"]))

    if not found_any:
        rpt.line("  (no label/class column detected)")


# ── Main entry point ──────────────────────────────────────────────────────────

def run_quality_audit(
    file_paths: Optional[List[str | Path]] = None,
    output_path: Optional[str | Path] = None,
) -> str:
    """
    Run a schema-agnostic data quality audit.

    Parameters
    ----------
    file_paths : list of CSV/Parquet paths (uses default project data if None)
    output_path : optional path to write report to; default is the script's folder
                  (quality_audit_report.txt next to quality_report.py).

    Returns
    -------
    The report as a string (printed to terminal and written to default or --output path).
    """
    if file_paths is None or len(file_paths) == 0:
        paths = _default_file_paths()
    else:
        paths = [Path(p) for p in file_paths]

    script_dir = Path(__file__).resolve().parent
    default_out = script_dir / "quality_audit_report.txt"
    write_to_file: Optional[Path] = None
    if output_path is not None:
        output_path = Path(output_path)
        root = _resolve_project_root()
        resolved = output_path.resolve()
        processed_dir = (root / "data" / "processed").resolve()
        if processed_dir in resolved.parents or resolved.parent == processed_dir:
            print("[quality_report] Output under data/processed/ not allowed; using script folder.")
            write_to_file = default_out
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_to_file = output_path
    else:
        write_to_file = default_out

    rpt = _Report()
    rpt.line("QUALITY AUDIT REPORT  (detect only — no fixes applied)")
    rpt.line(f"Files: {[str(p) for p in paths]}")

    con = duckdb.connect()
    tables, table_paths = _ingest_files(con, paths)
    rpt.line(f"Tables loaded: {list(tables.keys())}")

    classes = _section_schema(con, tables, table_paths, rpt)
    _section_keys(con, tables, classes, rpt)
    _section_missingness(con, tables, classes, rpt)
    _section_normalization(con, tables, classes, rpt)
    _section_robust_stats(con, tables, classes, rpt)
    _section_fingerprint(con, tables, classes, rpt)
    _section_duplicates(con, tables, classes, rpt)
    _section_label(con, tables, classes, rpt)

    rpt.section("AUDIT COMPLETE")
    con.close()

    report_text = rpt.text()
    print(report_text)

    write_to_file.write_text(report_text, encoding="utf-8")
    print(f"[quality_report] Written to: {write_to_file}")

    return report_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Schema-agnostic data quality audit. Accepts any CSV/Parquet files.",
    )
    parser.add_argument(
        "files", nargs="*", default=[],
        help="CSV or Parquet file paths. If omitted, loads default project data.",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Path to write report to (default: quality_audit_report.txt in the script's folder)",
    )
    args = parser.parse_args()
    run_quality_audit(
        file_paths=args.files if args.files else None,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
