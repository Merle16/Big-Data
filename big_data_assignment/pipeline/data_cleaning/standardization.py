"""Step 3: NFKD Unicode normalization + string & categorical standardization."""

import re
import unicodedata

import duckdb

_STRIP_PUNCT = re.compile(r"[^\w\s]")
_COLLAPSE_WS = re.compile(r"\s+")
_ID_LIKE_RE  = re.compile(r"^[a-zA-Z]{1,4}\d+$")


def _normalize(s: str) -> str | None:
    """NFKD + ASCII + strip punctuation + collapse whitespace. (from ilesh/dq.py)"""
    if not isinstance(s, str):
        return s
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = _STRIP_PUNCT.sub(" ", s)
    return _COLLAPSE_WS.sub(" ", s).strip() or None


def _fingerprint(s: str) -> str:
    """Sort-token fingerprint key for near-duplicate detection. (from quality_report.py)"""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s).lower().encode("ascii", "ignore").decode("ascii")
    return " ".join(sorted(set(_STRIP_PUNCT.sub(" ", s).split())))


class StringStandardizer:
    """Apply NFKD normalization to VARCHAR text columns.
    ID-like columns (e.g. tconst, nconst) are skipped to preserve join keys.
    Adds a __fp_{col} fingerprint column per normalized column for deduplication."""

    def _is_id_col(self, con: duckdb.DuckDBPyConnection, table: str, col: str) -> bool:
        sample = con.execute(f"""
            SELECT TRIM(CAST("{col}" AS VARCHAR)) AS v
            FROM {table} WHERE "{col}" IS NOT NULL LIMIT 100
        """).fetchdf()["v"].tolist()
        return bool(sample) and sum(_ID_LIKE_RE.match(str(v)) is not None for v in sample) / len(sample) > 0.8

    def transform(self, con: duckdb.DuckDBPyConnection, table: str) -> str:
        for name, fn in [("_normalize", _normalize), ("_fingerprint", _fingerprint)]:
            try:
                con.create_function(name, fn, [str], str, null_handling="special")
            except Exception:
                pass  # already registered on this connection

        cols = con.execute(f"DESCRIBE {table}").fetchall()
        exprs = []
        for col, dtype, *_ in cols:
            q = f'"{col}"'
            if "VARCHAR" in dtype.upper() and not self._is_id_col(con, table, col):
                exprs.append(f'_normalize({q}) AS {q}')
                exprs.append(f'_fingerprint({q}) AS "__fp_{col}"')
            else:
                exprs.append(q)

        out = f"{table}_std"
        con.execute(f"CREATE OR REPLACE VIEW {out} AS SELECT {', '.join(exprs)} FROM {table}")
        return out
