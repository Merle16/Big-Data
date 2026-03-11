"""Step 3: NFKD Unicode normalization + string & categorical standardization."""

import unicodedata
import re

import duckdb

from .s0_enforce_schema import get_id_cols

_STRIP_PUNCT = re.compile(r"[^\w\s]")
_COLLAPSE_WS = re.compile(r"\s+")


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
    ID columns (from enforce_schema) are skipped to preserve join keys.
    Adds a __fp_{col} fingerprint column per normalized column for deduplication."""

    def transform(self, con: duckdb.DuckDBPyConnection, table: str) -> str:
        for name, fn in [("_normalize", _normalize), ("_fingerprint", _fingerprint)]:
            try:
                con.create_function(name, fn, [str], str, null_handling="special")
            except Exception:
                pass

        skip = set(get_id_cols(table))
        cols = con.execute(f"DESCRIBE {table}").fetchall()
        exprs = []
        for col, dtype, *_ in cols:
            q = f'"{col}"'
            if "VARCHAR" in dtype.upper() and col not in skip:
                exprs.append(f'_normalize({q}) AS {q}')
                exprs.append(f'_fingerprint({q}) AS "__fp_{col}"')
            else:
                exprs.append(q)

        out = f"{table}_std"
        con.execute(f"CREATE OR REPLACE VIEW {out} AS SELECT {', '.join(exprs)} FROM {table}")
        return out
