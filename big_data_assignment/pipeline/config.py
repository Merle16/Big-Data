"""
pipeline/config.py — centralised configuration loader.

Reads config.yaml from the project root (big_data_assignment/).
Falls back to an empty dict so every caller must supply its own default
via  cfg.get("key", default).

Usage (from any pipeline module):
    try:
        from ..config import CFG
    except ImportError:          # standalone / script execution
        from pipeline.config import CFG
"""
from __future__ import annotations

from pathlib import Path

import yaml

_ROOT     = Path(__file__).resolve().parent.parent  # big_data_assignment/
_CFG_PATH = _ROOT / "config.yaml"


def _load() -> dict:
    if _CFG_PATH.exists():
        with open(_CFG_PATH, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


CFG: dict = _load()
