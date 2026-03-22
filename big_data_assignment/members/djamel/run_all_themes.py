#!/usr/bin/env python3
"""
Run all 12 pipeline themes in sequence.

Usage:
    python run_all_themes.py                   # run all
    python run_all_themes.py --from 5          # start from theme 5
    python run_all_themes.py --only 4,5,6      # run only specific themes
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

MEMBER = Path(__file__).resolve().parent

THEMES = [
    (1,  "theme_01_repository_audit"),
    (2,  "theme_02_engine_choice"),
    (3,  "theme_03_many_to_many"),
    (4,  "theme_04_data_cleaning"),
    (4,  "theme_04b_genre_enrichment"),
    (5,  "theme_05_candidate_features"),
    (6,  "theme_06_feature_selection"),
    (7,  "theme_07_feature_quality"),
    (8,  "theme_08_model_training"),
    (9,  "theme_09_diagnostics"),
    (10, "theme_10_reduced_model"),
    (11, "theme_11_export"),
    (12, "theme_12_rotten_tomatoes_join"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from",  dest="from_theme", type=int, default=1)
    parser.add_argument("--only",  dest="only",       type=str, default=None)
    args = parser.parse_args()

    if args.only:
        only_set = {int(x.strip()) for x in args.only.split(",")}
    else:
        only_set = None

    sys.path.insert(0, str(MEMBER))
    state: dict = {}
    total_start = time.time()

    for num, module_name in THEMES:
        if only_set and num not in only_set:
            continue
        if only_set is None and num < args.from_theme:
            continue

        print(f"\n{'='*60}")
        print(f"  Theme {num:02d} — {module_name}")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            import importlib
            mod = importlib.import_module(module_name)
            state = mod.run(state)
            elapsed = time.time() - t0
            print(f"  ✓ Done in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ✗ FAILED in {elapsed:.1f}s: {e}")
            traceback.print_exc()
            print(f"  Continuing with next theme...")

    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  All themes completed in {total:.1f}s")
    out = MEMBER / "outputs_restart"
    html_files = sorted(out.glob("theme_*.html"))
    print(f"  HTML reports ({len(html_files)}):")
    for f in html_files:
        print(f"    {f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
