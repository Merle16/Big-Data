"""HTML report generation for the full pipeline.

Entry point
-----------
  make_full_report.run(out_html)   Build and write full_pipeline_report.html
"""
from .make_full_report import run

__all__ = ["run"]
