"""External dataset enrichment — genre, Rotten Tomatoes, and Oscar.

Entry points
------------
  genre.run(state, genre_dir, out_dir)       Genre labels from Movies_by_Genre/
  rt_oscar.run(state)                        RT scores + Oscar nominations
"""
from .genre import run as run_genre
from .rt_oscar import run as run_rt_oscar

__all__ = ["run_genre", "run_rt_oscar"]
