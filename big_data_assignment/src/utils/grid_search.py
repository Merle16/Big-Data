"""
Grid search utilities: expand config with list values into multiple scalar configs.
"""

from __future__ import annotations

import copy
import itertools
from typing import Any, Dict, Iterator


def _first_if_list(value: Any) -> Any:
    """If value is a list, return first element; otherwise return value."""
    return value[0] if isinstance(value, list) else value


def flatten_model_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a model params dict that may have list values into a scalar-only dict
    by taking the first element of each list. Use for single-run (no grid search).
    """
    return {k: _first_if_list(v) for k, v in params.items()}


def expand_model_grid(config: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """
    Yield one full config per grid point for the current model type.

    For config['model'][model_type], any value that is a list defines a dimension
    of the grid; scalars are treated as a single value. Yields deep copies of
    config with config['model'][model_type] replaced by a dict of scalars for
    each combination.
    """
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "logistic_regression")
    if model_type not in model_cfg:
        yield copy.deepcopy(config)
        return

    block = model_cfg[model_type]
    if not isinstance(block, dict):
        yield copy.deepcopy(config)
        return

    keys = list(block.keys())
    values = [block[k] if isinstance(block[k], list) else [block[k]] for k in keys]

    for combo in itertools.product(*values):
        one_point = dict(zip(keys, combo))
        new_config = copy.deepcopy(config)
        new_config.setdefault("model", {})[model_type] = one_point
        yield new_config


def count_grid_points(config: Dict[str, Any]) -> int:
    """Return the number of grid points for the current model type."""
    return sum(1 for _ in expand_model_grid(config))
