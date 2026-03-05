from __future__ import annotations

import os
from typing import Any, Dict, Optional

import yaml


def _default_config_path() -> str:
    """
    Return the default path to the central config file.
    """
    # src/utils/config.py -> src/utils -> src -> project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(project_root, "config", "config.yaml")


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the YAML configuration file and return it as a dictionary.

    Parameters
    ----------
    path:
        Optional path to a YAML config file. If not provided, the default
        central config file under ``config/config.yaml`` is used.
    """
    config_path = path or _default_config_path()
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Loaded config is not a mapping/dictionary.")

    return config


def resolve_path_from_config(config: Dict[str, Any], *keys: str, base: Optional[str] = None) -> str:
    """
    Resolve a path under the project root using values from the config.

    ``keys`` is a path of nested keys inside the config dictionary.
    """
    node: Any = config
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"Key path {'/'.join(keys)} not found in config.")
        node = node[key]

    rel_path = str(node)
    # src/utils/config.py -> src/utils -> src -> project root
    base_dir = base or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base_dir, rel_path)

