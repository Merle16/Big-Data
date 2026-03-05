from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List

import pandas as pd

from src.utils.config import resolve_path_from_config


def _iter_json_objects(path: str) -> Iterable[Dict[str, Any]]:
    """
    Iterate over JSON objects from either a JSONL file (one JSON per line)
    or a standard JSON array file.
    """
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            data = json.load(f)
            for obj in data:
                if isinstance(obj, dict):
                    yield obj
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj


def convert_json_to_tabular(input_path: str, output_path: str, config: Dict[str, Any]) -> None:
    """
    Convert raw IMDB JSON reviews into a tabular CSV file.

    This function is intentionally minimal and can be extended later. It expects
    configuration under ``json_conversion`` in the central ``config.yaml``.
    """
    json_cfg = config.get("json_conversion", {})
    fields_cfg = json_cfg.get("fields", {})

    text_key = fields_cfg.get("text", "review")
    label_key = fields_cfg.get("label", "label")

    records: List[Dict[str, Any]] = []
    for obj in _iter_json_objects(input_path):
        record: Dict[str, Any] = {}
        if text_key in obj:
            record["review"] = obj[text_key]
        if label_key in obj:
            record["label"] = obj[label_key]
        if record:
            records.append(record)

    if not records:
        raise ValueError(f"No usable records were extracted from JSON file at {input_path}")

    df = pd.DataFrame.from_records(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


def convert_all_from_config(config: Dict[str, Any]) -> None:
    """
    Convenience wrapper that uses the ``json_conversion`` paths in the config
    to convert all JSON files in the configured input directory.
    """
    input_dir = resolve_path_from_config(config, "json_conversion", "input_dir")
    output_dir = resolve_path_from_config(config, "json_conversion", "output_dir")

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"JSON input directory not found at {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith((".json", ".jsonl")):
            continue
        input_path = os.path.join(input_dir, filename)
        stem, _ = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{stem}.csv")
        convert_json_to_tabular(input_path, output_path, config)

