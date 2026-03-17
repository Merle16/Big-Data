"""Test that config.yaml is valid and complete."""
import pytest
import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def test_config_exists():
    assert CONFIG_PATH.exists(), f"config.yaml not found at {CONFIG_PATH}"


def test_config_valid_yaml():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    assert cfg is not None


def test_required_sections_present():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    for section in ["global", "imputation", "feature_engineering", "models"]:
        assert section in cfg, f"Missing required config section: {section}"


def test_seed_is_integer():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    seed = cfg["global"]["seed"]
    assert isinstance(seed, int), f"seed must be int, got {type(seed)}"


def test_goodness_weights_sum_to_one():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    weights = cfg["feature_quality"]["goodness_weights"]
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-6, f"goodness_weights must sum to 1.0, got {total}"


def test_pipeline_version_exists():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    assert "pipeline" in cfg, "Missing 'pipeline' section in config.yaml"
    assert "version" in cfg["pipeline"], "Missing pipeline.version in config.yaml"
