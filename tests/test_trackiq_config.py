"""Unit tests for trackiq config loader."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from trackiq.config import Config, ConfigManager


class TestConfig:
    """Config container get/update."""

    def test_get_nested_key(self):
        c = Config({"benchmark": {"batch_sizes": [1, 4, 8]}})
        assert c.get("benchmark.batch_sizes") == [1, 4, 8]

    def test_get_missing_returns_default(self):
        c = Config({})
        assert c.get("missing.key", "default") == "default"

    def test_to_dict(self):
        c = Config({"a": 1, "b": {"c": 2}})
        d = c.to_dict()
        assert d["a"] == 1
        assert d["b"]["c"] == 2


class TestConfigManager:
    """Config file parsing (YAML/JSON) without error."""

    def test_load_yaml_parses_without_error(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(
            "benchmark:\n  batch_sizes: [1, 4, 8]\nlatency:\n  threshold_ms: 100\n"
        )
        config = ConfigManager.load_yaml(str(yaml_path))
        assert config.get("benchmark.batch_sizes") == [1, 4, 8]
        assert config.get("latency.threshold_ms") == 100

    def test_load_json_parses_without_error(self, tmp_path):
        json_path = tmp_path / "config.json"
        json_path.write_text(
            '{"benchmark": {"batch_sizes": [1, 4, 8]}, "latency": {"threshold_ms": 100}}'
        )
        config = ConfigManager.load_json(str(json_path))
        assert config.get("benchmark.batch_sizes") == [1, 4, 8]
        assert config.get("latency.threshold_ms") == 100

    def test_load_or_default_with_missing_file(self):
        config = ConfigManager.load_or_default(
            "/nonexistent.yaml", default_config={"a": 1}
        )
        assert config.get("a") == 1

    def test_load_or_default_with_existing_yaml(self, tmp_path):
        yaml_path = tmp_path / "c.yaml"
        yaml_path.write_text("key: value\n")
        config = ConfigManager.load_or_default(str(yaml_path))
        assert config.get("key") == "value"

    def test_save_and_load_roundtrip_yaml(self, tmp_path):
        c = Config({"x": 1, "y": {"z": 2}})
        path = tmp_path / "out.yaml"
        ConfigManager.save_yaml(c, str(path))
        loaded = ConfigManager.load_yaml(str(path))
        assert loaded.get("x") == 1
        assert loaded.get("y.z") == 2

    def test_save_and_load_roundtrip_json(self, tmp_path):
        c = Config({"x": 1, "y": {"z": 2}})
        path = tmp_path / "out.json"
        ConfigManager.save_json(c, str(path))
        loaded = ConfigManager.load_json(str(path))
        assert loaded.get("x") == 1
        assert loaded.get("y.z") == 2
