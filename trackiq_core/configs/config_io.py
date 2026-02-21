"""Shared configuration file I/O (YAML/JSON load/save as dict)."""

import json
import os
from typing import Any

# Optional YAML; allow runtime to have pyyaml
try:
    import yaml

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


def ensure_parent_dir(filepath: str) -> None:
    """Create parent directory of filepath if needed."""
    parent = os.path.dirname(filepath) or "."
    os.makedirs(parent, exist_ok=True)


def load_yaml_file(filepath: str) -> dict[str, Any]:
    """Load a YAML file into a dictionary.

    Args:
        filepath: Path to YAML file

    Returns:
        Loaded config as dict; empty dict if file is empty or invalid
    """
    if not _YAML_AVAILABLE:
        raise ImportError("PyYAML is required for load_yaml_file")
    with open(filepath, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def load_json_file(filepath: str) -> dict[str, Any]:
    """Load a JSON file into a dictionary.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded config as dict; empty dict if file is empty
    """
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    return data if data is not None else {}


def save_yaml_file(filepath: str, data: dict[str, Any]) -> None:
    """Save a dictionary to a YAML file."""
    if not _YAML_AVAILABLE:
        raise ImportError("PyYAML is required for save_yaml_file")
    ensure_parent_dir(filepath)
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False)


def save_json_file(filepath: str, data: dict[str, Any], indent: int = 2) -> None:
    """Save a dictionary to a JSON file."""
    ensure_parent_dir(filepath)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
