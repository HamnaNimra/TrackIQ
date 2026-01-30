"""Shared dictionary utilities."""

from typing import Any, Dict


def safe_get(d: Dict, *keys, default: Any = None) -> Any:
    """Safely get nested dictionary value.

    Args:
        d: Dictionary to search
        *keys: Keys to traverse (e.g. "cpu", "avg_utilization_percent")
        default: Default value if key not found

    Returns:
        Value at nested key or default
    """
    value = d
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key, default)
        else:
            return default
    return value if value is not None else default
