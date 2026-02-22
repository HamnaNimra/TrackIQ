"""Tests for dependency loader namespace behavior."""

from __future__ import annotations

import importlib.util

# Importing deps installs lightweight namespace packages for trackiq_core modules.
from trackiq_compare import deps  # noqa: F401


def test_deps_namespace_allows_trackiq_core_ui_resolution() -> None:
    """trackiq_compare.deps should not block resolving trackiq_core.ui modules."""
    spec = importlib.util.find_spec("trackiq_core.ui")
    assert spec is not None
