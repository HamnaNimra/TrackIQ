"""Compare command handler for AutoPerfPy CLI."""

from __future__ import annotations

from typing import Any


def run_compare(args: Any) -> int:
    """Compare current run against baseline (delegates to trackiq_core)."""
    from trackiq_core.cli.commands.compare import run_compare as trackiq_run_compare

    return trackiq_run_compare(args)
