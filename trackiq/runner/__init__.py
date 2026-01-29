"""Benchmark runner for TrackIQ.

Orchestrates running a collector for a given duration and returns
exported results.
"""

from .runner import BenchmarkRunner

__all__ = ["BenchmarkRunner"]
