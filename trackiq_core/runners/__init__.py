"""Benchmark runner module for TrackIQ.

Provides BenchmarkRunner class and utilities for running benchmarks
across devices with different configurations.
"""

from .benchmark_runner import (
    BenchmarkRunner,
    run_single_benchmark,
    run_auto_benchmarks,
    make_run_label,
    base_collector_config,
)

__all__ = [
    "BenchmarkRunner",
    "run_single_benchmark",
    "run_auto_benchmarks",
    "make_run_label",
    "base_collector_config",
]
