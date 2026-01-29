"""Collectors module for AutoPerfPy.

This module provides data collection interfaces and implementations for
gathering performance metrics from various sources.

Available Collectors:
    - CollectorBase: Abstract base class defining the collector interface
    - SyntheticCollector: Generates realistic synthetic performance data

Data Classes:
    - CollectorSample: Single data point with timestamp and metrics
    - CollectorExport: Complete export of all collected data

Example usage:
    from autoperfpy.collectors import SyntheticCollector

    # Create and run a synthetic collector
    collector = SyntheticCollector()
    collector.start()

    for _ in range(100):
        metrics = collector.sample(time.time())
        time.sleep(0.1)

    collector.stop()
    export = collector.export()
    print(f"Collected {len(export.samples)} samples")
    print(f"Mean latency: {export.summary['latency']['mean_ms']:.2f}ms")
"""

from .base import CollectorBase, CollectorSample, CollectorExport
from .synthetic import SyntheticCollector

__all__ = [
    "CollectorBase",
    "CollectorSample",
    "CollectorExport",
    "SyntheticCollector",
]
