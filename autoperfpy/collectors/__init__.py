"""Collectors module for AutoPerfPy.

This module provides data collection interfaces and implementations for
gathering performance metrics from various sources.

Available Collectors:
    - CollectorBase: Abstract base class defining the collector interface
    - SyntheticCollector: Generates realistic synthetic performance data
    - NVMLCollector: NVIDIA GPU metrics via pynvml (requires pynvml)
    - TegrastatsCollector: Jetson/DriveOS metrics via tegrastats
    - PsutilCollector: Cross-platform system metrics via psutil (requires psutil)

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

    # For NVIDIA GPU metrics:
    from autoperfpy.collectors import NVMLCollector
    collector = NVMLCollector(device_index=0)

    # For Jetson/DriveOS:
    from autoperfpy.collectors import TegrastatsCollector
    collector = TegrastatsCollector(mode="live")

    # For cross-platform system metrics:
    from autoperfpy.collectors import PsutilCollector
    collector = PsutilCollector()
"""

from .base import CollectorBase, CollectorSample, CollectorExport
from .synthetic import SyntheticCollector
from .nvml_collector import NVMLCollector
from .tegrastats_collector import TegrastatsCollector
from .psutil_collector import PsutilCollector

__all__ = [
    "CollectorBase",
    "CollectorSample",
    "CollectorExport",
    "SyntheticCollector",
    "NVMLCollector",
    "TegrastatsCollector",
    "PsutilCollector",
]
