"""TrackIQ - Generic performance tracking and analysis library."""

__version__ = "0.1.0"

from .config import Config, ConfigManager
from .collectors import (
    CollectorBase, CollectorSample, CollectorExport,
    SyntheticCollector, PsutilCollector, NVMLCollector,
)
from .core import (
    BaseAnalyzer, BaseBenchmark, BaseMonitor, AnalysisResult,
    DataLoader, LatencyStats, PerformanceComparator,
    RegressionDetector, RegressionThreshold, MetricComparison,
    EfficiencyMetrics, EfficiencyCalculator, BatchEfficiencyAnalyzer,
)
from .hardware import (
    query_nvidia_smi, parse_gpu_metrics, get_memory_metrics,
    get_performance_metrics, DEFAULT_NVIDIA_SMI_TIMEOUT,
)
__all__ = [
    "Config", "ConfigManager",
    "CollectorBase", "CollectorSample", "CollectorExport",
    "SyntheticCollector", "PsutilCollector", "NVMLCollector",
    "BaseAnalyzer", "BaseBenchmark", "BaseMonitor", "AnalysisResult",
    "DataLoader", "LatencyStats", "PerformanceComparator",
    "RegressionDetector", "RegressionThreshold", "MetricComparison",
    "EfficiencyMetrics", "EfficiencyCalculator", "BatchEfficiencyAnalyzer",
    "query_nvidia_smi", "parse_gpu_metrics", "get_memory_metrics",
    "get_performance_metrics", "DEFAULT_NVIDIA_SMI_TIMEOUT",
]
