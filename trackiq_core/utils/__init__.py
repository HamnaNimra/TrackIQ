"""Shared utilities, analyzers, compare, core base classes."""

from .base import BaseAnalyzer, BaseBenchmark, BaseMonitor
try:
    from .analysis_utils import DataLoader, LatencyStats, PerformanceComparator
except Exception:  # pragma: no cover - optional dependency path (e.g., missing pandas)
    DataLoader = None
    LatencyStats = None
    PerformanceComparator = None
try:
    from .efficiency import (
        EfficiencyMetrics,
        EfficiencyCalculator,
        BatchEfficiencyAnalyzer,
    )
except Exception:  # pragma: no cover - optional dependency path (e.g., missing numpy)
    EfficiencyMetrics = None
    EfficiencyCalculator = None
    BatchEfficiencyAnalyzer = None
from .errors import (
    TrackIQError,
    HardwareNotFoundError,
    ConfigError,
    DependencyError,
    ProfileValidationError,
)
from trackiq_core.schemas import AnalysisResult
from .compare import RegressionDetector, RegressionThreshold, MetricComparison
try:
    from .analyzers import (
        PercentileLatencyAnalyzer,
        LogAnalyzer,
        EfficiencyAnalyzer,
        VariabilityAnalyzer,
    )
except Exception:  # pragma: no cover - optional dependency path
    PercentileLatencyAnalyzer = None
    LogAnalyzer = None
    EfficiencyAnalyzer = None
    VariabilityAnalyzer = None

__all__ = [
    "BaseAnalyzer",
    "BaseBenchmark",
    "BaseMonitor",
    "TrackIQError",
    "HardwareNotFoundError",
    "ConfigError",
    "DependencyError",
    "ProfileValidationError",
    "AnalysisResult",
    "RegressionDetector",
    "RegressionThreshold",
    "MetricComparison",
]

if DataLoader is not None:
    __all__.extend(["DataLoader", "LatencyStats", "PerformanceComparator"])
if EfficiencyMetrics is not None:
    __all__.extend(
        ["EfficiencyMetrics", "EfficiencyCalculator", "BatchEfficiencyAnalyzer"]
    )
if PercentileLatencyAnalyzer is not None:
    __all__.extend(
        [
            "PercentileLatencyAnalyzer",
            "LogAnalyzer",
            "EfficiencyAnalyzer",
            "VariabilityAnalyzer",
        ]
    )
