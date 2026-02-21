"""AutoPerfPy - Comprehensive Performance Analysis Toolkit."""

__version__ = "1.0"
__author__ = "Hamna Nimra"
__description__ = "Performance analysis and benchmarking toolkit for NVIDIA platforms"

from .analyzers import (
    DNNPipelineAnalyzer,
    EfficiencyAnalyzer,
    LogAnalyzer,
    PercentileLatencyAnalyzer,
    TegrastatsAnalyzer,
    VariabilityAnalyzer,
)
from .benchmarks import BatchingTradeoffBenchmark, LLMLatencyBenchmark
from .collectors import (
    CollectorBase,
    CollectorExport,
    CollectorSample,
    SyntheticCollector,
)
from .config import Config, ConfigManager
from .core import (  # DNN Pipeline (TensorRT/DriveWorks); Tegrastats (DriveOS/Jetson)
    BaseAnalyzer,
    BaseBenchmark,
    BaseMonitor,
    BatchEfficiencyAnalyzer,
    DataLoader,
    DNNPipelineCalculator,
    DNNPipelineParser,
    EfficiencyCalculator,
    EfficiencyMetrics,
    InferenceRun,
    LatencyStats,
    LayerTiming,
    PerformanceComparator,
    RegressionDetector,
    TegrastatsCalculator,
    TegrastatsParser,
    TegrastatsSnapshot,
)
from .monitoring import GPUMemoryMonitor, LLMKVCacheMonitor
from .profiles import (
    CollectorType,
    Profile,
    ProfileValidationError,
    get_profile,
    get_profile_info,
    list_profiles,
    register_profile,
    validate_profile_collector,
    validate_profile_precision,
)
from .reporting import HTMLReportGenerator, PDFReportGenerator, PerformanceVisualizer

__all__ = [
    "Config",
    "ConfigManager",
    # Collectors
    "CollectorBase",
    "CollectorSample",
    "CollectorExport",
    "SyntheticCollector",
    # Core base classes
    "BaseAnalyzer",
    "BaseBenchmark",
    "BaseMonitor",
    "DataLoader",
    "LatencyStats",
    "PerformanceComparator",
    "RegressionDetector",
    # Efficiency metrics
    "EfficiencyMetrics",
    "EfficiencyCalculator",
    "BatchEfficiencyAnalyzer",
    # Tegrastats (DriveOS/Jetson)
    "TegrastatsParser",
    "TegrastatsCalculator",
    "TegrastatsSnapshot",
    # DNN Pipeline (TensorRT/DriveWorks)
    "DNNPipelineParser",
    "DNNPipelineCalculator",
    "InferenceRun",
    "LayerTiming",
    # Analyzers
    "PercentileLatencyAnalyzer",
    "LogAnalyzer",
    "EfficiencyAnalyzer",
    "VariabilityAnalyzer",
    "TegrastatsAnalyzer",
    "DNNPipelineAnalyzer",
    # Benchmarks
    "BatchingTradeoffBenchmark",
    "LLMLatencyBenchmark",
    # Monitoring
    "GPUMemoryMonitor",
    "LLMKVCacheMonitor",
    # Reporting
    "PerformanceVisualizer",
    "PDFReportGenerator",
    "HTMLReportGenerator",
    # Profiles
    "Profile",
    "CollectorType",
    "get_profile",
    "list_profiles",
    "register_profile",
    "get_profile_info",
    "validate_profile_collector",
    "validate_profile_precision",
    "ProfileValidationError",
]
