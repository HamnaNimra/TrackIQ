"""AutoPerfPy - Comprehensive Performance Analysis Toolkit."""

__version__ = "1.0"
__author__ = "Hamna Nimra"
__description__ = "Performance analysis and benchmarking toolkit for NVIDIA platforms"

from .config import Config, ConfigManager
from .collectors import (
    CollectorBase,
    CollectorSample,
    CollectorExport,
    SyntheticCollector,
)
from .core import (
    BaseAnalyzer,
    BaseBenchmark,
    BaseMonitor,
    DataLoader,
    LatencyStats,
    PerformanceComparator,
    RegressionDetector,
    EfficiencyMetrics,
    EfficiencyCalculator,
    BatchEfficiencyAnalyzer,
    # Tegrastats (DriveOS/Jetson)
    TegrastatsParser,
    TegrastatsCalculator,
    TegrastatsSnapshot,
    # DNN Pipeline (TensorRT/DriveWorks)
    DNNPipelineParser,
    DNNPipelineCalculator,
    InferenceRun,
    LayerTiming,
)
from .analyzers import (
    PercentileLatencyAnalyzer,
    LogAnalyzer,
    EfficiencyAnalyzer,
    VariabilityAnalyzer,
    TegrastatsAnalyzer,
    DNNPipelineAnalyzer,
)
from .benchmarks import BatchingTradeoffBenchmark, LLMLatencyBenchmark
from .monitoring import GPUMemoryMonitor, LLMKVCacheMonitor
from .reporting import PerformanceVisualizer, PDFReportGenerator, HTMLReportGenerator
from .profiles import (
    Profile,
    CollectorType,
    get_profile,
    list_profiles,
    register_profile,
    get_profile_info,
    validate_profile_collector,
    validate_profile_precision,
    ProfileValidationError,
)

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
