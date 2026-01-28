"""AutoPerfPy - Comprehensive Performance Analysis Toolkit."""

__version__ = "0.1.0"
__author__ = "Hamna Nimra"
__description__ = "Performance analysis and benchmarking toolkit for NVIDIA platforms"

from .config import Config, ConfigManager
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
from .reporting import PerformanceVisualizer, PDFReportGenerator

__all__ = [
    "Config",
    "ConfigManager",
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
]
