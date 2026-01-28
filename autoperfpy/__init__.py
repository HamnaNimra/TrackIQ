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
)
from .analyzers import PercentileLatencyAnalyzer, LogAnalyzer
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
    "PercentileLatencyAnalyzer",
    "LogAnalyzer",
    "BatchingTradeoffBenchmark",
    "LLMLatencyBenchmark",
    "GPUMemoryMonitor",
    "LLMKVCacheMonitor",
    "PerformanceVisualizer",
    "PDFReportGenerator",
]
