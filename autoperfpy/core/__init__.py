"""Core module for AutoPerfPy."""

from .base import BaseAnalyzer, BaseBenchmark, BaseMonitor, AnalysisResult
from .utils import DataLoader, LatencyStats, PerformanceComparator
from .regression import RegressionDetector, RegressionThreshold
from .efficiency import EfficiencyMetrics, EfficiencyCalculator, BatchEfficiencyAnalyzer
from .tegrastats import (
    TegrastatsParser,
    TegrastatsCalculator,
    TegrastatsSnapshot,
    TegrastatsAggregateStats,
    CPUCoreStats,
    GPUStats,
    MemoryStats,
    ThermalStats,
)
from .dnn_pipeline import (
    DNNPipelineParser,
    DNNPipelineCalculator,
    InferenceRun,
    LayerTiming,
    MemoryTransfer,
    EngineOptimizationMetrics,
)

__all__ = [
    "BaseAnalyzer",
    "BaseBenchmark",
    "BaseMonitor",
    "AnalysisResult",
    "DataLoader",
    "LatencyStats",
    "PerformanceComparator",
    "RegressionDetector",
    "RegressionThreshold",
    # Efficiency metrics
    "EfficiencyMetrics",
    "EfficiencyCalculator",
    "BatchEfficiencyAnalyzer",
    # Tegrastats (DriveOS/Jetson)
    "TegrastatsParser",
    "TegrastatsCalculator",
    "TegrastatsSnapshot",
    "TegrastatsAggregateStats",
    "CPUCoreStats",
    "GPUStats",
    "MemoryStats",
    "ThermalStats",
    # DNN Pipeline (TensorRT/DriveWorks)
    "DNNPipelineParser",
    "DNNPipelineCalculator",
    "InferenceRun",
    "LayerTiming",
    "MemoryTransfer",
    "EngineOptimizationMetrics",
]
