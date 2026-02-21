"""Monitoring module for AutoPerfPy.

Exports compatibility shims backed by ``trackiq_core`` monitoring.
"""

from trackiq_core.hardware import (
    get_memory_metrics,
    get_performance_metrics,
    parse_gpu_metrics,
    query_nvidia_smi,
)
from .gpu import GPUMemoryMonitor, LLMKVCacheMonitor

__all__ = [
    "GPUMemoryMonitor",
    "LLMKVCacheMonitor",
    # Shared GPU utilities
    "query_nvidia_smi",
    "parse_gpu_metrics",
    "get_memory_metrics",
    "get_performance_metrics",
]
