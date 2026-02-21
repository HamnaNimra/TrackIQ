"""AutoPerfPy benchmark compatibility layer.

This module intentionally re-exports the shared benchmark implementations
from ``trackiq_core`` so all tools use one source of truth.
"""

from trackiq_core.benchmarks.latency import BatchingTradeoffBenchmark, LLMLatencyBenchmark

__all__ = ["BatchingTradeoffBenchmark", "LLMLatencyBenchmark"]
