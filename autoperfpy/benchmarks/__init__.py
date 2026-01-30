"""Benchmarking module for AutoPerfPy.

Re-exports generic benchmarks from trackiq_core.
"""

from trackiq_core.benchmarks import BatchingTradeoffBenchmark, LLMLatencyBenchmark

__all__ = [
    "BatchingTradeoffBenchmark",
    "LLMLatencyBenchmark",
]
