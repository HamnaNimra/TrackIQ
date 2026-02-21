"""Benchmarking module for AutoPerfPy.

Exports compatibility shims backed by ``trackiq_core`` implementations.
"""

from .latency import BatchingTradeoffBenchmark, LLMLatencyBenchmark

__all__ = [
    "BatchingTradeoffBenchmark",
    "LLMLatencyBenchmark",
]
