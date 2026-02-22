"""Benchmarking module for AutoPerfPy.

Exports compatibility shims backed by ``trackiq_core`` implementations.
"""

from .latency import BatchingTradeoffBenchmark, LLMLatencyBenchmark
from .vllm_bench import run_inference_benchmark, save_inference_benchmark

__all__ = [
    "BatchingTradeoffBenchmark",
    "LLMLatencyBenchmark",
    "run_inference_benchmark",
    "save_inference_benchmark",
]
