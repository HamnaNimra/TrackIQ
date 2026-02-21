"""Command handler modules for AutoPerfPy CLI decomposition."""

from autoperfpy.commands.analyze import (
    run_analyze_dnn_pipeline,
    run_analyze_efficiency,
    run_analyze_latency,
    run_analyze_logs,
    run_analyze_tegrastats,
    run_analyze_variability,
)
from autoperfpy.commands.benchmark import run_benchmark_batching, run_benchmark_llm
from autoperfpy.commands.monitor import run_monitor_gpu, run_monitor_kv_cache

__all__ = [
    "run_analyze_latency",
    "run_analyze_logs",
    "run_analyze_dnn_pipeline",
    "run_analyze_tegrastats",
    "run_analyze_efficiency",
    "run_analyze_variability",
    "run_benchmark_batching",
    "run_benchmark_llm",
    "run_monitor_gpu",
    "run_monitor_kv_cache",
]
