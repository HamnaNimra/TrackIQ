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
from autoperfpy.commands.compare import run_compare
from autoperfpy.commands.monitor import run_monitor_gpu, run_monitor_kv_cache
from autoperfpy.commands.report import run_report_html, run_report_pdf
from autoperfpy.commands.run import (
    run_auto_benchmarks_cli,
    run_devices_list,
    run_manual_single,
    run_profiles,
    run_with_profile,
)
from autoperfpy.commands.ui import run_ui

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
    "run_report_html",
    "run_report_pdf",
    "run_ui",
    "run_compare",
    "run_profiles",
    "run_with_profile",
    "run_devices_list",
    "run_auto_benchmarks_cli",
    "run_manual_single",
]
