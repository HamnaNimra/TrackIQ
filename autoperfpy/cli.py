"""Command-line interface for AutoPerfPy."""

import argparse
import json
import os
import sys
import time
from typing import Optional

from autoperfpy.config import ConfigManager
from autoperfpy.analyzers import (
    PercentileLatencyAnalyzer,
    LogAnalyzer,
    DNNPipelineAnalyzer,
    TegrastatsAnalyzer,
    EfficiencyAnalyzer,
    VariabilityAnalyzer,
)
from autoperfpy.benchmarks import BatchingTradeoffBenchmark, LLMLatencyBenchmark
from autoperfpy.monitoring import GPUMemoryMonitor
from autoperfpy.reporting import (
    PerformanceVisualizer,
    PDFReportGenerator,
    HTMLReportGenerator,
)
from autoperfpy.collectors import SyntheticCollector
from autoperfpy.profiles import (
    get_profile,
    get_profile_info,
    validate_profile_collector,
    CollectorType,
    ProfileValidationError,
)
from trackiq.compare import RegressionDetector, RegressionThreshold
from trackiq.errors import HardwareNotFoundError, DependencyError
from trackiq.platform import DeviceProfile, get_all_devices
import numpy as np

# Phase 5: auto/manual run
from autoperfpy.device_config import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_ITERATIONS,
    DEFAULT_WARMUP_RUNS,
    InferenceConfig,
    PRECISION_FP32,
    PRECISIONS,
    get_devices_and_configs_auto,
    resolve_device,
)
from autoperfpy.auto_runner import run_auto_benchmarks, run_single_benchmark


def setup_parser() -> argparse.ArgumentParser:
    """Setup command-line argument parser.

    Returns:
        ArgumentParser configured for AutoPerfPy
    """
    parser = argparse.ArgumentParser(
        prog="autoperfpy",
        description="AutoPerfPy - Performance Analysis Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with a profile
  autoperfpy run --profile automotive_safety --batch-size 4
  autoperfpy run --profile ci_smoke --duration 10
  autoperfpy run --profile edge_low_power --collector synthetic

  # List available profiles
  autoperfpy profiles --list
  autoperfpy profiles --info automotive_safety

  # List detected devices (for run --auto or --device)
  autoperfpy devices --list

  # Latency analysis
  autoperfpy analyze latency --csv data.csv
  autoperfpy analyze logs --log performance.log --threshold 50

  # DNN pipeline analysis
  autoperfpy analyze dnn-pipeline --csv layer_times.csv --batch-size 4
  autoperfpy analyze dnn-pipeline --profiler profiler_output.txt

  # Tegrastats analysis
  autoperfpy analyze tegrastats --log tegrastats.log

  # Efficiency analysis
  autoperfpy analyze efficiency --csv benchmark_data.csv

  # Variability analysis
  autoperfpy analyze variability --csv latency_data.csv

  # Benchmarking
  autoperfpy benchmark batching --batch-sizes 1,4,8,16
  autoperfpy benchmark llm --prompt-length 512

  # Monitoring
  autoperfpy monitor gpu --duration 300

  # Report generation
  autoperfpy report html --csv data.csv --output report.html --title "My Report"
  autoperfpy report pdf --csv data.csv --output report.pdf

Environment Variables:
  AUTOPERFPY_PROFILE    Default profile name
  AUTOPERFPY_CONFIG     Default config file path
  AUTOPERFPY_COLLECTOR  Default collector type (synthetic, nvml, tegrastats, psutil)
        """,
    )

    parser.add_argument("--config", help="Path to configuration file (YAML/JSON)")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument(
        "--profile",
        "-p",
        help="Performance profile to use (automotive_safety, edge_max_perf, edge_low_power, ci_smoke)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Profiles command
    profiles_parser = subparsers.add_parser(
        "profiles", help="List and inspect performance profiles"
    )
    profiles_parser.add_argument(
        "--list", "-l", action="store_true", help="List all available profiles"
    )
    profiles_parser.add_argument(
        "--info", "-i", metavar="NAME", help="Show detailed info for a profile"
    )

    # Devices command (list detected hardware for run --auto / --device)
    devices_parser = subparsers.add_parser(
        "devices",
        help="List detected devices (GPUs, CPU, Jetson/DRIVE). Use before run --auto.",
    )
    devices_parser.add_argument(
        "--list", "-l", action="store_true", help="List all detected devices (default)"
    )

    # Run command (profile-based, auto, or manual)
    run_parser = subparsers.add_parser(
        "run", help="Run performance test (auto-detect devices, profile, or manual)"
    )
    run_parser.add_argument(
        "--profile",
        "-p",
        default=None,
        help="Profile to use (e.g. ci_smoke). If omitted, auto mode runs on all detected devices.",
    )
    run_parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatic mode: detect all devices, enumerate configs, run benchmarks (default if no --profile)",
    )
    run_parser.add_argument(
        "--manual",
        action="store_true",
        help="Manual mode: use --device, --collector, --precision, etc. for a single run",
    )
    run_parser.add_argument(
        "--collector",
        "-c",
        default=os.environ.get("AUTOPERFPY_COLLECTOR", "synthetic"),
        choices=["synthetic", "nvml", "tegrastats", "psutil"],
        help="Collector type (manual mode; default: synthetic)",
    )
    run_parser.add_argument(
        "--duration",
        "-d",
        type=int,
        help="Duration in seconds (default: 10 for auto, profile value for profile)",
    )
    run_parser.add_argument(
        "--batch-size", "-b", type=int, help="Batch size (manual or override)"
    )
    run_parser.add_argument(
        "--iterations", "-n", type=int, help="Override number of iterations"
    )
    run_parser.add_argument(
        "--warmup", "-w", type=int, help="Override warmup iterations"
    )
    run_parser.add_argument(
        "--export", "-e", metavar="FILE", help="Export results to JSON file"
    )
    run_parser.add_argument(
        "--quiet", "-q", action="store_true", help="Minimal output (summary only)"
    )
    run_parser.add_argument(
        "--validate-only", action="store_true", help="Validate profile and exit"
    )
    run_parser.add_argument(
        "--device",
        "-D",
        help="Device ID (e.g. nvidia_0, cpu_0) or GPU index (e.g. 0). Manual mode.",
    )
    run_parser.add_argument(
        "--precision",
        "-P",
        choices=["fp32", "fp16", "int8"],
        default="fp32",
        help="Inference precision (default: fp32)",
    )
    run_parser.add_argument(
        "--devices",
        metavar="IDS",
        help="Auto mode: comma-separated device IDs to include (e.g. nvidia_0,cpu_0). Omit to use all.",
    )
    run_parser.add_argument(
        "--precisions",
        default="fp32,fp16",
        help="Auto mode: comma-separated precisions (default: fp32,fp16)",
    )
    run_parser.add_argument(
        "--batch-sizes",
        default="1,4",
        help="Auto mode: comma-separated batch sizes (default: 1,4)",
    )
    run_parser.add_argument(
        "--max-configs-per-device",
        type=int,
        default=6,
        help="Auto mode: max (precision x batch) configs per device (default: 6)",
    )

    # Compare command (uses trackiq comparison module)
    compare_parser = subparsers.add_parser(
        "compare", help="Compare run results against a baseline (trackiq)"
    )
    compare_parser.add_argument(
        "--baseline", "-b", required=True, help="Baseline name or path to baseline JSON"
    )
    compare_parser.add_argument(
        "--current", "-c", required=True, help="Path to current run JSON (metrics)"
    )
    compare_parser.add_argument(
        "--baseline-dir",
        default=".trackiq/baselines",
        help="Directory for baseline files",
    )
    compare_parser.add_argument(
        "--latency-pct", type=float, default=5.0, help="Latency regression threshold %%"
    )
    compare_parser.add_argument(
        "--throughput-pct",
        type=float,
        default=5.0,
        help="Throughput regression threshold %%",
    )
    compare_parser.add_argument(
        "--p99-pct", type=float, default=10.0, help="P99 regression threshold %%"
    )
    compare_parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save --current as new baseline named --baseline",
    )

    # Analyze commands
    analyze_parser = subparsers.add_parser("analyze", help="Analyze performance data")
    analyze_subparsers = analyze_parser.add_subparsers(dest="analyze_type")

    # Analyze latency
    latency_parser = analyze_subparsers.add_parser(
        "latency", help="Analyze percentile latencies"
    )
    latency_parser.add_argument(
        "--csv", required=True, help="CSV file with benchmark data"
    )

    # Analyze logs
    log_parser = analyze_subparsers.add_parser("logs", help="Analyze performance logs")
    log_parser.add_argument("--log", required=True, help="Log file to analyze")
    log_parser.add_argument(
        "--threshold", type=float, default=50.0, help="Latency threshold (ms)"
    )

    # Analyze DNN pipeline
    dnn_parser = analyze_subparsers.add_parser(
        "dnn-pipeline", help="Analyze DNN inference pipeline"
    )
    dnn_parser.add_argument("--csv", help="CSV file with layer timings")
    dnn_parser.add_argument("--profiler", help="Profiler output text file")
    dnn_parser.add_argument("--batch-size", type=int, default=1, help="Batch size used")
    dnn_parser.add_argument(
        "--top-layers", type=int, default=5, help="Number of slowest layers to report"
    )

    # Analyze tegrastats
    tegra_parser = analyze_subparsers.add_parser(
        "tegrastats", help="Analyze Tegrastats output"
    )
    tegra_parser.add_argument("--log", required=True, help="Tegrastats log file")
    tegra_parser.add_argument(
        "--throttle-threshold",
        type=float,
        default=85.0,
        help="Thermal throttling threshold (°C)",
    )

    # Analyze efficiency
    eff_parser = analyze_subparsers.add_parser(
        "efficiency", help="Analyze power efficiency"
    )
    eff_parser.add_argument("--csv", required=True, help="CSV file with benchmark data")

    # Analyze variability
    var_parser = analyze_subparsers.add_parser(
        "variability", help="Analyze latency variability"
    )
    var_parser.add_argument("--csv", required=True, help="CSV file with latency data")
    var_parser.add_argument(
        "--column", default="latency_ms", help="Column name for latency values"
    )

    # Benchmark commands
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench_subparsers = bench_parser.add_subparsers(dest="bench_type")

    # Batch size benchmark
    batch_parser = bench_subparsers.add_parser(
        "batching", help="Batch size trade-off analysis"
    )
    batch_parser.add_argument(
        "--batch-sizes", default="1,4,8,16,32", help="Comma-separated batch sizes"
    )
    batch_parser.add_argument(
        "--images", type=int, default=1000, help="Number of images"
    )

    # LLM benchmark
    llm_parser = bench_subparsers.add_parser("llm", help="LLM inference latency")
    llm_parser.add_argument(
        "--prompt-length", type=int, default=512, help="Prompt token count"
    )
    llm_parser.add_argument(
        "--output-tokens", type=int, default=256, help="Output token count"
    )
    llm_parser.add_argument(
        "--runs", type=int, default=10, help="Number of benchmark runs"
    )

    # Monitor commands
    monitor_parser = subparsers.add_parser("monitor", help="Monitor system metrics")
    monitor_subparsers = monitor_parser.add_subparsers(dest="monitor_type")

    # GPU monitor
    gpu_parser = monitor_subparsers.add_parser("gpu", help="Monitor GPU metrics")
    gpu_parser.add_argument(
        "--duration", type=int, default=300, help="Monitor duration (seconds)"
    )
    gpu_parser.add_argument(
        "--interval", type=int, default=1, help="Sample interval (seconds)"
    )

    # KV cache monitor
    cache_parser = monitor_subparsers.add_parser("kv-cache", help="Monitor KV cache")
    cache_parser.add_argument(
        "--max-length", type=int, default=2048, help="Max sequence length"
    )

    # Report commands
    report_parser = subparsers.add_parser("report", help="Generate performance reports")
    report_subparsers = report_parser.add_subparsers(dest="report_type")

    # HTML report
    html_parser = report_subparsers.add_parser(
        "html", help="Generate interactive HTML report"
    )
    html_parser.add_argument("--csv", help="CSV file with benchmark data")
    html_parser.add_argument(
        "--json",
        "-j",
        metavar="FILE",
        help="JSON file from autoperfpy run --export (benchmark results)",
    )
    html_parser.add_argument(
        "--output",
        "-o",
        default="performance_report.html",
        help="Output HTML file path",
    )
    html_parser.add_argument(
        "--title", default="Performance Analysis Report", help="Report title"
    )
    html_parser.add_argument(
        "--theme", choices=["light", "dark"], default="light", help="Color theme"
    )
    html_parser.add_argument("--author", default="AutoPerfPy", help="Report author")

    # PDF report
    pdf_parser = report_subparsers.add_parser("pdf", help="Generate PDF report")
    pdf_parser.add_argument("--csv", help="CSV file with benchmark data")
    pdf_parser.add_argument(
        "--output", "-o", default="performance_report.pdf", help="Output PDF file path"
    )
    pdf_parser.add_argument(
        "--title", default="Performance Analysis Report", help="Report title"
    )
    pdf_parser.add_argument("--author", default="AutoPerfPy", help="Report author")

    # UI command (Streamlit dashboard)
    ui_parser = subparsers.add_parser(
        "ui", help="Launch interactive Streamlit dashboard"
    )
    ui_parser.add_argument(
        "--data", "-d", help="Path to collector export JSON or CSV file"
    )
    ui_parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8501,
        help="Port to run Streamlit on (default: 8501)",
    )
    ui_parser.add_argument(
        "--host", default="localhost", help="Host to bind to (default: localhost)"
    )
    ui_parser.add_argument(
        "--browser",
        action="store_true",
        default=True,
        help="Open browser automatically (default: True)",
    )
    ui_parser.add_argument(
        "--no-browser", action="store_true", help="Don't open browser automatically"
    )

    return parser


def run_analyze_latency(args, config):
    """Run latency analysis."""
    analyzer = PercentileLatencyAnalyzer(config)
    result = analyzer.analyze(args.csv)

    print("\n📊 Percentile Latency Analysis")
    print("=" * 60)
    for key, metrics in result.metrics.items():
        print(f"\n{key}:")
        print(f"  P99: {metrics.get('p99', 0):.2f}ms")
        print(f"  P95: {metrics.get('p95', 0):.2f}ms")
        print(f"  P50: {metrics.get('p50', 0):.2f}ms")
        print(f"  Mean: {metrics.get('mean', 0):.2f}ms ± {metrics.get('std', 0):.2f}ms")

    return result


def run_analyze_logs(args, config):
    """Run log analysis."""
    analyzer = LogAnalyzer(config)
    result = analyzer.analyze(args.log, args.threshold)

    print("\n📋 Log Analysis")
    print("=" * 60)
    print(f"Threshold: {result.metrics['threshold_ms']}ms")
    print(f"Total events: {result.metrics['total_events']}")
    print(f"Spike events: {result.metrics['spike_events']}")
    print(f"Spike percentage: {result.metrics['spike_percentage']:.2f}%")

    return result


def run_analyze_dnn_pipeline(args, config):
    """Run DNN pipeline analysis."""
    analyzer_config = {
        "top_n_layers": args.top_layers,
    }
    analyzer = DNNPipelineAnalyzer(config=analyzer_config)

    if args.csv:
        result = analyzer.analyze_layer_csv(args.csv, batch_size=args.batch_size)
    elif args.profiler:
        with open(args.profiler, "r") as f:
            content = f.read()
        result = analyzer.analyze_profiler_output(content)
    else:
        print("❌ Error: Either --csv or --profiler must be specified")
        return None

    print("\n🧠 DNN Pipeline Analysis")
    print("=" * 60)
    metrics = result.metrics

    print(f"\nSource: {metrics.get('source', 'unknown')}")
    print(f"Batch Size: {metrics.get('batch_size', 1)}")
    print(f"Number of Layers: {metrics.get('num_layers', 0)}")

    timing = metrics.get("timing", {})
    print("\n⏱️  Timing:")
    print(
        f"  Total Time: {timing.get('total_time_ms', timing.get('avg_total_ms', 0)):.2f}ms"
    )
    print(f"  GPU Time: {timing.get('gpu_time_ms', 0):.2f}ms")
    print(f"  DLA Time: {timing.get('dla_time_ms', 0):.2f}ms")

    device_split = metrics.get("device_split", {})
    print("\n📊 Device Split:")
    print(f"  GPU: {device_split.get('gpu_percentage', 0):.1f}%")
    print(f"  DLA: {device_split.get('dla_percentage', 0):.1f}%")

    throughput = metrics.get(
        "throughput_fps", metrics.get("throughput", {}).get("avg_fps", 0)
    )
    print(f"\n🚀 Throughput: {throughput:.1f} FPS")

    slowest = metrics.get("slowest_layers", [])
    if slowest:
        print(f"\n🐢 Slowest Layers:")
        for layer in slowest[:5]:
            name = layer.get("name", "unknown")
            time_ms = layer.get("time_ms", layer.get("avg_time_ms", 0))
            device = layer.get("device", "GPU")
            print(f"  {name}: {time_ms:.2f}ms ({device})")

    recommendations = metrics.get("recommendations", [])
    if recommendations:
        print("\n💡 Recommendations:")
        for rec in recommendations:
            print(f"  • {rec}")

    return result


def run_analyze_tegrastats(args, _config):
    """Run tegrastats analysis."""
    analyzer = TegrastatsAnalyzer(
        config={"throttle_temp_c": getattr(args, "throttle_threshold", 85.0)}
    )
    result = analyzer.analyze(args.log)

    print("\n📊 Tegrastats Analysis")
    print("=" * 60)
    metrics = result.metrics

    print(f"\nSamples Analyzed: {metrics.get('num_samples', 0)}")

    # CPU metrics
    cpu = metrics.get("cpu", {})
    print(f"\n🖥️  CPU:")
    print(f"  Average Utilization: {cpu.get('avg_utilization', 0):.1f}%")
    print(f"  Max Utilization: {cpu.get('max_utilization', 0):.1f}%")

    # GPU metrics
    gpu = metrics.get("gpu", {})
    print("\n🎮 GPU:")
    print(f"  Average Utilization: {gpu.get('avg_utilization', 0):.1f}%")
    print(f"  Max Utilization: {gpu.get('max_utilization', 0):.1f}%")
    print(f"  Average Frequency: {gpu.get('avg_frequency_mhz', 0):.0f} MHz")

    # Memory metrics
    memory = metrics.get("memory", {})
    print(f"\n💾 Memory:")
    print(f"  Average Used: {memory.get('avg_used_mb', 0):.0f} MB")
    print(f"  Max Used: {memory.get('max_used_mb', 0):.0f} MB")

    # Thermal metrics
    thermal = metrics.get("thermal", {})
    print("\n🌡️  Thermal:")
    print(f"  Average Temperature: {thermal.get('avg_temperature', 0):.1f}°C")
    print(f"  Max Temperature: {thermal.get('max_temperature', 0):.1f}°C")
    print(f"  Throttling Events: {thermal.get('throttle_events', 0)}")

    # Health status
    health = metrics.get("health", {})
    status = health.get("status", "unknown")
    status_emoji = "✅" if status == "healthy" else "⚠️" if status == "warning" else "❌"
    print(f"\n{status_emoji} Health Status: {status.upper()}")

    warnings = health.get("warnings", [])
    if warnings:
        print("  Warnings:")
        for warning in warnings:
            print(f"    • {warning}")

    return result


def run_analyze_efficiency(args, config):
    """Run efficiency analysis."""
    analyzer = EfficiencyAnalyzer(config)
    result = analyzer.analyze(args.csv)

    print("\n⚡ Efficiency Analysis")
    print("=" * 60)
    metrics = result.metrics

    for workload, data in metrics.items():
        if isinstance(data, dict):
            print(f"\n{workload}:")
            print(f"  Performance/Watt: {data.get('perf_per_watt', 0):.2f} infer/s/W")
            print(f"  Energy/Inference: {data.get('energy_per_inference_j', 0):.4f} J")
            print(f"  Throughput: {data.get('throughput_fps', 0):.1f} FPS")
            print(f"  Average Power: {data.get('avg_power_w', 0):.1f} W")

    return result


def run_analyze_variability(args, config):
    """Run variability analysis."""
    analyzer = VariabilityAnalyzer(config)
    result = analyzer.analyze(args.csv, latency_column=args.column)

    print("\n📈 Variability Analysis")
    print("=" * 60)
    metrics = result.metrics

    print(f"\nCoefficient of Variation: {metrics.get('cv_percent', 0):.2f}%")
    print(f"Jitter (Std Dev): {metrics.get('jitter_ms', 0):.2f}ms")
    print(f"IQR: {metrics.get('iqr_ms', 0):.2f}ms")
    print(f"Outliers: {metrics.get('outlier_count', 0)}")
    print(f"Consistency Rating: {metrics.get('consistency_rating', 'unknown')}")

    print("\n📊 Percentiles:")
    print(f"  P50: {metrics.get('p50_ms', 0):.2f}ms")
    print(f"  P95: {metrics.get('p95_ms', 0):.2f}ms")
    print(f"  P99: {metrics.get('p99_ms', 0):.2f}ms")

    return result


def run_benchmark_batching(args, config):
    """Run batching trade-off benchmark."""
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    benchmark = BatchingTradeoffBenchmark(config)
    results = benchmark.run(batch_sizes=batch_sizes, num_images=args.images)

    print("\n⚡ Batching Trade-off Analysis")
    print("=" * 60)
    print(f"{'Batch':<10} {'Latency (ms)':<15} {'Throughput (img/s)':<20}")
    print("-" * 60)
    for i, batch in enumerate(results["batch_size"]):
        latency = results["latency_ms"][i]
        throughput = results["throughput_img_per_sec"][i]
        print(f"{batch:<10} {latency:<15.2f} {throughput:<20.2f}")

    return results


def run_benchmark_llm(args, config):
    """Run LLM latency benchmark."""
    benchmark = LLMLatencyBenchmark(config)
    results = benchmark.run(
        prompt_tokens=args.prompt_length,
        output_tokens=args.output_tokens,
        num_runs=args.runs,
    )

    print("\n🤖 LLM Latency Benchmark")
    print("=" * 60)
    print("TTFT (Time-to-First-Token):")
    print(f"  P50: {results['ttft_p50']:.1f}ms")
    print(f"  P95: {results['ttft_p95']:.1f}ms")
    print(f"  P99: {results['ttft_p99']:.1f}ms")
    print("\nTime-per-Token (Decode):")
    print(f"  P50: {results['tpt_p50']:.1f}ms")
    print(f"  P95: {results['tpt_p95']:.1f}ms")
    print(f"  P99: {results['tpt_p99']:.1f}ms")
    print(f"\nThroughput: {results['throughput_tokens_per_sec']:.1f} tokens/sec")

    return results


def run_monitor_gpu(args, config):
    """Run GPU monitoring."""
    monitor = GPUMemoryMonitor(config)

    print(f"\n📊 Monitoring GPU for {args.duration} seconds...")
    monitor.start()

    try:
        remaining = args.duration
        while remaining > 0:
            metrics = monitor.get_metrics()
            if metrics:
                latest = metrics[-1]
                print(
                    f"GPU Memory: {latest['gpu_memory_used_mb']:.0f}MB / {latest['gpu_memory_total_mb']:.0f}MB "
                    f"({latest['gpu_memory_percent']:.1f}%), Utilization: {latest['gpu_utilization_percent']:.1f}%"
                )
            remaining -= args.interval
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")
    finally:
        monitor.stop()

    summary = monitor.get_summary()
    if summary:
        print("\n📈 Summary:")
        print(f"  Avg Memory: {summary['avg_memory_mb']:.0f}MB")
        print(f"  Max Memory: {summary['max_memory_mb']:.0f}MB")
        print(f"  Avg Utilization: {summary['avg_utilization_percent']:.1f}%")

    return summary


def run_report_html(args, config):
    """Generate HTML report."""
    import pandas as pd

    report = HTMLReportGenerator(
        title=args.title,
        author=args.author,
        theme=args.theme,
    )

    viz = PerformanceVisualizer()

    # Add metadata
    data_source = "Sample Data"
    if args.csv:
        data_source = args.csv
    elif args.json:
        data_source = args.json
    report.add_metadata("Data Source", data_source)

    if args.json:
        # Load benchmark export JSON (from autoperfpy run --export)
        with open(args.json, encoding="utf-8") as f:
            data = json.load(f)
        report.add_metadata("Collector", data.get("collector_name", "-"))
        if data.get("profile"):
            report.add_metadata("Profile", data["profile"])
        summary = data.get("summary", {})
        # Summary items from export summary (sample_count can be top-level in export)
        sample_count = (
            data.get("sample_count")
            or summary.get("sample_count")
            or len(data.get("samples", []))
        )
        report.add_summary_item("Samples", sample_count, "", "neutral")
        lat = summary.get("latency", {})
        if lat:
            report.add_summary_item(
                "P99 Latency", f"{lat.get('p99_ms', 0):.2f}", "ms", "neutral"
            )
            report.add_summary_item(
                "P50 Latency", f"{lat.get('p50_ms', 0):.2f}", "ms", "neutral"
            )
            report.add_summary_item(
                "Mean Latency", f"{lat.get('mean_ms', 0):.2f}", "ms", "neutral"
            )
        thr = summary.get("throughput", {})
        if thr:
            report.add_summary_item(
                "Mean Throughput", f"{thr.get('mean_fps', 0):.1f}", "FPS", "neutral"
            )
        pwr = summary.get("power", {})
        if pwr and pwr.get("mean_w") is not None:
            report.add_summary_item(
                "Mean Power", f"{pwr.get('mean_w', 0):.1f}", "W", "neutral"
            )
        if data.get("validation"):
            v = data["validation"]
            status = "good" if v.get("overall_pass") else "critical"
            report.add_summary_item(
                "Run Status", "PASS" if v.get("overall_pass") else "FAIL", "", status
            )
        # Latency percentiles figure from samples if available
        samples = data.get("samples", [])
        if samples:
            latencies = []
            for s in samples:
                m = (
                    s.get("metrics", s)
                    if isinstance(s, dict)
                    else getattr(s, "metrics", {})
                )
                if isinstance(m, dict) and "latency_ms" in m:
                    latencies.append(m["latency_ms"])
            if latencies:
                report.add_section("Latency Analysis", "From benchmark samples")
                by_run = {
                    "Run": {
                        "P50": float(np.percentile(latencies, 50)),
                        "P95": float(np.percentile(latencies, 95)),
                        "P99": float(np.percentile(latencies, 99)),
                    }
                }
                fig = viz.plot_latency_percentiles(by_run)
                report.add_figure(fig, "Latency Percentiles", "Latency Analysis")
        output_path = report.generate_html(args.output)
        print(f"\n[OK] HTML report generated: {output_path}")
        return {"output_path": output_path}

    if args.csv:
        # Load and analyze data
        df = pd.read_csv(args.csv)

        # Add summary items based on available columns
        if "latency_ms" in df.columns:
            report.add_summary_item("Samples", len(df), "", "neutral")
            report.add_summary_item(
                "Mean Latency", f"{df['latency_ms'].mean():.2f}", "ms", "neutral"
            )
            report.add_summary_item(
                "P99 Latency", f"{df['latency_ms'].quantile(0.99):.2f}", "ms", "neutral"
            )

            cv = df["latency_ms"].std() / df["latency_ms"].mean() * 100
            status = "good" if cv < 10 else "warning" if cv < 20 else "critical"
            report.add_summary_item("CV", f"{cv:.1f}", "%", status)

        # Generate visualizations based on available data
        if "workload" in df.columns and "latency_ms" in df.columns:
            report.add_section(
                "Latency Analysis", "Percentile latency comparisons across workloads"
            )

            latencies_by_workload = {}
            for workload in df["workload"].unique():
                wdf = df[df["workload"] == workload]["latency_ms"]
                latencies_by_workload[workload] = {
                    "P50": wdf.quantile(0.5),
                    "P95": wdf.quantile(0.95),
                    "P99": wdf.quantile(0.99),
                }
            fig = viz.plot_latency_percentiles(latencies_by_workload)
            report.add_figure(
                fig, "Latency Percentiles by Workload", "Latency Analysis"
            )

            # Distribution comparison
            data_dict = {
                w: df[df["workload"] == w]["latency_ms"].tolist()
                for w in df["workload"].unique()
            }
            fig = viz.plot_distribution(data_dict, "Latency Distribution Comparison")
            report.add_figure(fig, "Latency Distribution", "Latency Analysis")

        if "batch_size" in df.columns and "latency_ms" in df.columns:
            report.add_section("Batch Analysis", "Performance vs batch size")

            batch_df = (
                df.groupby("batch_size")
                .agg(
                    {
                        "latency_ms": "mean",
                    }
                )
                .reset_index()
            )

            # Calculate throughput if not present
            if "throughput" not in df.columns:
                batch_df["throughput"] = (
                    batch_df["batch_size"] * 1000 / batch_df["latency_ms"]
                )
            else:
                batch_df["throughput"] = (
                    df.groupby("batch_size")["throughput"].mean().values
                )

            fig = viz.plot_latency_throughput_tradeoff(
                batch_df["batch_size"].tolist(),
                batch_df["latency_ms"].tolist(),
                batch_df["throughput"].tolist(),
            )
            report.add_figure(fig, "Latency vs Throughput Trade-off", "Batch Analysis")

        if "power_w" in df.columns:
            report.add_section(
                "Power Analysis", "Power consumption and efficiency metrics"
            )

            if "workload" in df.columns:
                workloads = df["workload"].unique().tolist()
                power_values = [
                    df[df["workload"] == w]["power_w"].mean() for w in workloads
                ]
                if "latency_ms" in df.columns:
                    perf_values = [
                        1000 / df[df["workload"] == w]["latency_ms"].mean()
                        for w in workloads
                    ]
                    fig = viz.plot_power_vs_performance(
                        workloads, power_values, perf_values
                    )
                    report.add_figure(fig, "Power vs Performance", "Power Analysis")

        # Add data table
        if len(df) > 0:
            sample_df = df.head(20)
            headers = sample_df.columns.tolist()
            rows = sample_df.values.tolist()
            # Format numeric values
            rows = [
                [f"{v:.2f}" if isinstance(v, float) else v for v in row] for row in rows
            ]
            report.add_table(
                "Sample Data (First 20 rows)", headers, rows, "Data Overview"
            )
            report.add_section("Data Overview", "Raw data samples")

    else:
        # No data: report only metadata, no static/demo graphs
        report.add_summary_item("Status", "No data file provided", "", "warning")
        report.add_summary_item(
            "Note",
            "Provide --csv with benchmark data for dynamic graphs",
            "",
            "neutral",
        )
        report.add_section(
            "No Data",
            "Run a benchmark and pass --csv to generate visualizations from current data.",
        )

    output_path = report.generate_html(args.output)
    print(f"\n[OK] HTML report generated: {output_path}")
    return {"output_path": output_path}


def run_ui(args):
    """Launch Streamlit dashboard."""
    import subprocess
    from pathlib import Path

    # Get path to streamlit_app.py
    ui_module = Path(__file__).parent / "ui" / "streamlit_app.py"

    if not ui_module.exists():
        print(f"Error: Streamlit app not found at {ui_module}", file=sys.stderr)
        return 1

    # Build streamlit command
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ui_module),
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
    ]

    # Handle browser option
    if args.no_browser:
        cmd.extend(["--server.headless", "true"])

    # Pass data file if provided
    if args.data:
        cmd.extend(["--", "--data", args.data])

    print(f"Launching AutoPerfPy Dashboard...")
    print(f"URL: http://{args.host}:{args.port}")

    if args.data:
        print(f"Data file: {args.data}")

    print("\nPress Ctrl+C to stop the server\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nDashboard stopped")
    except subprocess.CalledProcessError as e:
        print(f"Error launching Streamlit: {e}", file=sys.stderr)
        print("\nMake sure Streamlit is installed: pip install streamlit plotly pandas")
        return 1
    except FileNotFoundError:
        print(
            "Error: Streamlit not found. Install with: pip install streamlit",
            file=sys.stderr,
        )
        return 1

    return 0


def run_report_pdf(args, config):
    """Generate PDF report."""
    import pandas as pd

    report = PDFReportGenerator(
        title=args.title,
        author=args.author,
    )

    viz = PerformanceVisualizer()

    # Add metadata
    report.add_metadata("Data Source", args.csv if args.csv else "Sample Data")

    if args.csv:
        df = pd.read_csv(args.csv)
        report.add_metadata("Total Samples", str(len(df)))

        # Generate visualizations based on available data
        if "workload" in df.columns and "latency_ms" in df.columns:
            latencies_by_workload = {}
            for workload in df["workload"].unique():
                wdf = df[df["workload"] == workload]["latency_ms"]
                latencies_by_workload[workload] = {
                    "P50": wdf.quantile(0.5),
                    "P95": wdf.quantile(0.95),
                    "P99": wdf.quantile(0.99),
                }
            fig = viz.plot_latency_percentiles(latencies_by_workload)
            report.add_figure(fig, "Latency Percentiles by Workload")

        if "batch_size" in df.columns and "latency_ms" in df.columns:
            batch_df = (
                df.groupby("batch_size").agg({"latency_ms": "mean"}).reset_index()
            )
            if "throughput" not in df.columns:
                batch_df["throughput"] = (
                    batch_df["batch_size"] * 1000 / batch_df["latency_ms"]
                )
            else:
                batch_df["throughput"] = (
                    df.groupby("batch_size")["throughput"].mean().values
                )

            fig = viz.plot_latency_throughput_tradeoff(
                batch_df["batch_size"].tolist(),
                batch_df["latency_ms"].tolist(),
                batch_df["throughput"].tolist(),
            )
            report.add_figure(fig, "Latency vs Throughput Trade-off")

    else:
        report.add_metadata(
            "Status", "No data file provided. Use --csv for dynamic report content."
        )

    output_path = report.generate_pdf(args.output)
    print(f"\n[OK] PDF report generated: {output_path}")
    return {"output_path": output_path}


def _flatten_summary_for_compare(summary: dict) -> dict:
    """Flatten export summary to metric_name -> float for trackiq compare."""
    flat = {}
    for section, data in summary.items():
        if section in ("sample_count", "warmup_samples", "duration_seconds"):
            continue
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (int, float)):
                    flat[f"{section}_{k}"] = float(v)
    return flat


def run_compare(args):
    """Compare current run against baseline using trackiq comparison module."""
    baseline_dir = (
        getattr(args, "baseline_dir", ".trackiq/baselines") or ".trackiq/baselines"
    )
    detector = RegressionDetector(baseline_dir=baseline_dir)
    thresholds = RegressionThreshold(
        latency_percent=getattr(args, "latency_pct", 5.0),
        throughput_percent=getattr(args, "throughput_pct", 5.0),
        p99_percent=getattr(args, "p99_pct", 10.0),
    )

    with open(args.current, "r", encoding="utf-8") as f:
        current_data = json.load(f)
    summary = current_data.get("summary", current_data.get("metrics", current_data))
    current_metrics = (
        _flatten_summary_for_compare(summary) if isinstance(summary, dict) else {}
    )

    if getattr(args, "save_baseline", False):
        detector.save_baseline(args.baseline, current_metrics)
        print(f"Baseline '{args.baseline}' saved from {args.current}")
        return 0

    try:
        detector.load_baseline(args.baseline)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(
            "Save a baseline first: autoperfpy compare --save-baseline --baseline NAME --current run.json",
            file=sys.stderr,
        )
        return 1

    report = detector.generate_report(args.baseline, current_metrics, thresholds)
    print(report)
    result = detector.detect_regressions(args.baseline, current_metrics, thresholds)
    return 1 if result.get("has_regressions") else 0


def run_profiles(args):
    """Handle profiles command."""
    if args.info:
        # Show detailed info for a specific profile
        try:
            profile = get_profile(args.info)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        print(f"\nProfile: {profile.name}")
        print("=" * 60)
        print(f"Description: {profile.description}")
        print("\nLatency Requirements:")
        print(f"  Threshold (P99): {profile.latency_threshold_ms}ms")
        print(f"  Target: {profile.latency_target_ms}ms")
        print(f"  Percentiles: {profile.latency_percentiles}")
        print(f"\nThroughput Requirements:")
        print(f"  Minimum: {profile.throughput_min_fps} FPS")
        print(f"  Target: {profile.throughput_target_fps} FPS")
        print("\nConstraints:")
        print(
            f"  Power Budget: {profile.power_budget_w}W"
            if profile.power_budget_w
            else "  Power Budget: None"
        )
        print(f"  Thermal Limit: {profile.thermal_limit_c}C")
        print(
            f"  Memory Limit: {profile.memory_limit_mb}MB"
            if profile.memory_limit_mb
            else "  Memory Limit: None"
        )
        print("\nBenchmark Settings:")
        print(f"  Batch Sizes: {profile.batch_sizes}")
        print(f"  Warmup Iterations: {profile.warmup_iterations}")
        print(f"  Test Iterations: {profile.test_iterations}")
        print(f"  Runs: {profile.num_runs}")
        print(f"\nMonitoring Settings:")
        print(f"  Sample Interval: {profile.sample_interval_ms}ms")
        print(f"  Duration: {profile.duration_seconds}s")
        print("\nSupported Collectors:")
        for c in profile.supported_collectors:
            print(f"  - {c.value}")
        print(f"\nTags: {', '.join(profile.tags)}")
        return 0

    # Default: list all profiles
    print("\nAvailable Performance Profiles")
    print("=" * 60)
    info = get_profile_info()
    for name, details in info.items():
        print(f"\n{name}")
        print(f"  {details['description']}")
        print(
            f"  Latency threshold: {details['latency_threshold_ms']}ms | "
            f"Throughput target: {details['throughput_target_fps']} FPS"
        )
        power = f"{details['power_budget_w']}W" if details["power_budget_w"] else "None"
        print(f"  Power budget: {power} | Tags: {', '.join(details['tags'][:3])}")
    print(f"\nUse 'autoperfpy profiles --info <name>' for detailed information.")
    return 0


def run_with_profile(args, _config):
    """Run performance test with a profile."""
    # Get the profile
    profile_name = args.profile
    try:
        profile = get_profile(profile_name)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return None

    # Map collector string to CollectorType
    collector_map = {
        "synthetic": CollectorType.SYNTHETIC,
        "nvml": CollectorType.NVML,
        "tegrastats": CollectorType.TEGRASTATS,
        "psutil": CollectorType.PSUTIL,
    }
    collector_type = collector_map.get(args.collector, CollectorType.SYNTHETIC)

    # Validate collector compatibility
    try:
        validate_profile_collector(profile, collector_type)
    except ProfileValidationError as e:
        print(f"Error: {e}", file=sys.stderr)
        return None

    if args.validate_only:
        print(
            f"Profile '{profile_name}' validated successfully with collector '{args.collector}'"
        )
        return {
            "status": "validated",
            "profile": profile_name,
            "collector": args.collector,
        }

    # Apply CLI overrides
    duration = args.duration if args.duration else profile.duration_seconds
    iterations = args.iterations if args.iterations else profile.test_iterations
    warmup = args.warmup if args.warmup else profile.warmup_iterations

    if not args.quiet:
        print(f"\nRunning with profile: {profile_name}")
        print("=" * 60)
        print(f"Collector: {args.collector}")
        print(f"Duration: {duration}s")
        print(f"Iterations: {iterations}")
        print(f"Warmup: {warmup}")
        print(f"Latency Threshold: {profile.latency_threshold_ms}ms")
        print("=" * 60)

    # Create collector based on type (no synthetic fallback: fail explicitly if unavailable)
    collector = None
    if collector_type == CollectorType.NVML:
        try:
            from autoperfpy.collectors import NVMLCollector
        except ImportError as e:
            print(f"Error: NVML collector requires nvidia-ml-py. {e}", file=sys.stderr)
            raise DependencyError(
                "NVML collector requires nvidia-ml-py. Install with: pip install nvidia-ml-py"
            ) from e
        from trackiq.platform import get_memory_metrics

        if get_memory_metrics() is None:
            raise HardwareNotFoundError(
                "No NVIDIA GPU or nvidia-smi not available. Use --collector synthetic for simulation."
            )
        device_index = 0
        if getattr(args, "device", None) is not None:
            try:
                device_index = int(args.device)
            except ValueError:
                pass  # device might be a name; NVML uses index
        collector = NVMLCollector(
            device_index=device_index, config=profile.get_synthetic_config() or {}
        )
    elif collector_type == CollectorType.PSUTIL:
        try:
            from autoperfpy.collectors import PsutilCollector
        except ImportError as e:
            print(f"Error: Psutil collector requires psutil. {e}", file=sys.stderr)
            raise DependencyError(
                "Psutil collector requires psutil. Install with: pip install psutil"
            ) from e
        collector = PsutilCollector(config=profile.get_synthetic_config() or {})
    elif collector_type == CollectorType.TEGRASTATS:
        try:
            from autoperfpy.collectors import TegrastatsCollector
        except ImportError as e:
            print(f"Error: Tegrastats collector not available. {e}", file=sys.stderr)
            raise DependencyError(
                "Tegrastats collector requires Jetson/tegrastats. Use --collector synthetic on non-Jetson."
            ) from e
        collector = TegrastatsCollector(config=profile.get_synthetic_config() or {})
    else:
        # Synthetic
        pass

    if collector_type == CollectorType.SYNTHETIC:
        collector_config = profile.get_synthetic_config()
        collector_config["warmup_samples"] = warmup
        if args.batch_size:
            collector_config["batch_sizes"] = [args.batch_size]
        collector = SyntheticCollector(config=collector_config)

    # Optional: pass device/precision into run context (for app-specific use)
    _device = getattr(args, "device", None)
    _precision = getattr(args, "precision", "fp32")
    if not args.quiet and (_device is not None or _precision != "fp32"):
        print(f"Device: {_device or 'default'} | Precision: {_precision}")

    # Run collection
    collector.start()
    sample_count = 0
    sample_interval = profile.sample_interval_ms / 1000.0  # Convert to seconds

    start_time = time.time()
    try:
        while time.time() - start_time < duration and sample_count < iterations:
            timestamp = time.time()
            metrics = collector.sample(timestamp)

            if not args.quiet and metrics:
                warmup_marker = "[WARMUP]" if metrics.get("is_warmup") else ""
                latency = metrics.get("latency_ms", 0)
                gpu = metrics.get("gpu_percent", 0)
                power = metrics.get("power_w", 0)
                print(
                    f"[{sample_count:4d}] "
                    f"Latency: {latency:6.2f}ms | "
                    f"GPU: {gpu:5.1f}% | "
                    f"Power: {power:5.1f}W "
                    f"{warmup_marker}"
                )

            sample_count += 1
            time.sleep(sample_interval)

    except KeyboardInterrupt:
        print("\nCollection interrupted by user")

    collector.stop()

    # Export and analyze results
    export = collector.export()
    summary = export.summary

    # Check against profile thresholds
    latency_p99 = summary.get("latency", {}).get("p99_ms", 0)
    throughput = summary.get("throughput", {}).get("mean_fps", 0)
    power_avg = summary.get("power", {}).get("mean_w", 0)

    latency_pass = latency_p99 <= profile.latency_threshold_ms
    throughput_pass = throughput >= profile.throughput_min_fps
    power_pass = profile.power_budget_w is None or power_avg <= profile.power_budget_w

    print(f"\n{'=' * 60}")
    print("Results Summary")
    print("=" * 60)
    print(f"Samples Collected: {summary.get('sample_count', 0)}")

    print("\nLatency (excluding warmup):")
    latency_status = "PASS" if latency_pass else "FAIL"
    print(
        f"  P99: {latency_p99:.2f}ms (threshold: {profile.latency_threshold_ms}ms) [{latency_status}]"
    )
    print(f"  P95: {summary.get('latency', {}).get('p95_ms', 0):.2f}ms")
    print(f"  P50: {summary.get('latency', {}).get('p50_ms', 0):.2f}ms")
    print(f"  Mean: {summary.get('latency', {}).get('mean_ms', 0):.2f}ms")

    print("\nThroughput:")
    throughput_status = "PASS" if throughput_pass else "FAIL"
    print(
        f"  Mean: {throughput:.1f} FPS (min: {profile.throughput_min_fps} FPS) [{throughput_status}]"
    )

    print(f"\nPower:")
    if profile.power_budget_w:
        power_status = "PASS" if power_pass else "FAIL"
        print(
            f"  Mean: {power_avg:.1f}W (budget: {profile.power_budget_w}W) [{power_status}]"
        )
    else:
        print(f"  Mean: {power_avg:.1f}W (no budget constraint)")

    print("\nResource Utilization:")
    print(f"  GPU: {summary.get('gpu', {}).get('mean_percent', 0):.1f}% avg")
    print(f"  CPU: {summary.get('cpu', {}).get('mean_percent', 0):.1f}% avg")
    print(f"  Memory: {summary.get('memory', {}).get('mean_mb', 0):.0f}MB avg")

    overall_pass = latency_pass and throughput_pass and power_pass
    status_emoji = "PASS" if overall_pass else "FAIL"
    print(f"\nOverall Status: [{status_emoji}]")

    # Save export if requested
    if args.export:
        export_data = export.to_dict()
        export_data["profile"] = profile.name
        export_data["validation"] = {
            "latency_pass": latency_pass,
            "throughput_pass": throughput_pass,
            "power_pass": power_pass,
            "overall_pass": overall_pass,
        }
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)
        print(f"\nResults exported to: {args.export}")

    return export


def _resolve_device(device_id: str) -> Optional[DeviceProfile]:
    """Resolve device ID (e.g. nvidia_0, cpu_0, 0) to a DeviceProfile."""
    return resolve_device(device_id)


def run_devices_list(args) -> int:
    """List all detected devices (for run --auto / --device)."""
    devices = get_all_devices()
    if not devices:
        print("No devices detected.", file=sys.stderr)
        print(
            "Run on CPU with: autoperfpy run --manual --device cpu_0",
            file=sys.stderr,
        )
        return 1
    print("Detected devices (use with: autoperfpy run --auto or --device <id>):")
    print("=" * 72)
    for d in devices:
        name = (d.device_name or d.device_id).strip()
        parts = [d.device_id, d.device_type, name]
        if d.gpu_model:
            parts.append(f"gpu={d.gpu_model}")
        if d.soc:
            parts.append(f"soc={d.soc}")
        if d.cpu_model:
            parts.append(f"cpu={d.cpu_model}")
        print("  " + "  ".join(parts))
    print("=" * 72)
    print(f"Total: {len(devices)} device(s)")
    return 0


def run_auto_benchmarks_cli(args) -> int:
    """Run automatic benchmarks on all detected devices and configs."""
    device_ids_filter = None
    if getattr(args, "devices", None):
        device_ids_filter = [s.strip() for s in args.devices.split(",") if s.strip()]
    precisions = [
        s.strip()
        for s in getattr(args, "precisions", ",".join(PRECISIONS)).split(",")
        if s.strip()
    ]
    batch_sizes = []
    for s in getattr(
        args, "batch_sizes", ",".join(map(str, DEFAULT_BATCH_SIZES))
    ).split(","):
        try:
            batch_sizes.append(int(s.strip()))
        except ValueError:
            pass
    if not batch_sizes:
        batch_sizes = list(DEFAULT_BATCH_SIZES)
    pairs = get_devices_and_configs_auto(
        device_ids_filter=device_ids_filter,
        precisions=precisions,
        batch_sizes=batch_sizes,
        max_configs_per_device=getattr(args, "max_configs_per_device", 6),
    )
    if not pairs:
        print("No (device, config) pairs to run.", file=sys.stderr)
        return 1
    duration = float(getattr(args, "duration", None) or 10)
    if not args.quiet:
        print("Auto mode: running benchmarks on all detected devices and configs")
        print("=" * 60)
        device_ids = list(dict.fromkeys(p[0].device_id for p in pairs))
        print(f"Devices: {device_ids}")
        print(f"Runs: {len(pairs)} (duration {duration}s each)")
        print("=" * 60)
    results = run_auto_benchmarks(
        pairs,
        duration_seconds=duration,
        sample_interval_seconds=0.2,
        quiet=args.quiet,
        progress_callback=(
            None
            if args.quiet
            else lambda i, t, d, c: print(
                f"[{i}/{t}] {d.device_id} {c.precision} bs{c.batch_size}..."
            )
        ),
    )
    if args.export:
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Exported {len(results)} runs to {args.export}")
    for r in results:
        if "error" in r:
            print(f"[FAIL] {r.get('run_label', '?')}: {r['error']}", file=sys.stderr)
        elif not args.quiet:
            s = r.get("summary", {})
            lat = s.get("latency", {}).get("p99_ms", "N/A")
            thr = s.get("throughput", {}).get("mean_fps", "N/A")
            print(f"[OK] {r.get('run_label', '?')} P99={lat}ms Throughput={thr} FPS")
    return 0


def run_manual_single(args):
    """Run a single benchmark with manually selected device and config."""
    device_id = getattr(args, "device", None) or "cpu_0"
    device = _resolve_device(device_id)
    if device is None:
        print("No device found. Use --device nvidia_0, cpu_0, or 0.", file=sys.stderr)
        return None
    config = InferenceConfig(
        precision=getattr(args, "precision", None) or PRECISION_FP32,
        batch_size=getattr(args, "batch_size", None) or 1,
        accelerator=device.device_id,
        streams=1,
        warmup_runs=getattr(args, "warmup", None) or DEFAULT_WARMUP_RUNS,
        iterations=getattr(args, "iterations", None) or DEFAULT_ITERATIONS,
    )
    duration = float(getattr(args, "duration", None) or 10)
    if not args.quiet:
        print("Manual mode: single run")
        print("=" * 60)
        print(f"Device: {device.device_name} ({device.device_id})")
        print(f"Precision: {config.precision}  Batch: {config.batch_size}")
        print("=" * 60)
    result = run_single_benchmark(
        device,
        config,
        duration_seconds=duration,
        sample_interval_seconds=0.2,
        quiet=args.quiet,
    )
    if args.export:
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\n[OK] Exported to {args.export}")
    return result


def main():
    """Main CLI entry point."""
    parser = setup_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Load configuration
    config = ConfigManager.load_or_default(args.config)

    # Backward compat: run with profile from env if no --profile and no manual/device
    if (
        args.command == "run"
        and args.profile is None
        and getattr(args, "device", None) is None
    ):
        env_profile = os.environ.get("AUTOPERFPY_PROFILE")
        if env_profile:
            args.profile = env_profile

    # Route to appropriate handler
    result = None
    try:
        if args.command == "profiles":
            return run_profiles(args)
        elif args.command == "devices":
            return run_devices_list(args)
        elif args.command == "run":
            if args.profile is not None:
                result = run_with_profile(args, config)
                if result is None:
                    return 1
            elif (
                getattr(args, "manual", False)
                or getattr(args, "device", None) is not None
            ):
                result = run_manual_single(args)
                if result is None:
                    return 1
            else:
                return run_auto_benchmarks_cli(args)
        elif args.command == "analyze":
            if args.analyze_type == "latency":
                result = run_analyze_latency(args, config)
            elif args.analyze_type == "logs":
                result = run_analyze_logs(args, config)
            elif args.analyze_type == "dnn-pipeline":
                result = run_analyze_dnn_pipeline(args, config)
            elif args.analyze_type == "tegrastats":
                result = run_analyze_tegrastats(args, config)
            elif args.analyze_type == "efficiency":
                result = run_analyze_efficiency(args, config)
            elif args.analyze_type == "variability":
                result = run_analyze_variability(args, config)
        elif args.command == "benchmark":
            if args.bench_type == "batching":
                result = run_benchmark_batching(args, config)
            elif args.bench_type == "llm":
                result = run_benchmark_llm(args, config)
        elif args.command == "monitor":
            if args.monitor_type == "gpu":
                result = run_monitor_gpu(args, config)
        elif args.command == "report":
            if args.report_type == "html":
                result = run_report_html(args, config)
            elif args.report_type == "pdf":
                result = run_report_pdf(args, config)
        elif args.command == "ui":
            return run_ui(args)
        elif args.command == "compare":
            return run_compare(args)
    except (HardwareNotFoundError, DependencyError, ProfileValidationError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1

    # Save output if requested (report html/pdf already write to args.output; do not overwrite)
    if args.output and result and getattr(args, "command", None) != "report":
        with open(args.output, "w", encoding="utf-8") as f:
            if hasattr(result, "to_dict"):
                json.dump(result.to_dict(), f, indent=2)
            else:
                json.dump(result, f, indent=2)
        print(f"\n[OK] Results saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
