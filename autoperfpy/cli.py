"""Command-line interface for AutoPerfPy."""

import argparse
import sys

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
import json


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
        """,
    )

    parser.add_argument("--config", help="Path to configuration file (YAML/JSON)")
    parser.add_argument("--output", help="Output file for results (JSON)")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Analyze commands
    analyze_parser = subparsers.add_parser("analyze", help="Analyze performance data")
    analyze_subparsers = analyze_parser.add_subparsers(dest="analyze_type")

    # Analyze latency
    latency_parser = analyze_subparsers.add_parser("latency", help="Analyze percentile latencies")
    latency_parser.add_argument("--csv", required=True, help="CSV file with benchmark data")

    # Analyze logs
    log_parser = analyze_subparsers.add_parser("logs", help="Analyze performance logs")
    log_parser.add_argument("--log", required=True, help="Log file to analyze")
    log_parser.add_argument("--threshold", type=float, default=50.0, help="Latency threshold (ms)")

    # Analyze DNN pipeline
    dnn_parser = analyze_subparsers.add_parser("dnn-pipeline", help="Analyze DNN inference pipeline")
    dnn_parser.add_argument("--csv", help="CSV file with layer timings")
    dnn_parser.add_argument("--profiler", help="Profiler output text file")
    dnn_parser.add_argument("--batch-size", type=int, default=1, help="Batch size used")
    dnn_parser.add_argument("--top-layers", type=int, default=5, help="Number of slowest layers to report")

    # Analyze tegrastats
    tegra_parser = analyze_subparsers.add_parser("tegrastats", help="Analyze Tegrastats output")
    tegra_parser.add_argument("--log", required=True, help="Tegrastats log file")
    tegra_parser.add_argument("--throttle-threshold", type=float, default=85.0, help="Thermal throttling threshold (°C)")

    # Analyze efficiency
    eff_parser = analyze_subparsers.add_parser("efficiency", help="Analyze power efficiency")
    eff_parser.add_argument("--csv", required=True, help="CSV file with benchmark data")

    # Analyze variability
    var_parser = analyze_subparsers.add_parser("variability", help="Analyze latency variability")
    var_parser.add_argument("--csv", required=True, help="CSV file with latency data")
    var_parser.add_argument("--column", default="latency_ms", help="Column name for latency values")

    # Benchmark commands
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench_subparsers = bench_parser.add_subparsers(dest="bench_type")

    # Batch size benchmark
    batch_parser = bench_subparsers.add_parser("batching", help="Batch size trade-off analysis")
    batch_parser.add_argument("--batch-sizes", default="1,4,8,16,32", help="Comma-separated batch sizes")
    batch_parser.add_argument("--images", type=int, default=1000, help="Number of images")

    # LLM benchmark
    llm_parser = bench_subparsers.add_parser("llm", help="LLM inference latency")
    llm_parser.add_argument("--prompt-length", type=int, default=512, help="Prompt token count")
    llm_parser.add_argument("--output-tokens", type=int, default=256, help="Output token count")
    llm_parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs")

    # Monitor commands
    monitor_parser = subparsers.add_parser("monitor", help="Monitor system metrics")
    monitor_subparsers = monitor_parser.add_subparsers(dest="monitor_type")

    # GPU monitor
    gpu_parser = monitor_subparsers.add_parser("gpu", help="Monitor GPU metrics")
    gpu_parser.add_argument("--duration", type=int, default=300, help="Monitor duration (seconds)")
    gpu_parser.add_argument("--interval", type=int, default=1, help="Sample interval (seconds)")

    # KV cache monitor
    cache_parser = monitor_subparsers.add_parser("kv-cache", help="Monitor KV cache")
    cache_parser.add_argument("--max-length", type=int, default=2048, help="Max sequence length")

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
    print(f"\n⏱️  Timing:")
    print(f"  Total Time: {timing.get('total_time_ms', timing.get('avg_total_ms', 0)):.2f}ms")
    print(f"  GPU Time: {timing.get('gpu_time_ms', 0):.2f}ms")
    print(f"  DLA Time: {timing.get('dla_time_ms', 0):.2f}ms")

    device_split = metrics.get("device_split", {})
    print(f"\n📊 Device Split:")
    print(f"  GPU: {device_split.get('gpu_percentage', 0):.1f}%")
    print(f"  DLA: {device_split.get('dla_percentage', 0):.1f}%")

    throughput = metrics.get("throughput_fps", metrics.get("throughput", {}).get("avg_fps", 0))
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
        print(f"\n💡 Recommendations:")
        for rec in recommendations:
            print(f"  • {rec}")

    return result


def run_analyze_tegrastats(args, config):
    """Run tegrastats analysis."""
    analyzer = TegrastatsAnalyzer(throttle_temp_threshold=args.throttle_threshold)
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
    print(f"\n🎮 GPU:")
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
    print(f"\n🌡️  Thermal:")
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

    print(f"\n📊 Percentiles:")
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
    results = benchmark.run(prompt_tokens=args.prompt_length, output_tokens=args.output_tokens, num_runs=args.runs)

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
        import time

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


def main():
    """Main CLI entry point."""
    parser = setup_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Load configuration
    config = ConfigManager.load_or_default(args.config)

    # Route to appropriate handler
    result = None
    try:
        if args.command == "analyze":
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
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1

    # Save output if requested
    if args.output and result:
        with open(args.output, "w") as f:
            if hasattr(result, "to_dict"):
                json.dump(result.to_dict(), f, indent=2)
            else:
                json.dump(result, f, indent=2)
        print(f"\n✅ Results saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
