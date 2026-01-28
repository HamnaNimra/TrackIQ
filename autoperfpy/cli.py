"""Command-line interface for AutoPerfPy."""

import argparse
import sys

from autoperfpy.config import ConfigManager
from autoperfpy.analyzers import PercentileLatencyAnalyzer, LogAnalyzer
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
  autoperfpy analyze latency --csv data.csv
  autoperfpy benchmark batching --batch-sizes 1,4,8,16
  autoperfpy benchmark llm --prompt-length 512
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
