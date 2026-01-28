#!/usr/bin/env python3
"""
Example: Analyze percentile latencies from benchmark data.

This demonstrates using the AutoPerfPy package programmatically.
"""

from autoperfpy import PercentileLatencyAnalyzer, ConfigManager


def main():
    """Run percentile latency analysis."""
    # Load configuration (or use defaults)
    config = ConfigManager.load_or_default("config.yaml")

    # Initialize analyzer
    analyzer = PercentileLatencyAnalyzer(config)

    # Analyze benchmark data
    print("📊 Analyzing benchmark data...")
    result = analyzer.analyze("scripts/data/automotive_benchmark_data.csv")

    # Display results
    print("\n" + "=" * 70)
    print("PERCENTILE LATENCY ANALYSIS")
    print("=" * 70)

    for workload_batch, metrics in result.metrics.items():
        print(f"\n{workload_batch}:")
        print(f"  P99: {metrics.get('p99', 0):.2f}ms")
        print(f"  P95: {metrics.get('p95', 0):.2f}ms")
        print(f"  P50: {metrics.get('p50', 0):.2f}ms (Median)")
        print(f"  Mean: {metrics.get('mean', 0):.2f}ms ± {metrics.get('std', 0):.2f}ms")
        print(f"  Range: [{metrics.get('min', 0):.1f}, {metrics.get('max', 0):.1f}]ms")
        print(f"  Samples: {metrics.get('num_samples', 0)}")
        if "power_mean" in metrics:
            print(f"  Avg Power: {metrics['power_mean']:.1f}W")

    # Display summary
    summary = analyzer.summarize()
    print(f"\n{'=' * 70}")
    print(f"Summary: {summary['total_analyses']} analysis(ses) completed")
    print(f"Best latency: {summary['best_latency']:.2f}ms")
    print(f"Worst latency: {summary['worst_latency']:.2f}ms")

    return result


if __name__ == "__main__":
    main()
