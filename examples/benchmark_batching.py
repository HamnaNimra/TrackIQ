#!/usr/bin/env python3
"""
Example: Benchmark batch size trade-offs.

Shows how to use the AutoPerfPy benchmarking module.
"""

from autoperfpy import BatchingTradeoffBenchmark, ConfigManager


def main():
    """Run batching trade-off benchmark."""
    # Load configuration
    config = ConfigManager.load_or_default("config.yaml")

    # Create benchmark with custom batch sizes
    benchmark = BatchingTradeoffBenchmark(config)

    print("⚡ Running Batching Trade-off Analysis...")
    print("=" * 70)

    # Run benchmark
    results = benchmark.run(batch_sizes=[1, 2, 4, 8, 16, 32, 64])

    # Display results in table format
    print("\n{:<10} {:<20} {:<25}".format('Batch', 'Latency (ms)', 'Throughput (img/s)'))
    print("-" * 70)

    for i, batch in enumerate(results["batch_size"]):
        latency = results["latency_ms"][i]
        throughput = results["throughput_img_per_sec"][i]
        print(f"{batch:<10} {latency:<20.2f} {throughput:<25.1f}")

    # Get optimal batch sizes
    optimal_latency = benchmark.get_optimal_batch_size("latency")
    optimal_throughput = benchmark.get_optimal_batch_size("throughput")

    print("\n" + '=' * 70)
    print("📊 Recommendations:")
    print(f"  For minimum latency: Batch size {optimal_latency}")
    print(f"  For maximum throughput: Batch size {optimal_throughput}")

    return results


if __name__ == "__main__":
    main()
