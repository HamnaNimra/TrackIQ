#!/usr/bin/env python3
"""
Example: Collect synthetic performance metrics using the collector interface.

Demonstrates the collector abstraction with SyntheticCollector, showing:
- Collector lifecycle: start() -> sample() -> stop() -> export()
- Configuration options for realistic data generation
- Real-time metric display during collection
- Summary statistics and data export

This example can be used as a template for integrating real hardware
collectors (NVML, tegrastats, psutil) once implemented.
"""

import json
import time
from autoperfpy import SyntheticCollector


def main():
    """Run synthetic data collection demonstration."""
    # Configure the synthetic collector
    # See SyntheticCollector.DEFAULT_CONFIG for all options
    config = {
        "warmup_samples": 5,           # Samples with elevated latency
        "base_latency_ms": 25.0,       # Target inference latency
        "latency_jitter_percent": 15.0, # Random jitter range
        "base_cpu_percent": 40.0,       # Base CPU utilization
        "base_gpu_percent": 80.0,       # Base GPU utilization
        "workload_pattern": "cyclic",   # steady, cyclic, ramp, burst
        "cycle_period_samples": 20,     # Samples per cycle
    }

    # Create the collector
    collector = SyntheticCollector(config=config)

    # Collection parameters
    duration_seconds = 10
    sample_interval = 0.1  # 100ms between samples

    print("=" * 70)
    print("AutoPerfPy Synthetic Collector Demo")
    print("=" * 70)
    print(f"Duration: {duration_seconds}s | Interval: {sample_interval*1000:.0f}ms")
    print(f"Workload Pattern: {config['workload_pattern']}")
    print("=" * 70)

    # Start collection
    collector.start()
    start_time = time.time()

    try:
        sample_count = 0
        while time.time() - start_time < duration_seconds:
            # Collect a sample
            timestamp = time.time()
            metrics = collector.sample(timestamp)

            # Display real-time metrics
            warmup_marker = "[WARMUP]" if metrics["is_warmup"] else ""
            print(
                f"[{sample_count:3d}] "
                f"Latency: {metrics['latency_ms']:6.2f}ms | "
                f"CPU: {metrics['cpu_percent']:5.1f}% | "
                f"GPU: {metrics['gpu_percent']:5.1f}% | "
                f"Mem: {metrics['memory_used_mb']:7.0f}MB | "
                f"Power: {metrics['power_w']:5.1f}W | "
                f"Temp: {metrics['temperature_c']:4.1f}C "
                f"{warmup_marker}"
            )

            sample_count += 1
            time.sleep(sample_interval)

    except KeyboardInterrupt:
        print("\nCollection interrupted by user")

    # Stop collection
    collector.stop()

    # Export and display results
    export = collector.export()
    summary = export.summary

    print("\n" + "=" * 70)
    print("Collection Summary")
    print("=" * 70)
    print(f"Collector: {export.collector_name}")
    print(f"Total Samples: {summary['sample_count']}")
    print(f"Warmup Samples: {summary['warmup_samples']}")
    if summary.get('duration_seconds'):
        print(f"Duration: {summary['duration_seconds']:.2f}s")

    print("\nLatency Statistics (excluding warmup):")
    latency = summary['latency']
    print(f"  Mean:  {latency['mean_ms']:.2f}ms")
    print(f"  Min:   {latency['min_ms']:.2f}ms")
    print(f"  Max:   {latency['max_ms']:.2f}ms")
    print(f"  P50:   {latency['p50_ms']:.2f}ms")
    print(f"  P95:   {latency['p95_ms']:.2f}ms")
    print(f"  P99:   {latency['p99_ms']:.2f}ms")

    print("\nResource Utilization:")
    print(f"  CPU:   {summary['cpu']['mean_percent']:.1f}% avg, {summary['cpu']['max_percent']:.1f}% max")
    print(f"  GPU:   {summary['gpu']['mean_percent']:.1f}% avg, {summary['gpu']['max_percent']:.1f}% max")
    print(f"  Memory: {summary['memory']['mean_mb']:.0f}MB avg, {summary['memory']['max_mb']:.0f}MB max")

    print("\nPower & Thermal:")
    print(f"  Power: {summary['power']['mean_w']:.1f}W avg, {summary['power']['max_w']:.1f}W max")
    print(f"  Temp:  {summary['temperature']['mean_c']:.1f}C avg, {summary['temperature']['max_c']:.1f}C max")

    print("\nThroughput:")
    print(f"  FPS:   {summary['throughput']['mean_fps']:.1f} avg, {summary['throughput']['min_fps']:.1f} min")

    # Optional: Save full export to JSON
    save_export = False  # Set to True to save
    if save_export:
        output_file = "synthetic_metrics_export.json"
        with open(output_file, "w") as f:
            json.dump(export.to_dict(), f, indent=2)
        print(f"\nFull export saved to: {output_file}")

    return export


if __name__ == "__main__":
    main()
