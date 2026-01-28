#!/usr/bin/env python3
"""
Example: Monitor GPU memory during inference.

Demonstrates the monitoring module.
"""

import time
from autoperfpy import GPUMemoryMonitor, ConfigManager


def main():
    """Run GPU memory monitoring."""
    # Load configuration
    config = ConfigManager.load_or_default("config.yaml")

    # Create monitor
    monitor = GPUMemoryMonitor(config)

    # Monitor for 30 seconds
    duration = 30
    print(f"📊 Monitoring GPU memory for {duration} seconds...")
    print("=" * 70)

    monitor.start()

    try:
        for i in range(duration):
            time.sleep(1)
            metrics = monitor.get_metrics()
            if metrics:
                latest = metrics[-1]
                memory_used = latest.get("gpu_memory_used_mb", 0)
                memory_total = latest.get("gpu_memory_total_mb", 0)
                memory_percent = latest.get("gpu_memory_percent", 0)
                utilization = latest.get("gpu_utilization_percent", 0)

                print(
                    f"[{i+1:2d}s] Memory: {memory_used:8.0f}MB / {memory_total:8.0f}MB "
                    f"({memory_percent:5.1f}%) | Utilization: {utilization:5.1f}%"
                )

            if (i + 1) % 10 == 0:
                print("-" * 70)
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")
    finally:
        monitor.stop()

    # Display summary
    summary = monitor.get_summary()
    if summary:
        print(f"\n{'=' * 70}")
        print("📈 Summary Statistics:")
        print(f"  Average Memory: {summary['avg_memory_mb']:.0f}MB")
        print(f"  Peak Memory: {summary['max_memory_mb']:.0f}MB")
        print(f"  Avg Utilization: {summary['avg_utilization_percent']:.1f}%")
        print(f"  Peak Utilization: {summary['max_utilization_percent']:.1f}%")
        print(f"  Samples Collected: {summary['samples_collected']}")

    return summary


if __name__ == "__main__":
    main()
