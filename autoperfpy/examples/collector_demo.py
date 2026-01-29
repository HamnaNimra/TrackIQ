#!/usr/bin/env python3
"""Demo script showing collector usage in AutoPerfPy.

This example demonstrates how to use the different collectors:
- SyntheticCollector: For testing without hardware
- NVMLCollector: For NVIDIA GPU metrics
- TegrastatsCollector: For Jetson/DriveOS platforms
- PsutilCollector: For cross-platform system metrics

Usage:
    python -m autoperfpy.examples.collector_demo --collector synthetic
    python -m autoperfpy.examples.collector_demo --collector psutil
    python -m autoperfpy.examples.collector_demo --collector nvml
"""

import argparse
import json
import time


def demo_synthetic_collector(num_samples: int = 50, output_file: str = None):
    """Demo the SyntheticCollector."""
    from autoperfpy.collectors import SyntheticCollector

    print("=" * 60)
    print("SyntheticCollector Demo")
    print("=" * 60)

    config = {
        "warmup_samples": 5,
        "base_latency_ms": 25.0,
        "workload_pattern": "cyclic",
    }

    collector = SyntheticCollector(config=config)
    collector.start()

    print(f"Collecting {num_samples} samples...")
    print()
    print(f"{'Sample':>6} {'Latency':>10} {'CPU':>8} {'GPU':>8} {'Power':>8} {'Temp':>8} {'Warmup':>8}")
    print("-" * 60)

    for i in range(num_samples):
        timestamp = time.time()
        metrics = collector.sample(timestamp)

        print(
            f"{i:>6} "
            f"{metrics['latency_ms']:>10.2f}ms "
            f"{metrics['cpu_percent']:>7.1f}% "
            f"{metrics['gpu_percent']:>7.1f}% "
            f"{metrics['power_w']:>7.1f}W "
            f"{metrics['temperature_c']:>7.1f}C "
            f"{'*' if metrics['is_warmup'] else '':>8}"
        )
        time.sleep(0.05)

    collector.stop()
    export = collector.export()

    print()
    print("Summary:")
    print(f"  Samples: {export.summary['sample_count']}")
    print(f"  Duration: {export.summary['duration_seconds']:.2f}s")
    print(f"  Latency P99: {export.summary['latency']['p99_ms']:.2f}ms")
    print(f"  Mean GPU: {export.summary['gpu']['mean_percent']:.1f}%")

    if output_file:
        with open(output_file, "w") as f:
            json.dump(export.to_dict(), f, indent=2)
        print(f"  Exported to: {output_file}")


def demo_psutil_collector(num_samples: int = 20, output_file: str = None):
    """Demo the PsutilCollector."""
    from autoperfpy.collectors import PsutilCollector

    print("=" * 60)
    print("PsutilCollector Demo")
    print("=" * 60)

    if not PsutilCollector.is_available():
        print("ERROR: psutil is not installed. Run: pip install psutil")
        return

    collector = PsutilCollector(
        include_per_cpu=True,
        include_disk_io=True,
        config={"warmup_samples": 2},
    )

    # Print system info
    collector.start()
    info = collector.get_system_info()
    print(f"System: {info['platform']['system']} {info['platform']['release']}")
    print(f"CPU: {info['cpu']['cores_logical']} logical cores ({info['cpu']['cores_physical']} physical)")
    print(f"Memory: {info['memory']['total_gb']:.1f} GB")
    print()

    print(f"{'Sample':>6} {'CPU':>8} {'Memory':>10} {'MemUsed':>10} {'Temp':>8}")
    print("-" * 50)

    for i in range(num_samples):
        timestamp = time.time()
        metrics = collector.sample(timestamp)

        temp_str = f"{metrics.get('temperature_c', 0):.1f}C" if metrics.get('temperature_c') else "N/A"

        print(
            f"{i:>6} "
            f"{metrics['cpu_percent']:>7.1f}% "
            f"{metrics['memory_percent']:>9.1f}% "
            f"{metrics['memory_used_mb']:>9.0f}MB "
            f"{temp_str:>8}"
        )
        time.sleep(0.5)

    collector.stop()
    export = collector.export()

    print()
    print("Summary:")
    print(f"  Samples: {export.summary['sample_count']}")
    print(f"  Duration: {export.summary['duration_seconds']:.2f}s")
    print(f"  Mean CPU: {export.summary['cpu']['mean_percent']:.1f}%")
    print(f"  Max Memory: {export.summary['memory']['max_mb']:.0f}MB")

    if output_file:
        with open(output_file, "w") as f:
            json.dump(export.to_dict(), f, indent=2)
        print(f"  Exported to: {output_file}")


def demo_nvml_collector(num_samples: int = 20, output_file: str = None):
    """Demo the NVMLCollector."""
    from autoperfpy.collectors import NVMLCollector

    print("=" * 60)
    print("NVMLCollector Demo")
    print("=" * 60)

    try:
        devices = NVMLCollector.get_available_devices()
        print(f"Found {len(devices)} NVIDIA GPU(s):")
        for d in devices:
            print(f"  [{d['index']}] {d['name']} ({d['memory_total_mb']:.0f}MB)")
        print()
    except ImportError:
        print("ERROR: pynvml is not installed. Run: pip install pynvml")
        return
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return

    collector = NVMLCollector(device_index=0, config={"warmup_samples": 2})
    collector.start()

    # Print device info
    info = collector.get_device_info()
    print(f"Device: {info['name']}")
    print(f"Memory: {info.get('memory_total_mb', 0):.0f}MB")
    print()

    print(f"{'Sample':>6} {'GPU':>8} {'Memory':>10} {'Power':>8} {'Temp':>8}")
    print("-" * 50)

    for i in range(num_samples):
        timestamp = time.time()
        metrics = collector.sample(timestamp)

        power_str = f"{metrics.get('power_w', 0):.1f}W" if metrics.get('power_w') else "N/A"
        temp_str = f"{metrics.get('temperature_c', 0):.1f}C" if metrics.get('temperature_c') else "N/A"

        print(
            f"{i:>6} "
            f"{metrics['gpu_percent']:>7.1f}% "
            f"{metrics['memory_used_mb']:>9.0f}MB "
            f"{power_str:>8} "
            f"{temp_str:>8}"
        )
        time.sleep(0.5)

    collector.stop()
    export = collector.export()

    print()
    print("Summary:")
    print(f"  Samples: {export.summary['sample_count']}")
    print(f"  Duration: {export.summary['duration_seconds']:.2f}s")
    if export.summary.get('gpu', {}).get('mean'):
        print(f"  Mean GPU: {export.summary['gpu']['mean']:.1f}%")
    if export.summary.get('power', {}).get('mean_w'):
        print(f"  Mean Power: {export.summary['power']['mean_w']:.1f}W")

    if output_file:
        with open(output_file, "w") as f:
            json.dump(export.to_dict(), f, indent=2)
        print(f"  Exported to: {output_file}")


def demo_tegrastats_collector(filepath: str = None, output_file: str = None):
    """Demo the TegrastatsCollector with file-based collection."""
    from autoperfpy.collectors import TegrastatsCollector

    print("=" * 60)
    print("TegrastatsCollector Demo")
    print("=" * 60)

    if filepath is None:
        # Check if live tegrastats is available
        if TegrastatsCollector.is_available():
            print("tegrastats is available. Run with --file <path> to analyze a log file.")
            print("Or the collector can be used in live mode on Jetson/DriveOS platforms.")
        else:
            print("tegrastats not found. This collector requires a Jetson/DriveOS platform.")
            print("Provide a tegrastats log file with --file <path>")
        return

    print(f"Analyzing: {filepath}")
    print()

    collector = TegrastatsCollector(mode="file", filepath=filepath)
    collector.start()

    remaining = collector.get_remaining_samples()
    print(f"Found {remaining} tegrastats samples")
    print()

    print(f"{'Sample':>6} {'CPU':>8} {'GPU':>8} {'Memory':>10} {'Temp':>8}")
    print("-" * 50)

    sample_count = 0
    max_display = 20

    while True:
        timestamp = time.time()
        metrics = collector.sample(timestamp)

        if metrics is None:
            break

        if sample_count < max_display:
            print(
                f"{sample_count:>6} "
                f"{metrics['cpu_percent']:>7.1f}% "
                f"{metrics['gpu_percent']:>7.1f}% "
                f"{metrics['memory_used_mb']:>9.0f}MB "
                f"{metrics['temperature_c']:>7.1f}C"
            )
        elif sample_count == max_display:
            print("... (remaining samples not displayed)")

        sample_count += 1

    collector.stop()
    export = collector.export()

    print()
    print("Summary:")
    print(f"  Samples: {export.summary['sample_count']}")
    print(f"  Duration: {export.summary['duration_seconds']:.2f}s")
    print(f"  Mean CPU: {export.summary['cpu']['mean_percent']:.1f}%")
    print(f"  Mean GPU: {export.summary['gpu']['mean_percent']:.1f}%")
    print(f"  Max Temp: {export.summary['temperature']['max_c']:.1f}C")

    if output_file:
        with open(output_file, "w") as f:
            json.dump(export.to_dict(), f, indent=2)
        print(f"  Exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Demo AutoPerfPy collectors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m autoperfpy.examples.collector_demo --collector synthetic
  python -m autoperfpy.examples.collector_demo --collector psutil --samples 30
  python -m autoperfpy.examples.collector_demo --collector nvml --output gpu_metrics.json
  python -m autoperfpy.examples.collector_demo --collector tegrastats --file tegrastats.log
        """,
    )

    parser.add_argument(
        "--collector", "-c",
        choices=["synthetic", "psutil", "nvml", "tegrastats"],
        default="synthetic",
        help="Collector to demo (default: synthetic)",
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=50,
        help="Number of samples to collect (default: 50)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for JSON export",
    )
    parser.add_argument(
        "--file", "-f",
        help="Input file (for tegrastats file mode)",
    )

    args = parser.parse_args()

    if args.collector == "synthetic":
        demo_synthetic_collector(args.samples, args.output)
    elif args.collector == "psutil":
        demo_psutil_collector(args.samples, args.output)
    elif args.collector == "nvml":
        demo_nvml_collector(args.samples, args.output)
    elif args.collector == "tegrastats":
        demo_tegrastats_collector(args.file, args.output)


if __name__ == "__main__":
    main()
