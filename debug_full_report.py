"""Generate a full test HTML report to verify chart rendering."""

from trackiq.reporting import HTMLReportGenerator, charts as shared_charts
import random
import time


def generate_synthetic_demo_data():
    """Generate synthetic demo data for testing."""
    random.seed(42)
    base_time = time.time() - 60
    num_samples = 100
    warmup_samples = 10

    samples = []
    for i in range(num_samples):
        is_warmup = i < warmup_samples
        workload_factor = 1.0 + 0.2 * (i / num_samples)

        if is_warmup:
            base_latency = 50.0 - (i * 2.5)
        else:
            base_latency = 25.0
        latency = base_latency * workload_factor + random.gauss(0, 2)

        gpu_percent = 70 + random.gauss(0, 5) + (10 * workload_factor)
        gpu_percent = max(0, min(100, gpu_percent))

        cpu_percent = 40 + random.gauss(0, 8) + (5 * workload_factor)
        cpu_percent = max(0, min(100, cpu_percent))

        memory_used = 4096 + i * 2 + random.gauss(0, 50)
        power = 15 + (gpu_percent / 100) * 120 + random.gauss(0, 3)
        temp = 45 + (power - 15) / 135 * 30 + random.gauss(0, 1)
        throughput = 1000 / latency

        samples.append(
            {
                "timestamp": base_time + i * 0.6,
                "metrics": {
                    "latency_ms": round(latency, 2),
                    "cpu_percent": round(cpu_percent, 1),
                    "gpu_percent": round(gpu_percent, 1),
                    "memory_used_mb": round(memory_used, 0),
                    "memory_total_mb": 16384,
                    "memory_percent": round(memory_used / 16384 * 100, 1),
                    "power_w": round(power, 1),
                    "temperature_c": round(temp, 1),
                    "throughput_fps": round(throughput, 1),
                    "is_warmup": is_warmup,
                },
                "metadata": {"sample_index": i},
            }
        )

    # Calculate summary
    steady_samples = samples[warmup_samples:]
    latencies = [s["metrics"]["latency_ms"] for s in steady_samples]
    gpus = [s["metrics"]["gpu_percent"] for s in steady_samples]
    cpus = [s["metrics"]["cpu_percent"] for s in steady_samples]
    powers = [s["metrics"]["power_w"] for s in steady_samples]
    temps = [s["metrics"]["temperature_c"] for s in steady_samples]
    mems = [s["metrics"]["memory_used_mb"] for s in steady_samples]
    throughputs = [s["metrics"]["throughput_fps"] for s in steady_samples]

    def percentile(data, p):
        sorted_data = sorted(data)
        idx = int(p / 100 * (len(sorted_data) - 1))
        return sorted_data[idx]

    summary = {
        "sample_count": num_samples,
        "warmup_samples": warmup_samples,
        "duration_seconds": num_samples * 0.6,
        "latency": {
            "mean_ms": round(sum(latencies) / len(latencies), 2),
            "min_ms": round(min(latencies), 2),
            "max_ms": round(max(latencies), 2),
            "p50_ms": round(percentile(latencies, 50), 2),
            "p95_ms": round(percentile(latencies, 95), 2),
            "p99_ms": round(percentile(latencies, 99), 2),
        },
        "cpu": {
            "mean_percent": round(sum(cpus) / len(cpus), 1),
            "max_percent": round(max(cpus), 1),
        },
        "gpu": {
            "mean_percent": round(sum(gpus) / len(gpus), 1),
            "max_percent": round(max(gpus), 1),
        },
        "power": {
            "mean_w": round(sum(powers) / len(powers), 1),
            "max_w": round(max(powers), 1),
        },
        "temperature": {
            "mean_c": round(sum(temps) / len(temps), 1),
            "max_c": round(max(temps), 1),
        },
        "memory": {
            "mean_mb": round(sum(mems) / len(mems), 1),
            "max_mb": round(max(mems), 1),
        },
        "throughput": {
            "mean_fps": round(sum(throughputs) / len(throughputs), 1),
            "min_fps": round(min(throughputs), 1),
            "max_fps": round(max(throughputs), 1),
        },
    }

    return {"samples": samples, "summary": summary}


def main():
    data = generate_synthetic_demo_data()
    samples = data["samples"]
    summary = data["summary"]

    # Convert to DataFrame
    df = shared_charts.samples_to_dataframe(samples)

    print(f"DataFrame columns: {list(df.columns)}")
    print(f"Summary keys: {list(summary.keys())}")

    # Create report
    report = HTMLReportGenerator(
        title="Test Performance Report - Full Charts",
        author="AutoPerfPy",
        theme="light",
    )

    # Add metadata
    report.add_metadata("Device", "Test Device")
    report.add_metadata("Test Type", "Synthetic Benchmark")

    # Add summary items
    report.add_summary_item("Samples", summary["sample_count"], "", "neutral")

    lat = summary.get("latency", {})
    report.add_summary_item(
        "P99 Latency", f"{lat.get('p99_ms', 0):.2f}", "ms", "warning"
    )
    report.add_summary_item(
        "P50 Latency", f"{lat.get('p50_ms', 0):.2f}", "ms", "neutral"
    )
    report.add_summary_item(
        "Mean Latency", f"{lat.get('mean_ms', 0):.2f}", "ms", "neutral"
    )

    thr = summary.get("throughput", {})
    report.add_summary_item(
        "Throughput", f"{thr.get('mean_fps', 0):.1f}", "FPS", "good"
    )

    pwr = summary.get("power", {})
    report.add_summary_item("Power", f"{pwr.get('mean_w', 0):.1f}", "W", "neutral")

    gpu = summary.get("gpu", {})
    report.add_summary_item(
        "GPU Util", f"{gpu.get('mean_percent', 0):.1f}", "%", "good"
    )

    mem = summary.get("memory", {})
    report.add_summary_item("Memory", f"{mem.get('mean_mb', 0):.0f}", "MB", "neutral")

    # Build charts
    sections = shared_charts.build_all_charts(df, summary)
    print(f"\nSections built: {list(sections.keys())}")
    for sec_name, charts in sections.items():
        print(f"  {sec_name}: {len(charts)} charts - {[c[0] for c in charts]}")

    # Add charts to report
    shared_charts.add_charts_to_html_report(report, df, summary)

    # Generate HTML
    output_path = "debug_test_report.html"
    report.generate_html(output_path)
    print(f"\nGenerated: {output_path}")

    # Check file size
    import os

    size = os.path.getsize(output_path)
    print(f"File size: {size} bytes ({size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
