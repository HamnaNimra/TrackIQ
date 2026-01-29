"""Test HTML report generation with all charts."""

from trackiq.reporting import HTMLReportGenerator, charts as shared_charts
import random
import time


def generate_test_data():
    """Generate test data matching Streamlit UI format."""
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
                    "power_w": round(power, 1),
                    "temperature_c": round(temp, 1),
                    "throughput_fps": round(throughput, 1),
                    "is_warmup": is_warmup,
                },
            }
        )

    # Build summary
    steady = samples[warmup_samples:]
    lats = [s["metrics"]["latency_ms"] for s in steady]
    gpus = [s["metrics"]["gpu_percent"] for s in steady]
    cpus = [s["metrics"]["cpu_percent"] for s in steady]
    pwrs = [s["metrics"]["power_w"] for s in steady]
    tmps = [s["metrics"]["temperature_c"] for s in steady]
    mems = [s["metrics"]["memory_used_mb"] for s in steady]
    thrs = [s["metrics"]["throughput_fps"] for s in steady]

    def pct(arr, p):
        s = sorted(arr)
        return s[int(p / 100 * (len(s) - 1))]

    summary = {
        "sample_count": num_samples,
        "warmup_samples": warmup_samples,
        "latency": {
            "mean_ms": sum(lats) / len(lats),
            "min_ms": min(lats),
            "max_ms": max(lats),
            "p50_ms": pct(lats, 50),
            "p95_ms": pct(lats, 95),
            "p99_ms": pct(lats, 99),
        },
        "cpu": {"mean_percent": sum(cpus) / len(cpus), "max_percent": max(cpus)},
        "gpu": {"mean_percent": sum(gpus) / len(gpus), "max_percent": max(gpus)},
        "power": {"mean_w": sum(pwrs) / len(pwrs), "max_w": max(pwrs)},
        "temperature": {"mean_c": sum(tmps) / len(tmps), "max_c": max(tmps)},
        "memory": {"mean_mb": sum(mems) / len(mems), "max_mb": max(mems)},
        "throughput": {
            "mean_fps": sum(thrs) / len(thrs),
            "min_fps": min(thrs),
            "max_fps": max(thrs),
        },
    }

    return {"samples": samples, "summary": summary}


def main():
    data = generate_test_data()
    samples = data["samples"]
    summary = data["summary"]

    # Convert to DataFrame
    df = shared_charts.samples_to_dataframe(samples)
    print(f"DataFrame columns: {list(df.columns)}")
    print(f"DataFrame shape: {df.shape}")

    # Create report
    report = HTMLReportGenerator(
        title="Performance Analysis Report",
        author="AutoPerfPy",
        theme="light",
    )

    # Add metadata
    report.add_metadata("Device", "Test GPU")
    report.add_metadata("Profile", "benchmark")

    # Add summary items
    report.add_summary_item("Samples", summary["sample_count"], "", "neutral")

    lat = summary["latency"]
    report.add_summary_item("P99 Latency", f"{lat['p99_ms']:.2f}", "ms", "warning")
    report.add_summary_item("Mean Latency", f"{lat['mean_ms']:.2f}", "ms", "neutral")

    thr = summary["throughput"]
    report.add_summary_item("Throughput", f"{thr['mean_fps']:.1f}", "FPS", "good")

    pwr = summary["power"]
    report.add_summary_item("Power", f"{pwr['mean_w']:.1f}", "W", "neutral")

    gpu = summary["gpu"]
    report.add_summary_item("GPU Util", f"{gpu['mean_percent']:.1f}", "%", "good")

    # Add charts using same logic as Streamlit UI
    shared_charts.add_charts_to_html_report(report, df, summary)

    # Generate
    output = "test_performance_report.html"
    report.generate_html(output)

    import os

    size = os.path.getsize(output)
    print(f"\nGenerated: {output}")
    print(f"File size: {size:,} bytes ({size/1024:.1f} KB)")

    # Verify content
    with open(output, "r") as f:
        content = f.read()

    sections = ["latency", "utilization", "power", "memory", "throughput"]
    print("\nSections in HTML:")
    for s in sections:
        found = f'id="{s}' in content
        print(f"  {s}: {'OK' if found else 'MISSING'}")


if __name__ == "__main__":
    main()
