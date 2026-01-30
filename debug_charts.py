"""Debug script to check what columns are in the data."""

from autoperf_app.reports import charts as shared_charts

# Simulate the generate_synthetic_demo_data function from the UI
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

    return {"samples": samples}


def main():
    data = generate_synthetic_demo_data()
    samples = data["samples"]

    print(f"Number of samples: {len(samples)}")
    print(f"First sample keys: {samples[0].keys()}")
    print(f"First sample metrics keys: {samples[0]['metrics'].keys()}")

    # Convert to DataFrame using the charts module
    df = shared_charts.samples_to_dataframe(samples)

    print(f"\nDataFrame shape: {df.shape}")
    print(f"DataFrame columns: {list(df.columns)}")
    print(f"\nFirst row:\n{df.iloc[0].to_dict()}")

    # Check which chart functions would work
    print("\n--- Chart Function Checks ---")

    # Utilization
    has_cpu = "cpu_percent" in df.columns
    has_gpu = "gpu_percent" in df.columns
    print(f"Has cpu_percent: {has_cpu}")
    print(f"Has gpu_percent: {has_gpu}")

    fig = shared_charts.create_utilization_timeline(df)
    print(f"create_utilization_timeline: {'OK' if fig else 'None'}")

    # Power
    has_power = "power_w" in df.columns
    has_temp = "temperature_c" in df.columns
    print(f"Has power_w: {has_power}")
    print(f"Has temperature_c: {has_temp}")

    fig = shared_charts.create_power_timeline(df)
    print(f"create_power_timeline: {'OK' if fig else 'None'}")

    fig = shared_charts.create_temperature_timeline(df)
    print(f"create_temperature_timeline: {'OK' if fig else 'None'}")

    # Memory
    has_memory = "memory_used_mb" in df.columns
    print(f"Has memory_used_mb: {has_memory}")

    fig = shared_charts.create_memory_timeline(df)
    print(f"create_memory_timeline: {'OK' if fig else 'None'}")

    # Latency
    has_latency = "latency_ms" in df.columns
    print(f"Has latency_ms: {has_latency}")

    fig = shared_charts.create_latency_timeline(df)
    print(f"create_latency_timeline: {'OK' if fig else 'None'}")


if __name__ == "__main__":
    main()
