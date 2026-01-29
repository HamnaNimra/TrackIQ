"""Streamlit-based UI for AutoPerfPy.

This module provides an interactive dashboard for visualizing performance metrics
collected by AutoPerfPy collectors. It supports:
- Loading collector output (JSON/CSV)
- Displaying metrics: latency percentiles, throughput, CPU/GPU utilization, power timelines
- Multi-run comparison
- Graceful handling of missing metrics

Example usage:
    # From command line:
    autoperfpy ui

    # Or directly with streamlit:
    streamlit run autoperfpy/ui/streamlit_app.py

    # With a specific data file:
    autoperfpy ui --data results.json

Authors:
    AutoPerfPy Team
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# Add parent directory to path for imports when running directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    st.error(f"Missing required dependency: {e}")
    st.info("Install with: pip install pandas plotly")
    st.stop()


# Page configuration
st.set_page_config(
    page_title="AutoPerfPy Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_json_data(filepath: str) -> Optional[Dict[str, Any]]:
    """Load collector export data from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary with collector data or None if loading fails
    """
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load JSON file: {e}")
        return None


def load_csv_data(filepath: str) -> Optional[pd.DataFrame]:
    """Load benchmark data from CSV file.

    Args:
        filepath: Path to CSV file

    Returns:
        DataFrame with data or None if loading fails
    """
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        st.error(f"Failed to load CSV file: {e}")
        return None


def generate_synthetic_demo_data() -> Dict[str, Any]:
    """Generate synthetic demo data for demonstration purposes.

    Returns:
        Dictionary mimicking CollectorExport format
    """
    import random
    import time

    random.seed(42)
    base_time = time.time() - 60  # Start 60 seconds ago
    num_samples = 100
    warmup_samples = 10

    samples = []
    for i in range(num_samples):
        is_warmup = i < warmup_samples

        # Generate realistic metrics
        workload_factor = 1.0 + 0.2 * (i / num_samples)  # Gradual increase

        # Latency with warmup effect and jitter
        if is_warmup:
            base_latency = 50.0 - (i * 2.5)  # Warmup improvement
        else:
            base_latency = 25.0
        latency = base_latency * workload_factor + random.gauss(0, 2)

        # GPU utilization
        gpu_percent = 70 + random.gauss(0, 5) + (10 * workload_factor)
        gpu_percent = max(0, min(100, gpu_percent))

        # CPU utilization
        cpu_percent = 40 + random.gauss(0, 8) + (5 * workload_factor)
        cpu_percent = max(0, min(100, cpu_percent))

        # Memory
        memory_used = 4096 + i * 2 + random.gauss(0, 50)

        # Power (correlated with GPU)
        power = 15 + (gpu_percent / 100) * 120 + random.gauss(0, 3)

        # Temperature (correlated with power, with inertia)
        temp = 45 + (power - 15) / 135 * 30 + random.gauss(0, 1)

        # Throughput (inverse of latency)
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

    def percentile(data, p):
        sorted_data = sorted(data)
        idx = int(p / 100 * (len(sorted_data) - 1))
        return sorted_data[idx]

    return {
        "collector_name": "SyntheticCollector (Demo)",
        "start_time": base_time,
        "end_time": base_time + num_samples * 0.6,
        "sample_count": num_samples,
        "samples": samples,
        "summary": {
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
                "mean_percent": round(
                    sum(s["metrics"]["cpu_percent"] for s in steady_samples)
                    / len(steady_samples),
                    1,
                ),
                "max_percent": round(
                    max(s["metrics"]["cpu_percent"] for s in steady_samples), 1
                ),
            },
            "gpu": {
                "mean_percent": round(
                    sum(s["metrics"]["gpu_percent"] for s in steady_samples)
                    / len(steady_samples),
                    1,
                ),
                "max_percent": round(
                    max(s["metrics"]["gpu_percent"] for s in steady_samples), 1
                ),
            },
            "memory": {
                "mean_mb": round(
                    sum(s["metrics"]["memory_used_mb"] for s in steady_samples)
                    / len(steady_samples),
                    0,
                ),
                "max_mb": round(
                    max(s["metrics"]["memory_used_mb"] for s in steady_samples), 0
                ),
                "min_mb": round(
                    min(s["metrics"]["memory_used_mb"] for s in steady_samples), 0
                ),
            },
            "power": {
                "mean_w": round(
                    sum(s["metrics"]["power_w"] for s in steady_samples)
                    / len(steady_samples),
                    1,
                ),
                "max_w": round(max(s["metrics"]["power_w"] for s in steady_samples), 1),
            },
            "temperature": {
                "mean_c": round(
                    sum(s["metrics"]["temperature_c"] for s in steady_samples)
                    / len(steady_samples),
                    1,
                ),
                "max_c": round(
                    max(s["metrics"]["temperature_c"] for s in steady_samples), 1
                ),
            },
            "throughput": {
                "mean_fps": round(
                    sum(s["metrics"]["throughput_fps"] for s in steady_samples)
                    / len(steady_samples),
                    1,
                ),
                "min_fps": round(
                    min(s["metrics"]["throughput_fps"] for s in steady_samples), 1
                ),
            },
        },
        "config": {
            "warmup_samples": warmup_samples,
            "base_latency_ms": 25.0,
            "workload_pattern": "ramp",
        },
    }


def samples_to_dataframe(samples: List[Dict]) -> pd.DataFrame:
    """Convert samples list to pandas DataFrame.

    Args:
        samples: List of sample dictionaries

    Returns:
        DataFrame with flattened metrics
    """
    rows = []
    for sample in samples:
        row = {"timestamp": sample["timestamp"]}
        row.update(sample.get("metrics", {}))
        if "metadata" in sample:
            row.update({f"meta_{k}": v for k, v in sample["metadata"].items()})
        rows.append(row)

    df = pd.DataFrame(rows)

    # Convert timestamp to datetime
    if "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        df["elapsed_seconds"] = df["timestamp"] - df["timestamp"].min()

    return df


def safe_get(d: Dict, *keys, default=None):
    """Safely get nested dictionary value.

    Args:
        d: Dictionary to search
        *keys: Keys to traverse
        default: Default value if key not found

    Returns:
        Value at nested key or default
    """
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d if d is not None else default


def get_platform_metadata() -> Dict[str, Any]:
    """Get device name, CPU, GPU, SoC, power mode for display."""
    meta = {
        "device_name": "Unknown",
        "cpu": "Unknown",
        "gpu": "Unknown",
        "soc": "N/A",
        "power_mode": "N/A",
    }
    try:
        import platform as plat

        meta["cpu"] = plat.processor() or plat.machine() or "Unknown"
    except Exception:
        pass
    try:
        from trackiq.platform import query_nvidia_smi, get_memory_metrics

        name_out = query_nvidia_smi(["name"], timeout=2)
        if name_out:
            meta["gpu"] = name_out.strip()
            meta["device_name"] = name_out.strip()
        mem = get_memory_metrics(timeout=2)
        if mem:
            meta["power_mode"] = f"GPU mem: {mem.get('gpu_memory_percent', 0):.0f}%"
    except Exception:
        pass
    return meta


def get_detected_devices() -> List[Dict[str, Any]]:
    """Get all detected devices for UI (Phase 5)."""
    try:
        from trackiq.platform import get_all_devices

        devices = get_all_devices()
        return [d.to_dict() for d in devices]
    except Exception:
        return []


def run_auto_benchmarks_ui(
    duration_seconds: int,
    precisions: List[str],
    batch_sizes: List[int],
    max_configs_per_device: int = 6,
) -> List[Dict[str, Any]]:
    """Run auto benchmarks from UI and return list of result dicts (Phase 5)."""
    try:
        from autoperfpy.device_config import (
            get_devices_and_configs_auto,
            enumerate_inference_configs,
        )
        from trackiq.platform import get_all_devices
        from autoperfpy.auto_runner import run_auto_benchmarks

        devices = get_all_devices()
        if not devices:
            return []
        pairs = enumerate_inference_configs(
            devices,
            precisions=precisions,
            batch_sizes=batch_sizes,
            max_configs_per_device=max_configs_per_device,
        )
        return run_auto_benchmarks(
            pairs,
            duration_seconds=float(duration_seconds),
            sample_interval_seconds=0.2,
            quiet=True,
        )
    except Exception:
        return []


def run_benchmark_from_ui(
    duration_seconds: int, device: str, precision: str
) -> Optional[Dict[str, Any]]:
    """Run benchmark from UI (Phase 5 manual: uses detected device + inference config)."""
    try:
        from trackiq.platform import get_all_devices
        from autoperfpy.device_config import InferenceConfig
        from autoperfpy.auto_runner import run_single_benchmark
    except ImportError:
        return _run_synthetic_fallback_ui(duration_seconds, device, precision)
    devices = get_all_devices()
    device_id = (device or "cpu_0").strip().lower()
    target = None
    if device_id.isdigit():
        idx = int(device_id)
        for d in devices:
            if (
                getattr(d, "device_type", "") == "nvidia_gpu"
                and getattr(d, "index", -1) == idx
            ):
                target = d
                break
        if not target:
            for d in devices:
                if getattr(d, "index", -1) == idx:
                    target = d
                    break
    else:
        for d in devices:
            if getattr(d, "device_id", "") == device_id:
                target = d
                break
    if not target:
        target = devices[0] if devices else None
    if not target:
        return _run_synthetic_fallback_ui(duration_seconds, device, precision)
    config = InferenceConfig(
        precision=precision or "fp32",
        batch_size=1,
        accelerator=target.device_id,
        streams=1,
        warmup_runs=5,
        iterations=100,
    )
    result = run_single_benchmark(
        target,
        config,
        duration_seconds=float(duration_seconds),
        sample_interval_seconds=0.2,
        quiet=True,
    )
    return result


def _run_synthetic_fallback_ui(
    duration_seconds: int, device: str, precision: str
) -> Optional[Dict[str, Any]]:
    """Fallback: run synthetic benchmark from UI when Phase 5 runner unavailable."""
    try:
        from trackiq.collectors import SyntheticCollector
        from trackiq.runner import BenchmarkRunner
    except ImportError:
        return None
    config = {"warmup_samples": 5, "seed": 42}
    collector = SyntheticCollector(config=config)
    runner = BenchmarkRunner(
        collector,
        duration_seconds=float(duration_seconds),
        sample_interval_seconds=0.2,
        quiet=True,
    )
    export = runner.run()
    data = export.to_dict()
    data["platform_metadata"] = get_platform_metadata()
    data["inference_config"] = {"device": device, "precision": precision}
    data["run_label"] = f"{device}_{precision}_bs1"
    return data


def render_summary_metrics(data: Dict[str, Any]):
    """Render summary metrics cards.

    Args:
        data: Collector export data
    """
    summary = data.get("summary", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Samples",
            summary.get("sample_count", "N/A"),
            delta=f"-{summary.get('warmup_samples', 0)} warmup",
        )

    with col2:
        latency = summary.get("latency", {})
        p99 = latency.get("p99_ms", "N/A")
        mean = latency.get("mean_ms", 0)
        if isinstance(p99, (int, float)) and isinstance(mean, (int, float)):
            delta = f"{((p99 - mean) / mean * 100):.1f}% vs mean" if mean > 0 else None
        else:
            delta = None
        st.metric("P99 Latency", f"{p99} ms" if p99 != "N/A" else "N/A", delta=delta)

    with col3:
        throughput = summary.get("throughput", {})
        fps = throughput.get("mean_fps", "N/A")
        st.metric("Throughput", f"{fps} FPS" if fps != "N/A" else "N/A")

    with col4:
        power = summary.get("power", {})
        mean_power = power.get("mean_w", "N/A")
        st.metric("Avg Power", f"{mean_power} W" if mean_power != "N/A" else "N/A")


def render_latency_analysis(df: pd.DataFrame, summary: Dict[str, Any]):
    """Render latency analysis section.

    Args:
        df: DataFrame with sample data
        summary: Summary statistics dictionary
    """
    st.subheader("Latency Analysis")

    if "latency_ms" not in df.columns:
        st.warning("No latency data available")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Latency timeline
        fig = px.line(
            df,
            x="elapsed_seconds",
            y="latency_ms",
            color="is_warmup" if "is_warmup" in df.columns else None,
            title="Latency Over Time",
            labels={"elapsed_seconds": "Time (seconds)", "latency_ms": "Latency (ms)"},
        )
        fig.update_layout(showlegend=True if "is_warmup" in df.columns else False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Latency distribution
        fig = px.histogram(
            df[~df.get("is_warmup", False)] if "is_warmup" in df.columns else df,
            x="latency_ms",
            nbins=50,
            title="Latency Distribution (excluding warmup)",
            labels={"latency_ms": "Latency (ms)", "count": "Frequency"},
        )

        # Add percentile lines
        latency = summary.get("latency", {})
        for pct, color in [
            ("p50_ms", "green"),
            ("p95_ms", "orange"),
            ("p99_ms", "red"),
        ]:
            val = latency.get(pct)
            if val:
                fig.add_vline(
                    x=val,
                    line_dash="dash",
                    line_color=color,
                    annotation_text=f"{pct.replace('_ms', '')}: {val:.1f}ms",
                )

        st.plotly_chart(fig, use_container_width=True)

    # Percentile breakdown
    latency = summary.get("latency", {})
    if latency:
        st.markdown("**Percentile Breakdown**")
        pcol1, pcol2, pcol3, pcol4, pcol5, pcol6 = st.columns(6)
        pcol1.metric("Min", f"{latency.get('min_ms', 'N/A')} ms")
        pcol2.metric("P50", f"{latency.get('p50_ms', 'N/A')} ms")
        pcol3.metric("Mean", f"{latency.get('mean_ms', 'N/A')} ms")
        pcol4.metric("P95", f"{latency.get('p95_ms', 'N/A')} ms")
        pcol5.metric("P99", f"{latency.get('p99_ms', 'N/A')} ms")
        pcol6.metric("Max", f"{latency.get('max_ms', 'N/A')} ms")


def render_utilization_analysis(df: pd.DataFrame, summary: Dict[str, Any]):
    """Render CPU/GPU utilization analysis.

    Args:
        df: DataFrame with sample data
        summary: Summary statistics dictionary
    """
    st.subheader("Resource Utilization")

    has_cpu = "cpu_percent" in df.columns
    has_gpu = "gpu_percent" in df.columns

    if not has_cpu and not has_gpu:
        st.warning("No utilization data available")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Utilization timeline
        fig = go.Figure()

        if has_cpu:
            fig.add_trace(
                go.Scatter(
                    x=df["elapsed_seconds"],
                    y=df["cpu_percent"],
                    mode="lines",
                    name="CPU",
                    line=dict(color="blue"),
                )
            )

        if has_gpu:
            fig.add_trace(
                go.Scatter(
                    x=df["elapsed_seconds"],
                    y=df["gpu_percent"],
                    mode="lines",
                    name="GPU",
                    line=dict(color="green"),
                )
            )

        fig.update_layout(
            title="CPU/GPU Utilization Over Time",
            xaxis_title="Time (seconds)",
            yaxis_title="Utilization (%)",
            yaxis=dict(range=[0, 105]),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Utilization summary bar chart
        cpu_data = summary.get("cpu", {})
        gpu_data = summary.get("gpu", {})

        categories = []
        means = []
        maxes = []

        if cpu_data:
            categories.append("CPU")
            means.append(cpu_data.get("mean_percent", 0))
            maxes.append(cpu_data.get("max_percent", 0))

        if gpu_data:
            categories.append("GPU")
            means.append(gpu_data.get("mean_percent", 0))
            maxes.append(gpu_data.get("max_percent", 0))

        fig = go.Figure(
            data=[
                go.Bar(name="Mean", x=categories, y=means, marker_color="steelblue"),
                go.Bar(name="Max", x=categories, y=maxes, marker_color="darkblue"),
            ]
        )
        fig.update_layout(
            title="Utilization Summary",
            barmode="group",
            yaxis_title="Utilization (%)",
            yaxis=dict(range=[0, 105]),
        )
        st.plotly_chart(fig, use_container_width=True)


def render_power_analysis(df: pd.DataFrame, summary: Dict[str, Any]):
    """Render power and thermal analysis.

    Args:
        df: DataFrame with sample data
        summary: Summary statistics dictionary
    """
    st.subheader("Power & Thermal")

    has_power = "power_w" in df.columns
    has_temp = "temperature_c" in df.columns

    if not has_power and not has_temp:
        st.warning("No power/thermal data available")
        return

    col1, col2 = st.columns(2)

    with col1:
        if has_power:
            fig = px.line(
                df,
                x="elapsed_seconds",
                y="power_w",
                title="Power Consumption Over Time",
                labels={"elapsed_seconds": "Time (seconds)", "power_w": "Power (W)"},
            )
            fig.update_traces(line_color="orange")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No power data available")

    with col2:
        if has_temp:
            fig = px.line(
                df,
                x="elapsed_seconds",
                y="temperature_c",
                title="Temperature Over Time",
                labels={
                    "elapsed_seconds": "Time (seconds)",
                    "temperature_c": "Temperature (C)",
                },
            )
            fig.update_traces(line_color="red")

            # Add thermal threshold line
            fig.add_hline(
                y=85,
                line_dash="dash",
                line_color="darkred",
                annotation_text="Throttle Threshold",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No temperature data available")

    # Summary metrics
    power = summary.get("power", {})
    temp = summary.get("temperature", {})

    if power or temp:
        st.markdown("**Power & Thermal Summary**")
        pcol1, pcol2, pcol3, pcol4 = st.columns(4)

        if power:
            pcol1.metric("Mean Power", f"{power.get('mean_w', 'N/A')} W")
            pcol2.metric("Max Power", f"{power.get('max_w', 'N/A')} W")

        if temp:
            pcol3.metric("Mean Temp", f"{temp.get('mean_c', 'N/A')} C")
            pcol4.metric("Max Temp", f"{temp.get('max_c', 'N/A')} C")


def render_memory_analysis(df: pd.DataFrame, summary: Dict[str, Any]):
    """Render memory usage analysis.

    Args:
        df: DataFrame with sample data
        summary: Summary statistics dictionary
    """
    st.subheader("Memory Usage")

    if "memory_used_mb" not in df.columns:
        st.warning("No memory data available")
        return

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(
            df,
            x="elapsed_seconds",
            y="memory_used_mb",
            title="Memory Usage Over Time",
            labels={
                "elapsed_seconds": "Time (seconds)",
                "memory_used_mb": "Memory Used (MB)",
            },
        )
        fig.update_traces(line_color="purple")

        # Add total memory line if available
        if "memory_total_mb" in df.columns:
            total_mem = df["memory_total_mb"].iloc[0]
            fig.add_hline(
                y=total_mem,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Total: {total_mem} MB",
            )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        memory = summary.get("memory", {})
        if memory:
            # Memory usage gauge
            mean_mb = memory.get("mean_mb", 0)
            max_mb = memory.get("max_mb", 0)
            total_mb = (
                df["memory_total_mb"].iloc[0]
                if "memory_total_mb" in df.columns
                else 16384
            )

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=mean_mb,
                    delta={
                        "reference": max_mb,
                        "relative": False,
                        "valueformat": ".0f",
                    },
                    title={"text": "Mean Memory Usage (MB)"},
                    gauge={
                        "axis": {"range": [0, total_mb]},
                        "bar": {"color": "purple"},
                        "steps": [
                            {"range": [0, total_mb * 0.7], "color": "lightgray"},
                            {
                                "range": [total_mb * 0.7, total_mb * 0.9],
                                "color": "lightyellow",
                            },
                            {
                                "range": [total_mb * 0.9, total_mb],
                                "color": "lightcoral",
                            },
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": max_mb,
                        },
                    },
                )
            )
            st.plotly_chart(fig, use_container_width=True)


def render_throughput_analysis(df: pd.DataFrame, summary: Dict[str, Any]):
    """Render throughput analysis.

    Args:
        df: DataFrame with sample data
        summary: Summary statistics dictionary
    """
    st.subheader("Throughput Analysis")

    if "throughput_fps" not in df.columns:
        st.warning("No throughput data available")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Filter out warmup if available
        plot_df = df[~df.get("is_warmup", False)] if "is_warmup" in df.columns else df

        fig = px.line(
            plot_df,
            x="elapsed_seconds",
            y="throughput_fps",
            title="Throughput Over Time (excluding warmup)",
            labels={
                "elapsed_seconds": "Time (seconds)",
                "throughput_fps": "Throughput (FPS)",
            },
        )
        fig.update_traces(line_color="teal")

        # Add mean line
        throughput = summary.get("throughput", {})
        mean_fps = throughput.get("mean_fps")
        if mean_fps:
            fig.add_hline(
                y=mean_fps,
                line_dash="dash",
                line_color="darkgreen",
                annotation_text=f"Mean: {mean_fps:.1f} FPS",
            )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Throughput statistics
        throughput = summary.get("throughput", {})
        if throughput:
            st.markdown("**Throughput Statistics**")
            st.metric("Mean Throughput", f"{throughput.get('mean_fps', 'N/A')} FPS")
            st.metric("Min Throughput", f"{throughput.get('min_fps', 'N/A')} FPS")

            # Calculate efficiency if power is available
            power = summary.get("power", {})
            if power and power.get("mean_w"):
                efficiency = throughput.get("mean_fps", 0) / power.get("mean_w", 1)
                st.metric("Efficiency", f"{efficiency:.2f} FPS/W")


def render_multi_run_comparison(runs: List[Dict[str, Any]]):
    """Render multi-run comparison view with platform_metadata and inference_config (Phase 5).

    Args:
        runs: List of collector export data dictionaries
    """
    st.subheader("Multi-Run Comparison")

    if len(runs) < 2:
        st.info(
            "Upload multiple run files or run Automatic benchmark to enable comparison"
        )
        return

    # Run labels: run_label > collector_name > Run N
    run_names = [
        r.get("run_label") or r.get("collector_name") or f"Run {i+1}"
        for i, r in enumerate(runs)
    ]

    # Latency comparison
    col1, col2 = st.columns(2)

    with col1:
        latency_data = []
        for i, run in enumerate(runs):
            latency = run.get("summary", {}).get("latency", {})
            for pct in ["p50_ms", "p95_ms", "p99_ms"]:
                val = latency.get(pct)
                if val is not None:
                    latency_data.append(
                        {
                            "Run": run_names[i],
                            "Percentile": pct.replace("_ms", "").upper(),
                            "Latency (ms)": val,
                        }
                    )

        if latency_data:
            df = pd.DataFrame(latency_data)
            fig = px.bar(
                df,
                x="Run",
                y="Latency (ms)",
                color="Percentile",
                barmode="group",
                title="Latency Comparison Across Runs",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        throughput_data = []
        for i, run in enumerate(runs):
            throughput = run.get("summary", {}).get("throughput", {})
            mean_fps = throughput.get("mean_fps")
            if mean_fps is not None:
                throughput_data.append(
                    {
                        "Run": run_names[i],
                        "Throughput (FPS)": mean_fps,
                    }
                )

        if throughput_data:
            df = pd.DataFrame(throughput_data)
            fig = px.bar(
                df,
                x="Run",
                y="Throughput (FPS)",
                title="Throughput Comparison Across Runs",
                color="Run",
            )
            st.plotly_chart(fig, use_container_width=True)

    # Comparison table with platform_metadata and inference_config
    st.markdown("**Summary Comparison**")
    comparison_data = []
    for i, run in enumerate(runs):
        summary = run.get("summary", {})
        pm = run.get("platform_metadata") or {}
        inf = run.get("inference_config") or {}
        comparison_data.append(
            {
                "Run": run_names[i],
                "Device": pm.get("device_name") or inf.get("accelerator", "N/A"),
                "Precision": inf.get("precision", "N/A"),
                "Batch": inf.get("batch_size", "N/A"),
                "Samples": summary.get("sample_count", "N/A"),
                "Duration (s)": round(summary.get("duration_seconds", 0), 1),
                "P99 Latency (ms)": summary.get("latency", {}).get("p99_ms", "N/A"),
                "Mean Throughput (FPS)": summary.get("throughput", {}).get(
                    "mean_fps", "N/A"
                ),
                "Mean Power (W)": summary.get("power", {}).get("mean_w", "N/A"),
                "Max Temp (C)": summary.get("temperature", {}).get("max_c", "N/A"),
            }
        )

    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

    # Per-run metadata expanders (side-by-side)
    st.markdown("**Run metadata**")
    n = len(runs)
    cols = st.columns(min(n, 4))
    for i, run in enumerate(runs):
        with cols[i % len(cols)]:
            with st.expander(run_names[i], expanded=False):
                pm = run.get("platform_metadata") or {}
                inf = run.get("inference_config") or {}
                st.json({"platform_metadata": pm, "inference_config": inf})


def render_raw_data_view(df: pd.DataFrame):
    """Render raw data table view.

    Args:
        df: DataFrame with sample data
    """
    st.subheader("Raw Data")

    # Column selector
    all_cols = df.columns.tolist()
    default_cols = [
        c
        for c in [
            "elapsed_seconds",
            "latency_ms",
            "cpu_percent",
            "gpu_percent",
            "memory_used_mb",
            "power_w",
            "temperature_c",
            "throughput_fps",
        ]
        if c in all_cols
    ]

    selected_cols = st.multiselect(
        "Select columns to display",
        options=all_cols,
        default=default_cols,
    )

    if selected_cols:
        st.dataframe(df[selected_cols], use_container_width=True, height=400)

    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name="autoperfpy_export.csv",
        mime="text/csv",
    )


def main():
    """Main Streamlit application."""

    st.title("AutoPerfPy Dashboard")
    st.markdown("Interactive visualization for performance analysis metrics")

    # Sidebar
    with st.sidebar:
        st.header("Data Source")

        data_source = st.radio(
            "Select data source",
            options=["Upload File", "Demo Data", "Run Benchmark"],
            index=1,  # Default to demo
        )

        uploaded_files = []
        data_list = []

        if data_source == "Upload File":
            uploaded_files = st.file_uploader(
                "Upload collector output (JSON/CSV)",
                type=["json", "csv"],
                accept_multiple_files=True,
                help="Upload JSON export from collectors or CSV benchmark data",
            )

            if uploaded_files:
                for uploaded_file in uploaded_files:
                    try:
                        if uploaded_file.name.endswith(".json"):
                            content = json.load(uploaded_file)
                            data_list.append(content)
                        elif uploaded_file.name.endswith(".csv"):
                            df = pd.read_csv(uploaded_file)
                            # Convert CSV to collector-like format
                            data_list.append(
                                {
                                    "collector_name": uploaded_file.name,
                                    "samples": df.to_dict("records"),
                                    "summary": {},
                                }
                            )
                    except Exception as e:
                        st.error(f"Error loading {uploaded_file.name}: {e}")
        elif data_source == "Run Benchmark":
            st.subheader("Benchmark mode")
            benchmark_mode = st.radio(
                "Mode",
                options=["Automatic", "Manual"],
                index=0,
                help="Automatic: run on all detected devices and configs. Manual: choose device and config.",
            )
            if benchmark_mode == "Automatic":
                detected = get_detected_devices()
                if detected:
                    st.markdown("**Detected devices**")
                    for d in detected:
                        st.caption(
                            f"• {d.get('device_id', '?')}: {d.get('device_name', 'Unknown')}"
                        )
                else:
                    st.caption("No devices detected; synthetic will be used.")
                duration_ui = st.number_input(
                    "Duration (seconds)",
                    min_value=1,
                    max_value=120,
                    value=10,
                    key="auto_dur",
                )
                precisions_ui = st.multiselect(
                    "Precisions",
                    options=["fp32", "fp16", "int8"],
                    default=["fp32", "fp16"],
                    key="auto_prec",
                )
                batch_sizes_ui = st.multiselect(
                    "Batch sizes",
                    options=[1, 2, 4, 8, 16],
                    default=[1, 4],
                    key="auto_bs",
                )
                max_cfg = st.number_input(
                    "Max configs per device",
                    min_value=1,
                    max_value=20,
                    value=6,
                    key="auto_max",
                )
                if st.button("Run benchmark (all devices & configs)"):
                    with st.spinner("Running benchmarks on all devices..."):
                        results = run_auto_benchmarks_ui(
                            duration_seconds=duration_ui,
                            precisions=precisions_ui or ["fp32"],
                            batch_sizes=batch_sizes_ui or [1],
                            max_configs_per_device=max_cfg,
                        )
                    if results:
                        if "data_list" not in st.session_state:
                            st.session_state["data_list"] = []
                        st.session_state["data_list"] = results
                        st.success(
                            f"Completed {len(results)} runs. Results loaded below."
                        )
                        if hasattr(st, "rerun"):
                            st.rerun()
                        else:
                            st.experimental_rerun()
                    else:
                        st.warning("No results. Try Manual mode or check devices.")
                if "data_list" in st.session_state and st.session_state["data_list"]:
                    data_list = st.session_state["data_list"]
                else:
                    st.info("Click **Run benchmark (all devices & configs)** to start.")
            else:
                st.subheader("Manual settings")
                device_ui = st.text_input(
                    "Device (e.g. nvidia_0, cpu_0, 0)",
                    value="cpu_0",
                    help="Device ID or GPU index",
                )
                precision_ui = st.selectbox(
                    "Inference precision",
                    options=["fp32", "fp16", "int8"],
                    index=0,
                    help="Precision for inference (fp32, fp16, int8)",
                )
                duration_ui = st.number_input(
                    "Duration (seconds)",
                    min_value=1,
                    max_value=300,
                    value=10,
                    key="man_dur",
                )
                if st.button("Run benchmark"):
                    with st.spinner("Running benchmark..."):
                        run_data = run_benchmark_from_ui(
                            duration_ui, device_ui, precision_ui
                        )
                    if run_data:
                        if "data_list" not in st.session_state:
                            st.session_state["data_list"] = []
                        st.session_state["data_list"] = [run_data]
                        st.success("Benchmark complete. Results loaded below.")
                        if hasattr(st, "rerun"):
                            st.rerun()
                        else:
                            st.experimental_rerun()
                if "data_list" in st.session_state and st.session_state["data_list"]:
                    data_list = st.session_state["data_list"]
                else:
                    st.info(
                        "Click **Run benchmark** to run a short benchmark and load results."
                    )
        else:
            st.info("Using synthetic demo data")
            data_list = [generate_synthetic_demo_data()]

        st.divider()

        # View options
        st.header("View Options")
        show_warmup = st.checkbox("Include warmup samples", value=False)
        show_raw_data = st.checkbox("Show raw data table", value=False)

    # Main content
    if not data_list:
        if data_source == "Run Benchmark":
            st.info(
                "Configure benchmark settings in the sidebar and click **Run benchmark** to start."
            )
        else:
            st.info("Please upload a data file or select Demo Data to get started")

        st.markdown(
            """
        ### Supported Data Formats

        **JSON (Collector Export)**
        ```json
        {
            "collector_name": "SyntheticCollector",
            "samples": [...],
            "summary": {...}
        }
        ```

        **CSV (Benchmark Data)**
        ```
        timestamp,latency_ms,cpu_percent,gpu_percent,...
        ```

        ### Getting Started

        1. Run a benchmark from the UI (sidebar → Run Benchmark) or CLI:
           ```bash
           autoperfpy run --profile ci_smoke --export results.json
           ```

        2. Upload the results file or use Demo Data to visualize
        """
        )
        return

    # Use first dataset as primary
    data = data_list[0]

    # Convert samples to DataFrame
    samples = data.get("samples", [])
    if samples:
        df = samples_to_dataframe(samples)

        # Filter warmup if requested
        if not show_warmup and "is_warmup" in df.columns:
            display_df = df[~df["is_warmup"]]
        else:
            display_df = df
    else:
        st.warning("No sample data found in the uploaded file")
        return

    summary = data.get("summary", {})

    # Display collector info
    st.markdown(f"**Collector:** {data.get('collector_name', 'Unknown')}")
    if summary.get("duration_seconds"):
        st.markdown(
            f"**Duration:** {summary['duration_seconds']:.1f} seconds | "
            f"**Samples:** {summary.get('sample_count', len(samples))}"
        )

    # Platform metadata (device name, CPU, GPU, SoC, power mode) and inference_config
    platform_meta = data.get("platform_metadata") or {}
    inference_cfg = data.get("inference_config") or {}
    run_label = data.get("run_label", "")
    if platform_meta or inference_cfg or run_label:
        with st.expander("Platform & run metadata", expanded=True):
            if run_label:
                st.markdown(f"**Run:** `{run_label}`")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Device**")
                st.text(
                    platform_meta.get("device_name")
                    or inference_cfg.get("accelerator")
                    or "N/A"
                )
                st.markdown("**GPU**")
                st.text(
                    platform_meta.get("gpu_model") or platform_meta.get("gpu", "N/A")
                )
            with c2:
                st.markdown("**CPU**")
                st.text(
                    platform_meta.get("cpu_model") or platform_meta.get("cpu", "N/A")
                )
                st.markdown("**Precision**")
                st.text(inference_cfg.get("precision") or "N/A")
            with c3:
                st.markdown("**SoC / Power**")
                st.text(platform_meta.get("soc", "N/A"))
                st.text(platform_meta.get("power_mode", "N/A"))

    st.divider()

    # Summary metrics
    render_summary_metrics(data)

    st.divider()

    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Latency",
            "Utilization",
            "Power & Thermal",
            "Memory",
            "Throughput",
        ]
    )

    with tab1:
        render_latency_analysis(display_df, summary)

    with tab2:
        render_utilization_analysis(display_df, summary)

    with tab3:
        render_power_analysis(display_df, summary)

    with tab4:
        render_memory_analysis(display_df, summary)

    with tab5:
        render_throughput_analysis(display_df, summary)

    st.divider()

    # Multi-run comparison if multiple files uploaded
    if len(data_list) > 1:
        render_multi_run_comparison(data_list)
        st.divider()

    # Raw data view
    if show_raw_data:
        render_raw_data_view(df)

    # Footer
    st.divider()
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "AutoPerfPy Dashboard | Performance Analysis Toolkit"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
