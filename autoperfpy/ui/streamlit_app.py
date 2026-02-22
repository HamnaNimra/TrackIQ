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
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import streamlit as st

# Add parent directory to path for imports when running directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import pandas as pd

    from autoperfpy.reports import charts as shared_charts

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

MAX_UI_AUTO_RUNS = 12


def _apply_ui_style() -> None:
    """Apply lightweight UX polish for readability and hierarchy."""
    st.markdown(
        """
        <style>
        .stApp {
            font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        }
        .ap-hero {
            border: 1px solid rgba(59,130,246,0.22);
            background: linear-gradient(135deg, rgba(59,130,246,0.12), rgba(16,185,129,0.10));
            border-radius: 14px;
            padding: 14px 16px;
            margin-bottom: 14px;
        }
        .ap-hero h2 {
            margin: 0 0 4px 0;
            font-size: 1.28rem;
        }
        .ap-hero p {
            margin: 0;
            color: #4b5563;
            font-size: 0.95rem;
        }
        [data-testid="stMetric"] {
            border: 1px solid rgba(148,163,184,0.24);
            border-radius: 12px;
            padding: 8px 10px;
            background: rgba(15,23,42,0.02);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdown"] p {
            line-height: 1.35;
        }
        button[kind="primary"] {
            border-radius: 10px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_page_intro() -> None:
    """Render top-level UX guidance."""
    st.markdown(
        """
        <div class="ap-hero">
          <h2>AutoPerfPy Performance Studio</h2>
          <p>Choose a data source in the sidebar, inspect charts in tabs, then export reports or run analyses.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Quick Start", expanded=False):
        st.markdown(
            "1. Pick a data source (`Upload File`, `Demo Data`, or `Run Benchmark`).\n"
            "2. Review `Overview`, then drill into `Latency`, `Utilization`, and `Power & Thermal`.\n"
            "3. Use `Generate HTML Report` and `Run analysis` for sharable outputs."
        )


def _normalize_max_configs_per_device(value: int | None) -> int | None:
    """Normalize UI max-config value: non-positive means unlimited."""
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _cap_pairs_with_precision_coverage(
    pairs: list[tuple[Any, Any]],
    max_total_runs: int,
) -> list[tuple[Any, Any]]:
    """Cap run pairs while preserving precision breadth when possible.

    Picks one run per (device, precision) bucket first, then fills remaining
    slots in original order. This avoids dropping later precisions when the UI
    run cap is lower than the full cartesian set.
    """
    if max_total_runs <= 0 or len(pairs) <= max_total_runs:
        return pairs

    buckets: dict[tuple[str, str], list[tuple[Any, Any]]] = {}
    bucket_order: list[tuple[str, str]] = []
    for pair in pairs:
        device, config = pair
        device_id = str(getattr(device, "device_id", "unknown"))
        precision = str(getattr(config, "precision", "unknown")).lower()
        key = (device_id, precision)
        if key not in buckets:
            buckets[key] = []
            bucket_order.append(key)
        buckets[key].append(pair)

    selected: list[tuple[Any, Any]] = []
    selected_ids: set[int] = set()

    for key in bucket_order:
        if len(selected) >= max_total_runs:
            break
        candidates = buckets.get(key, [])
        if not candidates:
            continue
        item = candidates.pop(0)
        selected.append(item)
        selected_ids.add(id(item))

    if len(selected) < max_total_runs:
        for item in pairs:
            if len(selected) >= max_total_runs:
                break
            if id(item) in selected_ids:
                continue
            selected.append(item)

    return selected


def load_json_data(filepath: str) -> dict[str, Any] | None:
    """Load collector export data from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary with collector data or None if loading fails
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load JSON file: {e}")
        return None


def load_csv_data(filepath: str) -> pd.DataFrame | None:
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


def generate_synthetic_demo_data() -> dict[str, Any]:
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

    def _percentile_nearest(data, p):
        """Nearest-index percentile for demo summary (preserves existing behavior)."""
        if not data:
            return 0.0
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
                "p50_ms": round(_percentile_nearest(latencies, 50), 2),
                "p95_ms": round(_percentile_nearest(latencies, 95), 2),
                "p99_ms": round(_percentile_nearest(latencies, 99), 2),
            },
            "cpu": {
                "mean_percent": round(
                    sum(s["metrics"]["cpu_percent"] for s in steady_samples) / len(steady_samples),
                    1,
                ),
                "max_percent": round(max(s["metrics"]["cpu_percent"] for s in steady_samples), 1),
            },
            "gpu": {
                "mean_percent": round(
                    sum(s["metrics"]["gpu_percent"] for s in steady_samples) / len(steady_samples),
                    1,
                ),
                "max_percent": round(max(s["metrics"]["gpu_percent"] for s in steady_samples), 1),
            },
            "memory": {
                "mean_mb": round(
                    sum(s["metrics"]["memory_used_mb"] for s in steady_samples) / len(steady_samples),
                    0,
                ),
                "max_mb": round(max(s["metrics"]["memory_used_mb"] for s in steady_samples), 0),
                "min_mb": round(min(s["metrics"]["memory_used_mb"] for s in steady_samples), 0),
            },
            "power": {
                "mean_w": round(
                    sum(s["metrics"]["power_w"] for s in steady_samples) / len(steady_samples),
                    1,
                ),
                "max_w": round(max(s["metrics"]["power_w"] for s in steady_samples), 1),
            },
            "temperature": {
                "mean_c": round(
                    sum(s["metrics"]["temperature_c"] for s in steady_samples) / len(steady_samples),
                    1,
                ),
                "max_c": round(max(s["metrics"]["temperature_c"] for s in steady_samples), 1),
            },
            "throughput": {
                "mean_fps": round(
                    sum(s["metrics"]["throughput_fps"] for s in steady_samples) / len(steady_samples),
                    1,
                ),
                "min_fps": round(min(s["metrics"]["throughput_fps"] for s in steady_samples), 1),
            },
        },
        "config": {
            "warmup_samples": warmup_samples,
            "base_latency_ms": 25.0,
            "workload_pattern": "ramp",
        },
    }


def build_sample_data_list() -> list[dict[str, Any]]:
    """Build a one-click sample dataset for empty dashboard states."""
    return [generate_synthetic_demo_data()]


def build_run_overview_row(run: dict[str, Any]) -> dict[str, Any]:
    """Build a normalized run overview row for UI tables."""
    summary = run.get("summary", {}) if isinstance(run, dict) else {}
    platform_meta = run.get("platform_metadata", {}) if isinstance(run, dict) else {}
    inference_cfg = run.get("inference_config", {}) if isinstance(run, dict) else {}
    latency = summary.get("latency", {}) if isinstance(summary, dict) else {}
    throughput = summary.get("throughput", {}) if isinstance(summary, dict) else {}
    power = summary.get("power", {}) if isinstance(summary, dict) else {}
    return {
        "Run": run.get("run_label") or run.get("collector_name") or "Current Run",
        "Device": platform_meta.get("device_name") or inference_cfg.get("accelerator") or "Unknown",
        "Precision": inference_cfg.get("precision") or "N/A",
        "Batch Size": inference_cfg.get("batch_size") if inference_cfg.get("batch_size") is not None else "N/A",
        "Samples": summary.get("sample_count"),
        "Duration (s)": summary.get("duration_seconds"),
        "P99 Latency (ms)": latency.get("p99_ms") if latency else None,
        "Mean Throughput (FPS)": throughput.get("mean_fps") if throughput else None,
        "Mean Power (W)": power.get("mean_w") if power else None,
    }


def samples_to_dataframe(samples: list[dict]) -> pd.DataFrame:
    """Convert samples list to pandas DataFrame.

    Uses shared implementation from autoperfpy.reports.charts.

    Args:
        samples: List of sample dictionaries

    Returns:
        DataFrame with flattened metrics
    """
    df = shared_charts.samples_to_dataframe(samples)
    # Add datetime column for UI display
    if "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def get_platform_metadata() -> dict[str, Any]:
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
        from trackiq_core.hardware import get_memory_metrics, query_nvidia_smi

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


def get_detected_devices() -> list[dict[str, Any]]:
    """Get all detected devices for UI (Phase 5)."""
    try:
        from trackiq_core.hardware import get_all_devices

        devices = get_all_devices()
        return [d.to_dict() for d in devices]
    except Exception:
        return []


def result_to_csv_path(result: dict[str, Any]) -> str | None:
    """Convert run result dict to a temp CSV file path for analyzers. Returns path or None."""
    content = result_to_csv_content(result)
    if not content:
        return None
    fd, path = tempfile.mkstemp(suffix=".csv", prefix="autoperfpy_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        return path
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        return None


def result_to_csv_content(result: dict[str, Any]) -> str | None:
    """Convert run result dict to CSV string (same format as CLI --export-csv). Returns None if no samples."""
    samples = result.get("samples", [])
    if not samples:
        return None
    batch_size = result.get("inference_config", {}).get("batch_size", 1)
    rows = []
    for s in samples:
        ts = s.get("timestamp", 0)
        m = s.get("metrics", s) if isinstance(s, dict) else {}
        lat = m.get("latency_ms", 0)
        pwr = m.get("power_w", 0)
        throughput = (1000 / lat) if lat else 0
        rows.append((ts, "default", batch_size, lat, pwr, throughput))
    lines = ["timestamp,workload,batch_size,latency_ms,power_w,throughput"]
    for r in rows:
        lines.append(",".join(str(x) for x in r))
    return "\n".join(lines)


def run_auto_benchmarks_ui(
    duration_seconds: int,
    precisions: list[str],
    batch_sizes: list[int],
    max_configs_per_device: int | None = None,
    device_ids_filter: list[str] | None = None,
    progress_callback: Callable[..., None] | None = None,
    max_total_runs: int = MAX_UI_AUTO_RUNS,
) -> list[dict[str, Any]]:
    """Run auto benchmarks from UI and return list of result dicts (Phase 5)."""
    try:
        from autoperfpy.auto_runner import run_auto_benchmarks
        from autoperfpy.device_config import get_devices_and_configs_auto

        normalized_max_cfg = _normalize_max_configs_per_device(max_configs_per_device)
        pairs = get_devices_and_configs_auto(
            precisions=precisions,
            batch_sizes=batch_sizes,
            max_configs_per_device=normalized_max_cfg,
            device_ids_filter=device_ids_filter,
        )
        if not pairs:
            return []
        pairs = _cap_pairs_with_precision_coverage(pairs, max_total_runs)
        return run_auto_benchmarks(
            pairs,
            duration_seconds=float(duration_seconds),
            sample_interval_seconds=0.2,
            quiet=True,
            progress_callback=progress_callback,
            enable_power=False,
        )
    except Exception:
        return []


def estimate_auto_benchmark_plan_ui(
    duration_seconds: int,
    precisions: list[str],
    batch_sizes: list[int],
    max_configs_per_device: int | None = None,
    device_ids_filter: list[str] | None = None,
) -> dict[str, int]:
    """Estimate run count and wall-clock time for auto-benchmark UI selections."""
    try:
        from autoperfpy.device_config import get_devices_and_configs_auto

        normalized_max_cfg = _normalize_max_configs_per_device(max_configs_per_device)
        pairs = get_devices_and_configs_auto(
            precisions=precisions,
            batch_sizes=batch_sizes,
            max_configs_per_device=normalized_max_cfg,
            device_ids_filter=device_ids_filter,
        )
        planned_runs = len(pairs)
        capped_runs = min(planned_runs, MAX_UI_AUTO_RUNS)
        return {
            "planned_runs": planned_runs,
            "capped_runs": capped_runs,
            "estimated_seconds": int(capped_runs * max(1, int(duration_seconds))),
        }
    except Exception:
        return {"planned_runs": 0, "capped_runs": 0, "estimated_seconds": 0}


def run_benchmark_from_ui(duration_seconds: int, device: str, precision: str) -> dict[str, Any] | None:
    """Run benchmark from UI (Phase 5 manual: uses detected device + inference config)."""
    try:
        from autoperfpy.auto_runner import run_single_benchmark
        from autoperfpy.device_config import (
            DEFAULT_ITERATIONS,
            DEFAULT_WARMUP_RUNS,
            PRECISION_FP32,
            InferenceConfig,
            resolve_device,
        )
    except ImportError:
        return _run_synthetic_fallback_ui(duration_seconds, device, precision)
    target = resolve_device(device or "cpu_0")
    if not target:
        return _run_synthetic_fallback_ui(duration_seconds, device, precision)
    config = InferenceConfig(
        precision=precision or PRECISION_FP32,
        batch_size=1,
        accelerator=target.device_id,
        streams=1,
        warmup_runs=DEFAULT_WARMUP_RUNS,
        iterations=DEFAULT_ITERATIONS,
    )
    result = run_single_benchmark(
        target,
        config,
        duration_seconds=float(duration_seconds),
        sample_interval_seconds=0.2,
        quiet=True,
        enable_power=False,
    )
    return result


def _run_synthetic_fallback_ui(duration_seconds: int, device: str, precision: str) -> dict[str, Any] | None:
    """Fallback: run synthetic benchmark from UI when Phase 5 runner unavailable."""
    try:
        from autoperfpy.device_config import DEFAULT_WARMUP_RUNS
        from trackiq_core.collectors import SyntheticCollector
        from trackiq_core.runners import BenchmarkRunner
    except ImportError:
        return None
    config = {"warmup_samples": DEFAULT_WARMUP_RUNS, "seed": 42}
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


def _generate_report_directly(data: dict[str, Any] | list[dict[str, Any]], report_type: str = "HTML") -> tuple:
    """Generate HTML report directly without using CLI subprocess.

    The HTML report includes a 'Print / Save as PDF' button that users can
    click to save the report as PDF using their browser's print dialog.

    Args:
        data: Collector export data with samples and summary
        report_type: Report format (only "HTML" supported; use browser print for PDF)

    Returns:
        Tuple of (report_bytes, filename) or (None, None) on failure
    """
    from autoperfpy.reports import HTMLReportGenerator
    from autoperfpy.reports.report_builder import (
        populate_multi_run_html_report,
        populate_standard_html_report,
    )

    del report_type
    if isinstance(data, dict):
        samples = data.get("samples", [])
        if not samples:
            return None, None
    else:
        valid_runs = [run for run in data if isinstance(run, dict) and run]
        if not valid_runs:
            return None, None

    # Create report generator
    report = HTMLReportGenerator(
        title="Performance Analysis Report",
        author="AutoPerfPy",
        theme="light",
    )

    if isinstance(data, list):
        populate_multi_run_html_report(
            report,
            data,
            include_run_details=True,
            chart_engine="plotly",
        )
    else:
        populate_standard_html_report(
            report,
            data,
            chart_engine="plotly",
        )

    # Generate report to temp file
    out_dir = tempfile.mkdtemp(prefix="autoperfpy_report_")
    try:
        report_path = Path(out_dir) / "performance_report.html"
        report.generate_html(str(report_path))
        report_bytes = report_path.read_bytes()
        return report_bytes, "performance_report.html"
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def render_summary_metrics(data: dict[str, Any]):
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


def render_result_purpose_guide(summary: dict[str, Any]) -> None:
    """Render plain-English meaning and purpose of key result metrics."""
    rows: list[tuple[str, str, str]] = []
    if summary.get("sample_count") is not None:
        rows.append(
            (
                "sample_count",
                "Total collected sample points in this run.",
                "Higher counts improve statistical confidence.",
            )
        )
    if summary.get("duration_seconds") is not None:
        rows.append(
            (
                "duration_seconds",
                "Total benchmark measurement window.",
                "Ensures fair cross-run comparison.",
            )
        )

    metric_map = {
        "latency.p50_ms": ("Typical latency for most requests.", "Use as baseline responsiveness."),
        "latency.p95_ms": ("Tail latency for slower requests.", "Detects user-visible latency spikes."),
        "latency.p99_ms": ("Worst-case tail latency.", "Primary straggler/regression indicator."),
        "throughput.mean_fps": ("Average processing rate.", "Capacity planning and scaling metric."),
        "power.mean_w": ("Average power draw.", "Cost and efficiency baseline."),
        "temperature.max_c": ("Peak observed temperature.", "Thermal throttle risk signal."),
        "cpu.mean_percent": ("Average CPU utilization.", "Host-side bottleneck indicator."),
        "gpu.mean_percent": ("Average GPU utilization.", "Low values can indicate data/sync stalls."),
        "memory.mean_mb": ("Average memory footprint.", "Headroom check for larger models/batches."),
    }
    for group_name, metrics in summary.items():
        if not isinstance(metrics, dict):
            continue
        for metric_key in metrics:
            full_name = f"{group_name}.{metric_key}"
            if full_name in metric_map:
                description, purpose = metric_map[full_name]
                rows.append((full_name, description, purpose))

    if not rows:
        return

    with st.expander("What each result means", expanded=False):
        st.dataframe(
            pd.DataFrame(rows, columns=["Metric", "What It Means", "Why It Matters"]),
            width="stretch",
            hide_index=True,
        )


def render_multi_run_column_guide() -> None:
    """Explain the purpose of each column in multi-run comparison output."""
    rows = [
        ("Run", "Unique run identifier.", "Join key across outputs and reports."),
        ("Device", "Detected hardware target.", "Separates hardware effects from config effects."),
        ("Precision", "Numerical precision mode.", "Directly impacts speed/quality trade-offs."),
        ("Batch", "Configured batch size.", "Key lever for throughput and memory pressure."),
        ("Samples", "Collected sample count.", "Low counts can hide regressions."),
        ("Duration (s)", "Measured run window.", "Enforces fair comparison time windows."),
        ("P99 Latency (ms)", "99th percentile latency.", "Best indicator of tail regressions."),
        ("Mean Throughput (FPS)", "Average throughput.", "Capacity and scaling metric."),
        ("Mean Power (W)", "Average power draw.", "Cost and efficiency indicator."),
        ("Max Temp (C)", "Peak temperature.", "Thermal risk and throttling signal."),
    ]
    with st.expander("What each comparison column means", expanded=False):
        st.dataframe(
            pd.DataFrame(rows, columns=["Column", "What It Means", "Why It Matters"]),
            width="stretch",
            hide_index=True,
        )


def render_overview_analysis(data: dict[str, Any], df: pd.DataFrame, summary: dict[str, Any]):
    """Render first-pass overview with key metrics, one chart, and run configuration."""
    st.subheader("Overview")
    render_summary_metrics(data)
    st.markdown("**Primary Signal Trend**")

    fig = shared_charts.create_latency_timeline(df)
    if fig is None:
        fig = shared_charts.create_throughput_timeline(df, summary)
    if fig:
        st.plotly_chart(fig, width="stretch", key="overview_primary_signal_chart")
    else:
        st.info("No latency or throughput timeline data available.")

    st.markdown("**Run Configuration**")
    st.dataframe(pd.DataFrame([build_run_overview_row(data)]), width="stretch", hide_index=True)
    render_result_purpose_guide(summary)


def render_latency_analysis(df: pd.DataFrame, summary: dict[str, Any]):
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
        # Latency timeline (from shared charts)
        fig = shared_charts.create_latency_timeline(df)
        if fig:
            st.plotly_chart(fig, width="stretch", key="latency_timeline_chart")

    with col2:
        # Latency distribution (from shared charts)
        fig = shared_charts.create_latency_histogram(df, summary)
        if fig:
            st.plotly_chart(fig, width="stretch", key="latency_distribution_chart")

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


def render_utilization_analysis(df: pd.DataFrame, summary: dict[str, Any]):
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
        # Utilization timeline (from shared charts)
        fig = shared_charts.create_utilization_timeline(df)
        if fig:
            st.plotly_chart(fig, width="stretch", key="utilization_timeline_chart")

    with col2:
        # Utilization summary bar chart (from shared charts)
        fig = shared_charts.create_utilization_summary_bar(summary)
        if fig:
            st.plotly_chart(fig, width="stretch", key="utilization_summary_chart")


def render_power_analysis(df: pd.DataFrame, summary: dict[str, Any]):
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
        # Power timeline (from shared charts)
        fig = shared_charts.create_power_timeline(df)
        if fig:
            st.plotly_chart(fig, width="stretch", key="power_timeline_chart")
        else:
            st.info("No power data available")

    with col2:
        # Temperature timeline (from shared charts)
        fig = shared_charts.create_temperature_timeline(df)
        if fig:
            st.plotly_chart(fig, width="stretch", key="temperature_timeline_chart")
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


def render_memory_analysis(df: pd.DataFrame, summary: dict[str, Any]):
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
        # Memory timeline (from shared charts)
        fig = shared_charts.create_memory_timeline(df)
        if fig:
            st.plotly_chart(fig, width="stretch", key="memory_timeline_chart")

    with col2:
        # Memory gauge (from shared charts)
        total_mb = float(df["memory_total_mb"].iloc[0]) if "memory_total_mb" in df.columns and len(df) > 0 else None
        fig = shared_charts.create_memory_gauge(summary, total_mb)
        if fig:
            st.plotly_chart(fig, width="stretch", key="memory_gauge_chart")


def render_throughput_analysis(df: pd.DataFrame, summary: dict[str, Any]):
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
        # Throughput timeline (from shared charts)
        fig = shared_charts.create_throughput_timeline(df, summary)
        if fig:
            st.plotly_chart(fig, width="stretch", key="throughput_timeline_chart")

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


def render_multi_run_comparison(runs: list[dict[str, Any]]):
    """Render multi-run comparison view with platform_metadata and inference_config (Phase 5).

    Args:
        runs: List of collector export data dictionaries
    """
    st.subheader("Multi-Run Comparison")

    if len(runs) < 2:
        st.info("Upload multiple run files or run a benchmark (one or more devices) to enable comparison")
        return

    # Run labels: run_label > collector_name > Run N
    run_names = [r.get("run_label") or r.get("collector_name") or f"Run {i+1}" for i, r in enumerate(runs)]

    # Latency comparison
    col1, col2 = st.columns(2)

    with col1:
        # Latency comparison bar chart (from shared charts)
        fig = shared_charts.create_latency_comparison_bar(runs, run_names)
        if fig:
            st.plotly_chart(fig, width="stretch", key="multi_run_latency_comparison_chart")

    with col2:
        # Throughput comparison bar chart (from shared charts)
        fig = shared_charts.create_throughput_comparison_bar(runs, run_names)
        if fig:
            st.plotly_chart(fig, width="stretch", key="multi_run_throughput_comparison_chart")

    # Comparison table with platform_metadata and inference_config (use None for missing numerics so Arrow can serialize)
    st.markdown("**Summary Comparison**")
    comparison_data = []
    for i, run in enumerate(runs):
        summary = run.get("summary", {})
        pm = run.get("platform_metadata") or {}
        inf = run.get("inference_config") or {}
        lat = summary.get("latency", {})
        thr = summary.get("throughput", {})
        pwr = summary.get("power", {})
        temp = summary.get("temperature", {})
        comparison_data.append(
            {
                "Run": run_names[i],
                "Device": pm.get("device_name") or inf.get("accelerator") or "",
                "Precision": inf.get("precision") or "",
                "Batch": inf.get("batch_size"),
                "Samples": summary.get("sample_count"),
                "Duration (s)": (
                    round(summary.get("duration_seconds", 0), 1)
                    if summary.get("duration_seconds") is not None
                    else None
                ),
                "P99 Latency (ms)": lat.get("p99_ms") if lat else None,
                "Mean Throughput (FPS)": thr.get("mean_fps") if thr else None,
                "Mean Power (W)": pwr.get("mean_w") if pwr else None,
                "Max Temp (C)": temp.get("max_c") if temp else None,
            }
        )

    st.dataframe(pd.DataFrame(comparison_data), width="stretch")
    render_multi_run_column_guide()

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
        st.dataframe(df[selected_cols], width="stretch", height=400)

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
    _apply_ui_style()

    st.title("AutoPerfPy Dashboard")
    _render_page_intro()

    # Sidebar
    with st.sidebar:
        st.header("Data Source")
        st.caption("Workflow: select input -> inspect tabs -> export report.")

        data_source = st.radio(
            "Select data source",
            options=["Upload File", "Demo Data", "Run Benchmark"],
            index=1,  # Default to demo
            horizontal=True,
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
            detected = get_detected_devices()
            try:
                from autoperfpy.device_config import (
                    DEFAULT_BATCH_SIZES,
                    PRECISIONS,
                )
            except ImportError:
                PRECISIONS = ["fp32", "fp16", "bf16", "int8", "int4", "mixed"]
                DEFAULT_BATCH_SIZES = [1, 4, 8]

            if detected:
                device_options = [
                    (
                        d.get("device_id", "?"),
                        d.get("device_name") or d.get("device_id", "Unknown"),
                    )
                    for d in detected
                ]
                device_labels = [f"{did} ({name})" for did, name in device_options]
                all_device_ids = [did for did, _ in device_options]
                selected_labels = st.multiselect(
                    "Devices (select one or more)",
                    options=device_labels,
                    default=device_labels[:1],
                    key="bench_devices",
                    help="Select at least one device. Same as CLI: autoperfpy run --device <id>",
                )
                selected_device_ids = [all_device_ids[device_labels.index(label)] for label in selected_labels]
            else:
                st.caption("No devices detected; enter a device ID (e.g. cpu_0, nvidia_0).")
                device_ui = st.text_input(
                    "Device",
                    value="cpu_0",
                    key="bench_device_txt",
                    help="Device ID or GPU index when none detected",
                )
                selected_device_ids = [device_ui.strip()] if device_ui.strip() else []

            duration_ui = st.number_input(
                "Duration (seconds)",
                min_value=1,
                max_value=300,
                value=5,
                key="bench_dur",
            )
            precisions_ui = st.multiselect(
                "Precisions",
                options=PRECISIONS,
                default=PRECISIONS[:2],
                key="bench_prec",
            )
            batch_sizes_ui = st.multiselect(
                "Batch sizes",
                options=DEFAULT_BATCH_SIZES + [2, 16],
                default=DEFAULT_BATCH_SIZES,
                key="bench_bs",
            )
            max_cfg = st.number_input(
                "Max configs per device (0 = all)",
                min_value=0,
                max_value=200,
                value=0,
                key="bench_max",
                help=(
                    "Set 0 to run all selected precision x batch combinations per device. "
                    "Set a positive value to limit per-device configs."
                ),
            )

            plan = {"planned_runs": 0, "capped_runs": 0, "estimated_seconds": 0}
            if selected_device_ids:
                plan = estimate_auto_benchmark_plan_ui(
                    duration_seconds=int(duration_ui),
                    precisions=precisions_ui or PRECISIONS[:1],
                    batch_sizes=batch_sizes_ui or DEFAULT_BATCH_SIZES[:1],
                    max_configs_per_device=int(max_cfg),
                    device_ids_filter=selected_device_ids,
                )
                if plan["planned_runs"] > 0:
                    st.caption(
                        "Planned runs: "
                        f"{plan['planned_runs']} (running first {plan['capped_runs']}) | "
                        f"Estimated runtime: ~{plan['estimated_seconds']}s"
                    )
                    if plan["planned_runs"] > MAX_UI_AUTO_RUNS:
                        st.info(
                            f"UI run cap is {MAX_UI_AUTO_RUNS} to keep execution responsive. "
                            "Narrow devices/configs to target specific runs."
                        )

            if st.button("Run benchmark"):
                if not selected_device_ids:
                    st.warning("Select at least one device.")
                elif detected:
                    if plan["capped_runs"] == 0:
                        st.warning("No valid benchmark configurations for selected inputs.")
                        results = []
                    else:
                        progress = st.progress(0.0)
                        status = st.empty()
                        status.caption("Preparing runs...")

                        def _on_progress(i, total, device, config):
                            progress.progress(min(1.0, i / max(total, 1)))
                            status.caption(
                                f"Run {i}/{total}: {device.device_name} ({device.device_id}) "
                                f"precision={config.precision} batch={config.batch_size}"
                            )

                        with st.spinner("Running benchmarks..."):
                            results = run_auto_benchmarks_ui(
                                duration_seconds=duration_ui,
                                precisions=precisions_ui or PRECISIONS[:1],
                                batch_sizes=batch_sizes_ui or DEFAULT_BATCH_SIZES[:1],
                                max_configs_per_device=int(max_cfg),
                                device_ids_filter=selected_device_ids,
                                progress_callback=_on_progress,
                                max_total_runs=MAX_UI_AUTO_RUNS,
                            )
                        progress.empty()
                        status.empty()
                    if results:
                        if "data_list" not in st.session_state:
                            st.session_state["data_list"] = []
                        st.session_state["data_list"] = results
                        st.success(f"Completed {len(results)} runs. Results loaded below.")
                        if hasattr(st, "rerun"):
                            st.rerun()
                        else:
                            st.experimental_rerun()
                    else:
                        st.warning("No results. Check devices and config.")
                else:
                    with st.spinner("Running benchmark..."):
                        run_results: list[dict[str, Any]] = []
                        selected_precisions = precisions_ui or PRECISIONS[:1]
                        for precision in selected_precisions:
                            run_data = run_benchmark_from_ui(
                                duration_ui,
                                selected_device_ids[0],
                                precision,
                            )
                            if run_data:
                                run_results.append(run_data)
                    if run_results:
                        if "data_list" not in st.session_state:
                            st.session_state["data_list"] = []
                        st.session_state["data_list"] = run_results
                        st.success(f"Completed {len(run_results)} run(s). Results loaded below.")
                        if hasattr(st, "rerun"):
                            st.rerun()
                        else:
                            st.experimental_rerun()
                    else:
                        st.warning("No results. Check device ID.")

            if "data_list" in st.session_state and st.session_state["data_list"]:
                data_list = st.session_state["data_list"]
            else:
                st.info("Select one or more devices, set options, then click **Run benchmark**.")
        else:
            st.info("Using synthetic demo data")
            data_list = [generate_synthetic_demo_data()]

        st.divider()

        # Reports & Analyze (same options as CLI)
        st.header("Reports & Analyze")
        report_analyze_source = st.radio(
            "Data for report/analysis",
            options=["Use current data above", "Run quick benchmark"],
            key="report_data_src",
            help="Same as CLI: omit --csv/--json to auto-run a benchmark",
        )
        ra_device_id = "cpu_0"
        ra_duration = 10
        if report_analyze_source == "Run quick benchmark":
            detected_ra = get_detected_devices()
            if detected_ra:
                ra_options = [
                    (
                        d.get("device_id", "?"),
                        d.get("device_name") or d.get("device_id", "?"),
                    )
                    for d in detected_ra
                ]
                ra_labels = [f"{did} ({name})" for did, name in ra_options]
                ra_idx = st.selectbox(
                    "Device",
                    range(len(ra_labels)),
                    format_func=lambda i: ra_labels[i],
                    key="ra_device",
                )
                ra_device_id = ra_options[ra_idx][0]
            else:
                ra_device_id = st.text_input("Device (e.g. cpu_0, nvidia_0)", value="cpu_0", key="ra_device_txt")
            ra_duration = st.number_input("Duration (seconds)", min_value=5, max_value=120, value=10, key="ra_dur")

        col_rep, col_an = st.columns(2)
        with col_rep:
            st.subheader("Generate report")
            st.caption("Download as HTML. Use the 'Print / Save as PDF' button in the report for PDF.")
            if st.button("Generate HTML Report"):
                data_for_report = None
                if report_analyze_source == "Use current data above" and data_list:
                    data_for_report = data_list if len(data_list) > 1 else data_list[0]
                elif report_analyze_source == "Run quick benchmark":
                    with st.spinner("Running quick benchmark..."):
                        data_for_report = run_benchmark_from_ui(ra_duration, ra_device_id, "fp32")
                if data_for_report:
                    try:
                        report_bytes, report_basename = _generate_report_directly(data_for_report, "HTML")
                        if report_bytes:
                            st.success("Report generated! Download below.")
                            st.info("Tip: Open the HTML file and click 'Print / Save as PDF' button to create a PDF.")
                            st.download_button(
                                "Download HTML Report",
                                data=report_bytes,
                                file_name=report_basename,
                                mime="text/html",
                                key="dl_report",
                            )
                        else:
                            st.error("Report generation failed")
                    except Exception as e:
                        st.error(f"Report generation failed: {e}")
                else:
                    st.warning("No data. Run a benchmark or upload a file first.")

        with col_an:
            st.subheader("Run analysis")
            analyze_type = st.selectbox(
                "Type",
                ["latency", "efficiency", "variability", "Run all"],
                key="analyze_type",
                help="Same as CLI: autoperfpy analyze <type>. 'Run all' runs latency, efficiency, and variability.",
            )
            if st.button("Run analysis"):
                data_for_analyze = None
                if report_analyze_source == "Use current data above" and data_list:
                    data_for_analyze = data_list[0]
                elif report_analyze_source == "Run quick benchmark":
                    with st.spinner("Running quick benchmark..."):
                        data_for_analyze = run_benchmark_from_ui(ra_duration, ra_device_id, "fp32")
                if data_for_analyze:
                    csv_path = result_to_csv_path(data_for_analyze)
                    if csv_path:
                        try:
                            from autoperfpy.analyzers import (
                                EfficiencyAnalyzer,
                                PercentileLatencyAnalyzer,
                                VariabilityAnalyzer,
                            )
                            from autoperfpy.config import ConfigManager

                            config = ConfigManager.load_or_default(None)
                            types_to_run = (
                                ["latency", "efficiency", "variability"]
                                if analyze_type == "Run all"
                                else [analyze_type]
                            )
                            for atype in types_to_run:
                                if atype == "latency":
                                    analyzer = PercentileLatencyAnalyzer(config)
                                    result = analyzer.analyze(csv_path)
                                elif atype == "efficiency":
                                    analyzer = EfficiencyAnalyzer(config)
                                    result = analyzer.analyze(csv_path)
                                else:
                                    analyzer = VariabilityAnalyzer(config)
                                    result = analyzer.analyze(csv_path, latency_col="latency_ms")
                                with st.expander(f"**{atype.capitalize()}**", expanded=True):
                                    st.json(result.metrics if hasattr(result, "metrics") else {})
                        finally:
                            try:
                                os.unlink(csv_path)
                            except OSError:
                                pass
                    else:
                        st.warning("No samples in data to analyze.")
                else:
                    st.warning("No data. Run a benchmark or upload a file first.")

        st.divider()

        # View options
        st.header("View Options")
        show_warmup = st.checkbox("Include warmup samples", value=False)
        show_raw_data = st.checkbox("Show raw data table", value=False)
        if data_list and result_to_csv_content(data_list[0]):
            csv_content = result_to_csv_content(data_list[0])
            st.download_button(
                "Download run as CSV",
                data=csv_content or "",
                file_name="run.csv",
                mime="text/csv",
                help="Same format as CLI --export-csv; use with autoperfpy analyze/report --csv",
                key="dl_run_csv",
            )

    # Main content
    if not data_list:
        if data_source == "Run Benchmark":
            st.info("Configure benchmark settings in the sidebar and click **Run benchmark** to start.")
        else:
            st.info("Please upload a data file or select Demo Data to get started")

        if st.button("Load sample data", key="load_sample_data_empty_state"):
            data_list = build_sample_data_list()
            st.session_state["data_list"] = data_list
            st.success("Loaded sample data.")

    if not data_list:
        st.markdown("""
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
        """)
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
                st.text(platform_meta.get("device_name") or inference_cfg.get("accelerator") or "N/A")
                st.markdown("**GPU**")
                st.text(platform_meta.get("gpu_model") or platform_meta.get("gpu", "N/A"))
            with c2:
                st.markdown("**CPU**")
                st.text(platform_meta.get("cpu_model") or platform_meta.get("cpu", "N/A"))
                st.markdown("**Precision**")
                st.text(inference_cfg.get("precision") or "N/A")
            with c3:
                st.markdown("**SoC / Power**")
                st.text(platform_meta.get("soc", "N/A"))
                st.text(platform_meta.get("power_mode", "N/A"))

    st.divider()

    # Tabs for different analyses
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Overview",
            "Latency",
            "Utilization",
            "Power & Thermal",
            "Memory",
            "Throughput",
        ]
    )

    with tab0:
        render_overview_analysis(data, display_df, summary)

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
        "<div style='text-align: center; color: gray;'>" "AutoPerfPy Dashboard | Performance Analysis Toolkit" "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
