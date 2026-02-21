"""Shared Plotly chart builders for performance analysis.

This module provides reusable chart-building functions used by both
the Streamlit UI and HTML/PDF report generators. All functions return
Plotly figure objects that can be displayed or embedded.

For HTML reports, Chart.js interactive charts can be used instead of
Plotly via add_interactive_charts_to_html_report (or add_charts_to_html_report
with chart_engine="chartjs") for a polished, example-style look.
"""

from typing import Any, Optional

try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


def is_available() -> bool:
    """Check if Plotly and pandas are available."""
    return PLOTLY_AVAILABLE and PANDAS_AVAILABLE


def samples_to_dataframe(samples: list[dict]) -> "pd.DataFrame":
    """Convert collector samples to a pandas DataFrame.

    Args:
        samples: List of sample dictionaries or CollectorSample objects from collector export

    Returns:
        DataFrame with flattened metrics and elapsed_seconds column
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for samples_to_dataframe")

    rows = []
    for sample in samples:
        # Handle both dict and dataclass/object samples
        if isinstance(sample, dict):
            timestamp = sample.get("timestamp", 0)
            metrics = sample.get("metrics", sample)
            metadata = sample.get("metadata", {})
        else:
            # Handle dataclass or object with attributes
            timestamp = getattr(sample, "timestamp", 0)
            metrics = getattr(sample, "metrics", {})
            metadata = getattr(sample, "metadata", {})

            # If metrics is not a dict, try to convert from dataclass
            if hasattr(metrics, "__dict__") and not isinstance(metrics, dict):
                metrics = vars(metrics)

        row = {"timestamp": timestamp}

        # Flatten metrics into row
        if isinstance(metrics, dict):
            row.update(metrics)

        # Add metadata with prefix
        if isinstance(metadata, dict):
            row.update({f"meta_{k}": v for k, v in metadata.items()})

        rows.append(row)

    df = pd.DataFrame(rows)
    if "timestamp" in df.columns and len(df) > 0:
        df["elapsed_seconds"] = df["timestamp"] - df["timestamp"].min()
    return df


def ensure_throughput_column(df: "pd.DataFrame") -> None:
    """Add throughput_fps column from latency_ms if missing (mutates df in place).

    Used by report generators when building charts from samples/CSV that lack
    throughput. Idempotent if throughput_fps already present.
    """
    if not PANDAS_AVAILABLE:
        return
    if "latency_ms" not in df.columns or "throughput_fps" in df.columns:
        return
    import numpy as np

    df["throughput_fps"] = 1000.0 / df["latency_ms"].replace(0, np.nan)


def compute_summary_from_dataframe(df: "pd.DataFrame") -> dict[str, Any]:
    """Compute summary statistics from a DataFrame when summary is missing.

    Produces a dict compatible with chart helpers and report generators (e.g.
    CSV upload, or export without precomputed summary).

    Args:
        df: DataFrame with sample data (latency_ms, throughput_fps, etc.)

    Returns:
        Summary dict with latency, throughput, cpu, gpu, memory, power, temperature.
    """
    if not PANDAS_AVAILABLE:
        return {}
    import numpy as np

    def _pct(arr: "pd.Series", p: float) -> float:
        arr = arr.dropna()
        if len(arr) == 0:
            return 0.0
        return float(np.percentile(arr, p))

    summary: dict[str, Any] = {"sample_count": len(df)}

    if "latency_ms" in df.columns:
        lat = df["latency_ms"].dropna()
        if len(lat) > 0:
            summary["latency"] = {
                "mean_ms": float(lat.mean()),
                "min_ms": float(lat.min()),
                "max_ms": float(lat.max()),
                "p50_ms": _pct(lat, 50),
                "p95_ms": _pct(lat, 95),
                "p99_ms": _pct(lat, 99),
            }

    if "throughput_fps" in df.columns:
        thr = df["throughput_fps"].dropna()
        if len(thr) > 0:
            summary["throughput"] = {
                "mean_fps": float(thr.mean()),
                "min_fps": float(thr.min()),
                "max_fps": float(thr.max()),
            }

    if "cpu_percent" in df.columns:
        cpu = df["cpu_percent"].dropna()
        if len(cpu) > 0:
            summary["cpu"] = {
                "mean_percent": float(cpu.mean()),
                "max_percent": float(cpu.max()),
            }

    if "gpu_percent" in df.columns:
        gpu = df["gpu_percent"].dropna()
        if len(gpu) > 0:
            summary["gpu"] = {
                "mean_percent": float(gpu.mean()),
                "max_percent": float(gpu.max()),
            }

    if "memory_used_mb" in df.columns:
        mem = df["memory_used_mb"].dropna()
        if len(mem) > 0:
            summary["memory"] = {
                "mean_mb": float(mem.mean()),
                "max_mb": float(mem.max()),
                "min_mb": float(mem.min()),
            }
            if "memory_total_mb" in df.columns:
                summary["memory"]["total_mb"] = float(df["memory_total_mb"].iloc[0])

    if "power_w" in df.columns:
        pwr = df["power_w"].dropna()
        if len(pwr) > 0:
            summary["power"] = {
                "mean_w": float(pwr.mean()),
                "max_w": float(pwr.max()),
            }

    if "temperature_c" in df.columns:
        temp = df["temperature_c"].dropna()
        if len(temp) > 0:
            summary["temperature"] = {
                "mean_c": float(temp.mean()),
                "max_c": float(temp.max()),
            }

    return summary


# -----------------------------------------------------------------------------
# Latency Charts
# -----------------------------------------------------------------------------


def create_latency_timeline(
    df: "pd.DataFrame",
    show_warmup: bool = True,
) -> Optional["go.Figure"]:
    """Create latency over time line chart.

    Args:
        df: DataFrame with latency_ms and elapsed_seconds columns
        show_warmup: Whether to color by warmup status if available

    Returns:
        Plotly figure or None if data unavailable
    """
    if not PLOTLY_AVAILABLE or "latency_ms" not in df.columns:
        return None

    color_col = "is_warmup" if show_warmup and "is_warmup" in df.columns else None
    fig = px.line(
        df,
        x="elapsed_seconds",
        y="latency_ms",
        color=color_col,
        title="Latency Over Time",
        labels={"elapsed_seconds": "Time (seconds)", "latency_ms": "Latency (ms)"},
    )
    fig.update_layout(showlegend=(color_col is not None))
    return fig


def create_latency_histogram(
    df: "pd.DataFrame",
    summary: dict[str, Any],
    exclude_warmup: bool = True,
    nbins: int = 50,
) -> Optional["go.Figure"]:
    """Create latency distribution histogram with percentile markers.

    Args:
        df: DataFrame with latency_ms column
        summary: Summary dict with latency percentiles (p50_ms, p95_ms, p99_ms)
        exclude_warmup: Whether to exclude warmup samples
        nbins: Number of histogram bins

    Returns:
        Plotly figure or None if data unavailable
    """
    if not PLOTLY_AVAILABLE or "latency_ms" not in df.columns:
        return None

    used_warmup_filter = bool(exclude_warmup and "is_warmup" in df.columns)
    plot_df = df[~df["is_warmup"]] if used_warmup_filter else df
    if used_warmup_filter and len(plot_df["latency_ms"].dropna()) == 0:
        # Fall back to all samples when warmup filtering removes all chartable points.
        plot_df = df
        used_warmup_filter = False

    plot_df = plot_df.dropna(subset=["latency_ms"])
    if len(plot_df) == 0:
        return None

    fig = px.histogram(
        plot_df,
        x="latency_ms",
        nbins=nbins,
        title=("Latency Distribution (excluding warmup)" if used_warmup_filter else "Latency Distribution"),
        labels={"latency_ms": "Latency (ms)", "count": "Frequency"},
    )

    latency = summary.get("latency", {})
    for pct, color in [("p50_ms", "green"), ("p95_ms", "orange"), ("p99_ms", "red")]:
        val = latency.get(pct)
        if val is not None:
            fig.add_vline(
                x=float(val),
                line_dash="dash",
                line_color=color,
                annotation_text=f"{pct.replace('_ms', '')}: {val:.1f}ms",
            )

    return fig


# -----------------------------------------------------------------------------
# Utilization Charts
# -----------------------------------------------------------------------------


def create_utilization_timeline(df: "pd.DataFrame") -> Optional["go.Figure"]:
    """Create CPU/GPU utilization over time chart.

    Args:
        df: DataFrame with cpu_percent and/or gpu_percent columns

    Returns:
        Plotly figure or None if no utilization data
    """
    if not PLOTLY_AVAILABLE:
        return None

    has_cpu = "cpu_percent" in df.columns
    has_gpu = "gpu_percent" in df.columns

    if not has_cpu and not has_gpu:
        return None

    fig = go.Figure()

    if has_cpu:
        fig.add_trace(
            go.Scatter(
                x=df["elapsed_seconds"],
                y=df["cpu_percent"],
                mode="lines",
                name="CPU %",
                line=dict(color="steelblue"),
            )
        )

    if has_gpu:
        fig.add_trace(
            go.Scatter(
                x=df["elapsed_seconds"],
                y=df["gpu_percent"],
                mode="lines",
                name="GPU %",
                line=dict(color="green"),
            )
        )

    fig.update_layout(
        title="CPU/GPU Utilization Over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Utilization (%)",
        yaxis=dict(range=[0, 105]),
    )
    return fig


def create_utilization_summary_bar(summary: dict[str, Any]) -> Optional["go.Figure"]:
    """Create utilization summary bar chart (mean vs max).

    Args:
        summary: Summary dict with cpu and gpu sections

    Returns:
        Plotly figure or None if no utilization data
    """
    if not PLOTLY_AVAILABLE:
        return None

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

    if not categories:
        return None

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
    return fig


# -----------------------------------------------------------------------------
# Power & Thermal Charts
# -----------------------------------------------------------------------------


def create_power_timeline(df: "pd.DataFrame") -> Optional["go.Figure"]:
    """Create power consumption over time chart.

    Args:
        df: DataFrame with power_w column

    Returns:
        Plotly figure or None if no power data
    """
    if not PLOTLY_AVAILABLE or "power_w" not in df.columns:
        return None

    fig = px.line(
        df,
        x="elapsed_seconds",
        y="power_w",
        title="Power Consumption Over Time",
        labels={"elapsed_seconds": "Time (seconds)", "power_w": "Power (W)"},
    )
    fig.update_traces(line_color="orange")
    return fig


def create_temperature_timeline(
    df: "pd.DataFrame",
    throttle_threshold: float = 85.0,
) -> Optional["go.Figure"]:
    """Create temperature over time chart with throttle threshold.

    Args:
        df: DataFrame with temperature_c column
        throttle_threshold: Temperature threshold to mark (default 85C)

    Returns:
        Plotly figure or None if no temperature data
    """
    if not PLOTLY_AVAILABLE or "temperature_c" not in df.columns:
        return None

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
    fig.add_hline(
        y=throttle_threshold,
        line_dash="dash",
        line_color="darkred",
        annotation_text="Throttle Threshold",
    )
    return fig


# -----------------------------------------------------------------------------
# Memory Charts
# -----------------------------------------------------------------------------


def create_memory_timeline(df: "pd.DataFrame") -> Optional["go.Figure"]:
    """Create memory usage over time chart.

    Args:
        df: DataFrame with memory_used_mb column (optionally memory_total_mb)

    Returns:
        Plotly figure or None if no memory data
    """
    if not PLOTLY_AVAILABLE or "memory_used_mb" not in df.columns:
        return None

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

    if "memory_total_mb" in df.columns and len(df) > 0:
        total_mem = df["memory_total_mb"].iloc[0]
        fig.add_hline(
            y=total_mem,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Total: {total_mem} MB",
        )

    return fig


def create_memory_gauge(
    summary: dict[str, Any],
    total_mb: float | None = None,
) -> Optional["go.Figure"]:
    """Create memory usage gauge indicator.

    Args:
        summary: Summary dict with memory section (mean_mb, max_mb)
        total_mb: Total memory in MB (default 16384 if not provided)

    Returns:
        Plotly figure or None if no memory data
    """
    if not PLOTLY_AVAILABLE:
        return None

    memory = summary.get("memory", {})
    if not memory:
        return None

    mean_mb = memory.get("mean_mb", 0)
    max_mb = memory.get("max_mb", 0)
    total = total_mb if total_mb is not None else 16384

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=mean_mb,
            delta={"reference": max_mb, "relative": False, "valueformat": ".0f"},
            title={"text": "Mean Memory Usage (MB)"},
            gauge={
                "axis": {"range": [0, total]},
                "bar": {"color": "purple"},
                "steps": [
                    {"range": [0, total * 0.7], "color": "lightgray"},
                    {"range": [total * 0.7, total * 0.9], "color": "lightyellow"},
                    {"range": [total * 0.9, total], "color": "lightcoral"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": max_mb,
                },
            },
        )
    )
    return fig


# -----------------------------------------------------------------------------
# Throughput Charts
# -----------------------------------------------------------------------------


def create_throughput_timeline(
    df: "pd.DataFrame",
    summary: dict[str, Any],
    exclude_warmup: bool = True,
) -> Optional["go.Figure"]:
    """Create throughput over time chart with mean line.

    Args:
        df: DataFrame with throughput_fps column
        summary: Summary dict with throughput section
        exclude_warmup: Whether to exclude warmup samples

    Returns:
        Plotly figure or None if no throughput data
    """
    if not PLOTLY_AVAILABLE or "throughput_fps" not in df.columns:
        return None

    used_warmup_filter = bool(exclude_warmup and "is_warmup" in df.columns)
    plot_df = df[~df["is_warmup"]] if used_warmup_filter else df
    if used_warmup_filter and len(plot_df["throughput_fps"].dropna()) == 0:
        # Fall back to all samples when warmup filtering removes all chartable points.
        plot_df = df
        used_warmup_filter = False

    plot_df = plot_df.dropna(subset=["elapsed_seconds", "throughput_fps"])
    if len(plot_df) == 0:
        return None

    fig = px.line(
        plot_df,
        x="elapsed_seconds",
        y="throughput_fps",
        title=("Throughput Over Time (excluding warmup)" if used_warmup_filter else "Throughput Over Time"),
        labels={
            "elapsed_seconds": "Time (seconds)",
            "throughput_fps": "Throughput (FPS)",
        },
    )
    fig.update_traces(line_color="teal")

    throughput = summary.get("throughput", {})
    mean_fps = throughput.get("mean_fps")
    if mean_fps is not None:
        fig.add_hline(
            y=mean_fps,
            line_dash="dash",
            line_color="darkgreen",
            annotation_text=f"Mean: {mean_fps:.1f} FPS",
        )

    return fig


# -----------------------------------------------------------------------------
# Multi-Run Comparison Charts
# -----------------------------------------------------------------------------


def create_latency_comparison_bar(
    runs: list[dict[str, Any]],
    run_names: list[str] | None = None,
) -> Optional["go.Figure"]:
    """Create latency comparison bar chart across multiple runs.

    Args:
        runs: List of run data dictionaries
        run_names: Optional list of run names (defaults to run_label or Run N)

    Returns:
        Plotly figure or None if insufficient data
    """
    if not PLOTLY_AVAILABLE or not PANDAS_AVAILABLE or len(runs) < 2:
        return None

    if run_names is None:
        run_names = [r.get("run_label") or r.get("collector_name") or f"Run {i+1}" for i, r in enumerate(runs)]

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

    if not latency_data:
        return None

    df = pd.DataFrame(latency_data)
    fig = px.bar(
        df,
        x="Run",
        y="Latency (ms)",
        color="Percentile",
        barmode="group",
        title="Latency Comparison Across Runs",
    )
    return fig


def create_throughput_comparison_bar(
    runs: list[dict[str, Any]],
    run_names: list[str] | None = None,
) -> Optional["go.Figure"]:
    """Create throughput comparison bar chart across multiple runs.

    Args:
        runs: List of run data dictionaries
        run_names: Optional list of run names

    Returns:
        Plotly figure or None if insufficient data
    """
    if not PLOTLY_AVAILABLE or not PANDAS_AVAILABLE or len(runs) < 2:
        return None

    if run_names is None:
        run_names = [r.get("run_label") or r.get("collector_name") or f"Run {i+1}" for i, r in enumerate(runs)]

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

    if not throughput_data:
        return None

    df = pd.DataFrame(throughput_data)
    fig = px.bar(
        df,
        x="Run",
        y="Throughput (FPS)",
        title="Throughput Comparison Across Runs",
        color="Run",
    )
    return fig


# -----------------------------------------------------------------------------
# Chart.js interactive charts for HTML reports (example-style)
# -----------------------------------------------------------------------------


def _downsample_for_chart(
    df: "pd.DataFrame",
    x_col: str,
    y_cols: list[str],
    max_points: int = 500,
) -> "pd.DataFrame":
    """Downsample DataFrame for line charts to avoid huge payloads."""
    if not PANDAS_AVAILABLE or len(df) <= max_points:
        return df
    step = len(df) / max_points
    indices = [int(i * step) for i in range(max_points)]
    indices = sorted(set(indices))
    cols = [c for c in [x_col] + y_cols if c in df.columns]
    return df.iloc[indices][cols].copy()


def add_interactive_charts_to_html_report(
    report: Any,
    df: "pd.DataFrame",
    summary: dict[str, Any],
) -> None:
    """Add Chart.js interactive charts and a Summary Statistics table to the report.

    Uses the same underlying data as Plotly charts (df/summary) but renders
    with Chart.js for a polished, example-style HTML report (hover, zoom, filter).

    Args:
        report: HTMLReportGenerator instance
        df: DataFrame from samples_to_dataframe(samples)
        summary: Summary statistics dict (latency, throughput, power, etc.)
    """
    if not PANDAS_AVAILABLE:
        return

    section_descriptions = {
        "Latency": "Latency analysis and distribution",
        "Utilization": "CPU and GPU resource utilization",
        "Power & Thermal": "Power consumption and temperature monitoring",
        "Memory": "Memory usage analysis",
        "Throughput": "Throughput performance analysis",
        "Summary Statistics": "Key metrics at a glance",
    }

    has_elapsed = "elapsed_seconds" in df.columns

    # ----- Latency -----
    if "latency_ms" in df.columns and has_elapsed:
        report.add_section("Latency", section_descriptions["Latency"])
        plot_df = _downsample_for_chart(df, "elapsed_seconds", ["latency_ms"], max_points=500)
        plot_df = plot_df.dropna(subset=["elapsed_seconds", "latency_ms"])
        labels = [f"{x:.1f}s" for x in plot_df["elapsed_seconds"].tolist()]
        if len(plot_df) > 0:
            report.add_interactive_line_chart(
                labels=labels,
                datasets=[{"label": "Latency", "data": plot_df["latency_ms"].tolist()}],
                title="Latency Over Time",
                section="Latency",
                description="Latency (ms) vs elapsed time. Scroll to zoom, drag to pan.",
                x_label="Time (s)",
                y_label="Latency (ms)",
                enable_zoom=True,
            )
        # Latency distribution (bar)
        exclude_warmup = "is_warmup" in df.columns
        plot_df_hist = df[~df["is_warmup"]] if exclude_warmup else df
        if exclude_warmup and len(plot_df_hist["latency_ms"].dropna()) == 0:
            plot_df_hist = df
            exclude_warmup = False
        plot_df_hist = plot_df_hist["latency_ms"].dropna()
        if len(plot_df_hist) > 0:
            hist, bin_edges = _histogram_bins(plot_df_hist.tolist(), nbins=20)
            if hist and len(bin_edges) > 1:
                bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges) - 1)]
                report.add_interactive_bar_chart(
                    labels=bin_labels,
                    datasets=[{"label": "Count", "data": hist}],
                    title="Latency Distribution",
                    section="Latency",
                    description=("Excluding warmup samples." if exclude_warmup else "Distribution of latency values."),
                    x_label="Latency (ms)",
                    y_label="Count",
                )

    # ----- Utilization -----
    has_cpu = "cpu_percent" in df.columns
    has_gpu = "gpu_percent" in df.columns
    if (has_cpu or has_gpu) and has_elapsed:
        report.add_section("Utilization", section_descriptions["Utilization"])
        plot_df = _downsample_for_chart(
            df,
            "elapsed_seconds",
            [c for c in ["cpu_percent", "gpu_percent"] if c in df.columns],
            max_points=500,
        )
        labels = [f"{x:.1f}s" for x in plot_df["elapsed_seconds"].tolist()]
        datasets = []
        if "cpu_percent" in plot_df.columns:
            datasets.append({"label": "CPU %", "data": plot_df["cpu_percent"].tolist()})
        if "gpu_percent" in plot_df.columns:
            datasets.append({"label": "GPU %", "data": plot_df["gpu_percent"].tolist()})
        if datasets:
            report.add_interactive_line_chart(
                labels=labels,
                datasets=datasets,
                title="CPU/GPU Utilization Over Time",
                section="Utilization",
                description="Scroll to zoom, drag to pan. Click legend to toggle series.",
                x_label="Time (s)",
                y_label="Utilization (%)",
                enable_zoom=True,
            )
        cpu_data = summary.get("cpu", {})
        gpu_data = summary.get("gpu", {})
        categories = []
        means = []
        maxes = []
        if cpu_data:
            categories.append("CPU")
            means.append(round(cpu_data.get("mean_percent", 0), 1))
            maxes.append(round(cpu_data.get("max_percent", 0), 1))
        if gpu_data:
            categories.append("GPU")
            means.append(round(gpu_data.get("mean_percent", 0), 1))
            maxes.append(round(gpu_data.get("max_percent", 0), 1))
        if categories:
            report.add_interactive_bar_chart(
                labels=categories,
                datasets=[
                    {"label": "Mean", "data": means},
                    {"label": "Max", "data": maxes},
                ],
                title="Utilization Summary",
                section="Utilization",
                description="Mean vs max utilization.",
                x_label="",
                y_label="Utilization (%)",
            )

    # ----- Power & Thermal -----
    has_power = "power_w" in df.columns
    has_temp = "temperature_c" in df.columns
    if (has_power or has_temp) and has_elapsed:
        report.add_section("Power & Thermal", section_descriptions["Power & Thermal"])
        y_cols = [c for c in ["power_w", "temperature_c"] if c in df.columns]
        plot_df = _downsample_for_chart(df, "elapsed_seconds", y_cols, max_points=500)
        labels = [f"{x:.1f}s" for x in plot_df["elapsed_seconds"].tolist()]
        if "power_w" in plot_df.columns:
            report.add_interactive_line_chart(
                labels=labels,
                datasets=[{"label": "Power", "data": plot_df["power_w"].tolist()}],
                title="Power Over Time",
                section="Power & Thermal",
                description="Power consumption (W) vs time.",
                x_label="Time (s)",
                y_label="Power (W)",
                enable_zoom=True,
            )
        if "temperature_c" in plot_df.columns:
            report.add_interactive_line_chart(
                labels=labels,
                datasets=[{"label": "Temperature", "data": plot_df["temperature_c"].tolist()}],
                title="Temperature Over Time",
                section="Power & Thermal",
                description="Temperature (°C) vs time.",
                x_label="Time (s)",
                y_label="Temperature (°C)",
                enable_zoom=True,
            )

    # ----- Memory -----
    if "memory_used_mb" in df.columns and has_elapsed:
        report.add_section("Memory", section_descriptions["Memory"])
        plot_df = _downsample_for_chart(df, "elapsed_seconds", ["memory_used_mb"], max_points=500)
        labels = [f"{x:.1f}s" for x in plot_df["elapsed_seconds"].tolist()]
        report.add_interactive_line_chart(
            labels=labels,
            datasets=[{"label": "Memory Used", "data": plot_df["memory_used_mb"].tolist()}],
            title="Memory Over Time",
            section="Memory",
            description="Memory used (MB) vs time.",
            x_label="Time (s)",
            y_label="Memory (MB)",
            enable_zoom=True,
        )
        mem = summary.get("memory", {})
        total_mb = None
        if "memory_total_mb" in df.columns and len(df) > 0:
            total_mb = float(df["memory_total_mb"].iloc[0])
        if mem and total_mb:
            used = mem.get("mean_mb", 0) or 0
            free = max(0, total_mb - used)
            report.add_interactive_pie_chart(
                labels=["Used (mean)", "Free"],
                data=[round(used, 1), round(free, 1)],
                title="Memory Summary",
                section="Memory",
                description=f"Mean usage vs free (total {total_mb:.0f} MB).",
                doughnut=True,
            )

    # ----- Throughput -----
    if "throughput_fps" in df.columns and has_elapsed:
        exclude_warmup = "is_warmup" in df.columns
        plot_df = df[~df["is_warmup"]] if exclude_warmup else df
        if exclude_warmup and len(plot_df["throughput_fps"].dropna()) == 0:
            plot_df = df
            exclude_warmup = False
        plot_df = _downsample_for_chart(plot_df, "elapsed_seconds", ["throughput_fps"], max_points=500)
        plot_df = plot_df.dropna(subset=["elapsed_seconds", "throughput_fps"])
        if len(plot_df) > 0:
            report.add_section("Throughput", section_descriptions["Throughput"])
            labels = [f"{x:.1f}s" for x in plot_df["elapsed_seconds"].tolist()]
            thr_list = plot_df["throughput_fps"].tolist()
            datasets = [{"label": "Throughput", "data": thr_list}]
            mean_fps = summary.get("throughput", {}).get("mean_fps")
            if mean_fps is not None:
                datasets.append(
                    {
                        "label": "Mean",
                        "data": [round(mean_fps, 2)] * len(thr_list),
                    }
                )
            report.add_interactive_line_chart(
                labels=labels,
                datasets=datasets,
                title="Throughput Over Time",
                section="Throughput",
                description=(
                    "Throughput (FPS) vs time, excluding warmup." if exclude_warmup else "Throughput (FPS) vs time."
                ),
                x_label="Time (s)",
                y_label="Throughput (FPS)",
                enable_zoom=True,
            )

    # ----- Summary Statistics table -----
    rows: list[list[str | int | float]] = []
    lat = summary.get("latency", {})
    for label, key, fmt, unit in [
        ("P50 Latency", "p50_ms", ".2f", "ms"),
        ("P95 Latency", "p95_ms", ".2f", "ms"),
        ("P99 Latency", "p99_ms", ".2f", "ms"),
        ("Mean Latency", "mean_ms", ".2f", "ms"),
    ]:
        v = lat.get(key)
        if v is not None:
            rows.append([label, format(float(v), fmt), unit])
    thr = summary.get("throughput", {})
    v = thr.get("mean_fps")
    if v is not None:
        rows.append(["Mean Throughput", format(float(v), ".1f"), "FPS"])
    pwr = summary.get("power", {})
    v = pwr.get("mean_w")
    if v is not None:
        rows.append(["Mean Power", format(float(v), ".1f"), "W"])
    gpu = summary.get("gpu", {})
    v = gpu.get("mean_percent")
    if v is not None:
        rows.append(["Avg GPU", format(float(v), ".1f"), "%"])
    cpu = summary.get("cpu", {})
    v = cpu.get("mean_percent")
    if v is not None:
        rows.append(["Avg CPU", format(float(v), ".1f"), "%"])
    mem = summary.get("memory", {})
    v = mem.get("mean_mb")
    if v is not None:
        rows.append(["Mean Memory", format(float(v), ".0f"), "MB"])
    if rows:
        report.add_section("Summary Statistics", section_descriptions["Summary Statistics"])
        report.add_table(
            title="Key metrics",
            headers=["Metric", "Value", "Unit"],
            rows=[[str(c) for c in r] for r in rows],
            section="Summary Statistics",
        )


def _histogram_bins(values: list[float], nbins: int = 30) -> tuple:
    """Return (counts per bin, bin edges)."""
    if not values or not PANDAS_AVAILABLE:
        return [], []
    import numpy as np

    arr = np.array(values, dtype=float)
    hist, bin_edges = np.histogram(arr, bins=nbins)
    return hist.tolist(), bin_edges.tolist()


# -----------------------------------------------------------------------------
# All Charts Builder (for HTML/PDF reports)
# -----------------------------------------------------------------------------


def build_all_charts(
    df: "pd.DataFrame",
    summary: dict[str, Any],
) -> dict[str, list[tuple]]:
    """Build all available charts organized by section.

    Args:
        df: DataFrame with sample data
        summary: Summary statistics dictionary

    Returns:
        Dict mapping section name to list of (caption, figure) tuples
    """
    sections: dict[str, list[tuple]] = {}

    # Latency
    latency_charts = []
    fig = create_latency_timeline(df)
    if fig:
        latency_charts.append(("Latency Over Time", fig))
    fig = create_latency_histogram(df, summary)
    if fig:
        latency_charts.append(("Latency Distribution", fig))
    if latency_charts:
        sections["Latency"] = latency_charts

    # Utilization
    util_charts = []
    fig = create_utilization_timeline(df)
    if fig:
        util_charts.append(("Utilization Over Time", fig))
    fig = create_utilization_summary_bar(summary)
    if fig:
        util_charts.append(("Utilization Summary", fig))
    if util_charts:
        sections["Utilization"] = util_charts

    # Power & Thermal
    power_charts = []
    fig = create_power_timeline(df)
    if fig:
        power_charts.append(("Power Over Time", fig))
    fig = create_temperature_timeline(df)
    if fig:
        power_charts.append(("Temperature Over Time", fig))
    if power_charts:
        sections["Power & Thermal"] = power_charts

    # Memory
    memory_charts = []
    fig = create_memory_timeline(df)
    if fig:
        memory_charts.append(("Memory Over Time", fig))
    total_mb = float(df["memory_total_mb"].iloc[0]) if "memory_total_mb" in df.columns and len(df) > 0 else None
    fig = create_memory_gauge(summary, total_mb)
    if fig:
        memory_charts.append(("Memory Summary", fig))
    if memory_charts:
        sections["Memory"] = memory_charts

    # Throughput
    throughput_charts = []
    fig = create_throughput_timeline(df, summary)
    if fig:
        throughput_charts.append(("Throughput Over Time", fig))
    if throughput_charts:
        sections["Throughput"] = throughput_charts

    return sections


def add_charts_to_html_report(
    report: Any,
    df: "pd.DataFrame",
    summary: dict[str, Any],
    chart_engine: str = "chartjs",
) -> None:
    """Add charts to an HTMLReportGenerator.

    By default uses Chart.js interactive charts (chart_engine="chartjs") for
    a polished, example-style report with hover, zoom, and filter. Set
    chart_engine="plotly" to use Plotly figures instead (matches Streamlit UI).

    Args:
        report: HTMLReportGenerator instance
        df: DataFrame with sample data
        summary: Summary statistics dictionary
        chart_engine: "chartjs" (default) or "plotly"
    """
    if chart_engine == "chartjs":
        add_interactive_charts_to_html_report(report, df, summary)
        return
    if not PLOTLY_AVAILABLE:
        return

    plot_id = [0]

    def _next_id():
        plot_id[0] += 1
        return f"plotly_{plot_id[0]}"

    def _fig_to_html(fig, caption: str) -> str:
        """Convert a Plotly figure to HTML, preserving original layout."""
        # Only update size-related properties, preserve all other layout settings
        # Use height that matches CSS container (400px - padding)
        fig.update_layout(
            autosize=True,
            height=380,
            margin=dict(l=50, r=50, t=50, b=50),
        )
        return fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            div_id=_next_id(),
            config={
                "responsive": True,
                "displayModeBar": True,
                "displaylogo": False,
                "scrollZoom": True,
            },
        )

    section_descriptions = {
        "Latency": "Latency analysis and distribution",
        "Utilization": "CPU and GPU resource utilization",
        "Power & Thermal": "Power consumption and temperature monitoring",
        "Memory": "Memory usage analysis",
        "Throughput": "Throughput performance analysis",
    }

    # Build charts using the SAME functions as Streamlit UI
    # Latency section
    latency_charts = []
    if "latency_ms" in df.columns:
        fig = create_latency_timeline(df)
        if fig:
            latency_charts.append(("Latency Over Time", fig))
        fig = create_latency_histogram(df, summary)
        if fig:
            latency_charts.append(("Latency Distribution", fig))

    if latency_charts:
        report.add_section("Latency", section_descriptions["Latency"])
        for caption, fig in latency_charts:
            report.add_html_figure(_fig_to_html(fig, caption), caption, "Latency")

    # Utilization section
    util_charts = []
    has_cpu = "cpu_percent" in df.columns
    has_gpu = "gpu_percent" in df.columns
    if has_cpu or has_gpu:
        fig = create_utilization_timeline(df)
        if fig:
            util_charts.append(("CPU/GPU Utilization Over Time", fig))
        fig = create_utilization_summary_bar(summary)
        if fig:
            util_charts.append(("Utilization Summary", fig))

    if util_charts:
        report.add_section("Utilization", section_descriptions["Utilization"])
        for caption, fig in util_charts:
            report.add_html_figure(_fig_to_html(fig, caption), caption, "Utilization")

    # Power & Thermal section
    power_charts = []
    has_power = "power_w" in df.columns
    has_temp = "temperature_c" in df.columns
    if has_power or has_temp:
        if has_power:
            fig = create_power_timeline(df)
            if fig:
                power_charts.append(("Power Over Time", fig))
        if has_temp:
            fig = create_temperature_timeline(df)
            if fig:
                power_charts.append(("Temperature Over Time", fig))

    if power_charts:
        report.add_section("Power & Thermal", section_descriptions["Power & Thermal"])
        for caption, fig in power_charts:
            report.add_html_figure(_fig_to_html(fig, caption), caption, "Power & Thermal")

    # Memory section
    memory_charts = []
    if "memory_used_mb" in df.columns:
        fig = create_memory_timeline(df)
        if fig:
            memory_charts.append(("Memory Over Time", fig))
        total_mb = float(df["memory_total_mb"].iloc[0]) if "memory_total_mb" in df.columns and len(df) > 0 else None
        fig = create_memory_gauge(summary, total_mb)
        if fig:
            memory_charts.append(("Memory Usage", fig))

    if memory_charts:
        report.add_section("Memory", section_descriptions["Memory"])
        for caption, fig in memory_charts:
            report.add_html_figure(_fig_to_html(fig, caption), caption, "Memory")

    # Throughput section
    throughput_charts = []
    if "throughput_fps" in df.columns:
        fig = create_throughput_timeline(df, summary)
        if fig:
            throughput_charts.append(("Throughput Over Time", fig))

    if throughput_charts:
        report.add_section("Throughput", section_descriptions["Throughput"])
        for caption, fig in throughput_charts:
            report.add_html_figure(_fig_to_html(fig, caption), caption, "Throughput")


__all__ = [
    "is_available",
    "samples_to_dataframe",
    "ensure_throughput_column",
    "compute_summary_from_dataframe",
    "create_latency_timeline",
    "create_latency_histogram",
    "create_utilization_timeline",
    "create_utilization_summary_bar",
    "create_power_timeline",
    "create_temperature_timeline",
    "create_memory_timeline",
    "create_memory_gauge",
    "create_throughput_timeline",
    "create_latency_comparison_bar",
    "create_throughput_comparison_bar",
    "build_all_charts",
    "add_charts_to_html_report",
    "add_interactive_charts_to_html_report",
]
