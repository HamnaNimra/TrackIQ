"""Shared Plotly chart builders for performance analysis.

This module provides reusable chart-building functions used by both
the Streamlit UI and HTML/PDF report generators. All functions return
Plotly figure objects that can be displayed or embedded.
"""

from typing import Any, Dict, List, Optional

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


def samples_to_dataframe(samples: List[Dict]) -> "pd.DataFrame":
    """Convert collector samples to a pandas DataFrame.

    Args:
        samples: List of sample dictionaries from collector export

    Returns:
        DataFrame with flattened metrics and elapsed_seconds column
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for samples_to_dataframe")

    rows = []
    for sample in samples:
        row = {"timestamp": sample.get("timestamp", 0)}
        m = (
            sample.get("metrics", sample)
            if isinstance(sample, dict)
            else getattr(sample, "metrics", {})
        )
        if isinstance(m, dict):
            row.update(m)
        if "metadata" in sample and isinstance(sample.get("metadata"), dict):
            row.update({f"meta_{k}": v for k, v in sample["metadata"].items()})
        rows.append(row)

    df = pd.DataFrame(rows)
    if "timestamp" in df.columns and len(df) > 0:
        df["elapsed_seconds"] = df["timestamp"] - df["timestamp"].min()
    return df


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
    summary: Dict[str, Any],
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

    plot_df = df
    if exclude_warmup and "is_warmup" in df.columns:
        plot_df = df[~df["is_warmup"]]

    fig = px.histogram(
        plot_df,
        x="latency_ms",
        nbins=nbins,
        title="Latency Distribution (excluding warmup)",
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


def create_utilization_summary_bar(summary: Dict[str, Any]) -> Optional["go.Figure"]:
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
    summary: Dict[str, Any],
    total_mb: Optional[float] = None,
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
    summary: Dict[str, Any],
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

    plot_df = df
    if exclude_warmup and "is_warmup" in df.columns:
        plot_df = df[~df["is_warmup"]]

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
    runs: List[Dict[str, Any]],
    run_names: Optional[List[str]] = None,
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
        run_names = [
            r.get("run_label") or r.get("collector_name") or f"Run {i+1}"
            for i, r in enumerate(runs)
        ]

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
    runs: List[Dict[str, Any]],
    run_names: Optional[List[str]] = None,
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
        run_names = [
            r.get("run_label") or r.get("collector_name") or f"Run {i+1}"
            for i, r in enumerate(runs)
        ]

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
# All Charts Builder (for HTML/PDF reports)
# -----------------------------------------------------------------------------


def build_all_charts(
    df: "pd.DataFrame",
    summary: Dict[str, Any],
) -> Dict[str, List[tuple]]:
    """Build all available charts organized by section.

    Args:
        df: DataFrame with sample data
        summary: Summary statistics dictionary

    Returns:
        Dict mapping section name to list of (caption, figure) tuples
    """
    sections: Dict[str, List[tuple]] = {}

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
    total_mb = (
        float(df["memory_total_mb"].iloc[0])
        if "memory_total_mb" in df.columns and len(df) > 0
        else None
    )
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
    report,
    df: "pd.DataFrame",
    summary: Dict[str, Any],
) -> None:
    """Add all Plotly charts to an HTMLReportGenerator as HTML figures.

    Args:
        report: HTMLReportGenerator instance
        df: DataFrame with sample data
        summary: Summary statistics dictionary
    """
    if not PLOTLY_AVAILABLE:
        return

    sections = build_all_charts(df, summary)
    plot_id = [0]

    def _next_id():
        plot_id[0] += 1
        return f"plotly_{plot_id[0]}"

    section_descriptions = {
        "Latency": "Latency analysis and distribution",
        "Utilization": "CPU and GPU resource utilization",
        "Power & Thermal": "Power consumption and temperature monitoring",
        "Memory": "Memory usage analysis",
        "Throughput": "Throughput performance analysis",
    }

    for section_name, chart_list in sections.items():
        report.add_section(section_name, section_descriptions.get(section_name, ""))
        for caption, fig in chart_list:
            # Update figure layout for better HTML embedding
            fig.update_layout(
                autosize=True,
                height=380,
                margin=dict(l=50, r=50, t=50, b=50),
            )
            html_str = fig.to_html(
                full_html=False,
                include_plotlyjs=False,
                div_id=_next_id(),
                config={"responsive": True, "displayModeBar": True},
            )
            report.add_html_figure(html_str, caption, section_name)


__all__ = [
    "is_available",
    "samples_to_dataframe",
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
]
