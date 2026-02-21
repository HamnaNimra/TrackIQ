"""Plotly report generators for MiniCluster cluster-health workflows."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from minicluster.deps import ensure_parent_dir

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency guard
    px = None
    go = None
    make_subplots = None
    PLOTLY_AVAILABLE = False


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _write_html_figure(fig: Any, output_path: str) -> None:
    ensure_parent_dir(output_path)
    html = fig.to_html(
        full_html=True,
        include_plotlyjs="cdn",
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
        },
    )
    Path(output_path).write_text(html, encoding="utf-8")


def generate_cluster_heatmap(results: list[dict], metric: str, output_path: str) -> None:
    """Render 1xN worker heatmap for a single metric."""
    if not PLOTLY_AVAILABLE:  # pragma: no cover
        raise RuntimeError("Plotly is required for cluster heatmap reporting.")
    if not results:
        raise ValueError("At least one worker result is required.")

    parsed: list[tuple[int, float]] = []
    for idx, item in enumerate(results):
        if not isinstance(item, dict):
            continue
        worker_id = _to_int(item.get("worker_id"))
        value = _to_float(item.get(metric))
        if worker_id is None or value is None:
            continue
        parsed.append((worker_id, value))

    if not parsed:
        raise ValueError(f"No valid worker metric values found for '{metric}'.")

    parsed.sort(key=lambda row: row[0])
    worker_ids = [worker_id for worker_id, _ in parsed]
    values = [value for _, value in parsed]
    median_value = statistics.median(values)
    straggler_flags = [(value > (1.5 * median_value)) if median_value > 0 else False for value in values]

    scale = "RdYlGn" if "throughput" in metric else "RdYlGn_r"
    fig = px.imshow(
        [values],
        x=worker_ids,
        y=[metric],
        aspect="auto",
        color_continuous_scale=scale,
        title=f"Cluster Heatmap: {metric}",
        labels={"x": "Worker / Rank", "y": "", "color": metric},
    )

    hover_text = [
        [
            (f"Worker ID: {worker_id}<br>" f"{metric}: {value:.6f}<br>" f"Straggler: {'YES' if is_straggler else 'NO'}")
            for worker_id, value, is_straggler in zip(worker_ids, values, straggler_flags)
        ]
    ]
    fig.update_traces(hovertemplate="%{text}<extra></extra>", text=hover_text)
    fig.update_coloraxes(colorbar_title=metric)

    # This is your straggler finder. In a 1000-node AMD cluster, one red cell in a sea of green = one limping GPU.
    # On production hardware, color by p99_allreduce_ms to surface the node dragging the All-Reduce.
    # On MI300X, expected allreduce_time_ms should be within 10% across all ranks —
    # anything beyond that is a thermal throttle or Infinity Fabric link degradation.
    total_workers = len(worker_ids)
    for index, is_straggler in enumerate(straggler_flags):
        if not is_straggler:
            continue
        fig.add_shape(
            type="rect",
            xref="x domain",
            yref="y domain",
            x0=index / total_workers,
            x1=(index + 1) / total_workers,
            y0=0.0,
            y1=1.0,
            line=dict(color="#dc2626", width=3),
            fillcolor="rgba(0,0,0,0)",
        )
        fig.add_annotation(
            x=worker_ids[index],
            y=metric,
            text="STRAGGLER",
            showarrow=False,
            yshift=28,
            font=dict(color="#dc2626", size=10),
        )

    fig.update_layout(margin=dict(l=48, r=48, t=72, b=48))
    _write_html_figure(fig, output_path)


def _extract_loss_curve(fault_report: dict[str, Any]) -> list[float]:
    explicit = fault_report.get("loss_curve")
    if isinstance(explicit, list):
        values = [float(v) for v in explicit if isinstance(v, (int, float))]
        if values:
            return values

    losses = fault_report.get("losses")
    if isinstance(losses, list):
        values = [float(v) for v in losses if isinstance(v, (int, float))]
        if values:
            return values

    steps = fault_report.get("steps")
    if isinstance(steps, list):
        values = [
            float(item.get("loss"))
            for item in steps
            if isinstance(item, dict) and isinstance(item.get("loss"), (int, float))
        ]
        if values:
            return values

    return []


def generate_fault_timeline(fault_report: dict, output_path: str) -> None:
    """Render fault injection timeline and detection-latency summary."""
    if not PLOTLY_AVAILABLE:  # pragma: no cover
        raise RuntimeError("Plotly is required for fault timeline reporting.")
    if not isinstance(fault_report, dict):
        raise ValueError("fault_report must be a dictionary.")

    results = fault_report.get("results", [])
    if not isinstance(results, list):
        results = []

    loss_values = _extract_loss_curve(fault_report)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.22,
        subplot_titles=(
            "Loss Curve with Fault Injection Points",
            "Fault Detection Summary",
        ),
    )

    if loss_values:
        steps = list(range(len(loss_values)))
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=loss_values,
                mode="lines+markers",
                name="loss",
                line=dict(color="#2563eb", width=2),
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_annotation(
            text="Loss curve unavailable in fault report.",
            xref="x domain",
            yref="y domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            row=1,
            col=1,
        )

    fault_labels: list[str] = []
    latencies: list[float] = []
    colors: list[str] = []
    bar_text: list[str] = []

    for idx, entry in enumerate(results):
        if not isinstance(entry, dict):
            continue
        fault_type = str(entry.get("fault_type", f"fault_{idx}")).upper()
        detected = bool(entry.get("was_detected", False))
        injection_step = _to_int(entry.get("injection_step"))
        detected_step = _to_int(entry.get("detected_step"))
        explicit_latency = _to_float(entry.get("detection_latency_steps"))

        if injection_step is not None:
            fig.add_vline(
                x=injection_step,
                line_dash="dash",
                line_color="#dc2626",
                annotation_text=fault_type,
                annotation_position="top left",
                row=1,
                col=1,
            )

        if detected_step is not None:
            marker_y = loss_values[detected_step] if loss_values and 0 <= detected_step < len(loss_values) else 0.0
            fig.add_trace(
                go.Scatter(
                    x=[detected_step],
                    y=[marker_y],
                    mode="markers",
                    marker=dict(color="#16a34a", size=10),
                    name=f"{fault_type} detected",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        latency_steps = explicit_latency
        if latency_steps is None and injection_step is not None and detected_step is not None:
            latency_steps = float(max(detected_step - injection_step, 0))
        if latency_steps is None:
            latency_steps = 0.0

        fault_labels.append(fault_type)
        latencies.append(float(latency_steps))
        colors.append("#16a34a" if detected else "#dc2626")
        if detected:
            if detected_step is not None:
                bar_text.append(f"Detected at step {detected_step}")
            else:
                bar_text.append("Detected at step ?")
        else:
            bar_text.append("MISSED")

    if not fault_labels:
        fault_labels = ["NO_FAULTS"]
        latencies = [0.0]
        colors = ["#9ca3af"]
        bar_text = ["No fault records"]

    fig.add_trace(
        go.Bar(
            x=latencies,
            y=fault_labels,
            orientation="h",
            marker_color=colors,
            text=bar_text,
            textposition="auto",
            name="Detection Latency (steps)",
        ),
        row=2,
        col=1,
    )

    detected_count = int(
        fault_report.get("num_detected", sum(1 for e in results if isinstance(e, dict) and e.get("was_detected")))
    )
    fault_count = int(fault_report.get("num_faults", len(results)))

    # This visualization proves your monitoring stack catches the three production failure modes:
    # SLOW_WORKER = thermal throttling, GRADIENT_SYNC_ANOMALY = HBM bit-flip / Silent Data Corruption,
    # WORKER_TIMEOUT = network partition. The detection latency (steps between injection and detection)
    # tells you how many training steps of corrupted math would slip through in production before your monitoring catches it.
    fig.update_layout(
        title="Fault Injection Validation Report",
        template="plotly_white",
        height=900,
        margin=dict(l=60, r=40, t=110, b=60),
        showlegend=False,
    )
    fig.add_annotation(
        x=0.0,
        y=1.08,
        xref="paper",
        yref="paper",
        text=f"Detection Rate: {detected_count}/{fault_count} faults caught",
        showarrow=False,
        font=dict(size=13, color="#1f2937"),
    )
    fig.update_xaxes(title_text="Step", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_xaxes(title_text="Detection latency (steps)", row=2, col=1)
    fig.update_yaxes(title_text="Fault Type", row=2, col=1)

    _write_html_figure(fig, output_path)


def load_worker_results_from_dir(results_dir: str, metric: str) -> list[dict[str, Any]]:
    """Load per-worker JSON files and extract one metric per file."""
    directory = Path(results_dir)
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    entries: list[dict[str, Any]] = []
    files = sorted(directory.glob("*.json"))
    for index, file_path in enumerate(files):
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue

        tool_payload = payload.get("tool_payload")
        tool_payload = tool_payload if isinstance(tool_payload, dict) else {}

        worker_id = _to_int(payload.get("worker_id"))
        if worker_id is None:
            worker_id = _to_int(tool_payload.get("worker_id"))
        if worker_id is None:
            worker_id = index

        metric_value = _to_float(payload.get(metric))
        if metric_value is None:
            metric_value = _to_float(tool_payload.get(metric))
        if metric_value is None and metric == "throughput_samples_per_sec":
            metrics = payload.get("metrics")
            if isinstance(metrics, dict):
                metric_value = _to_float(metrics.get("throughput_samples_per_sec"))
        if metric_value is None and metric in {"allreduce_time_ms", "compute_time_ms"}:
            steps = tool_payload.get("steps")
            if isinstance(steps, list):
                step_values = [
                    _to_float(step.get(metric))
                    for step in steps
                    if isinstance(step, dict) and _to_float(step.get(metric)) is not None
                ]
                clean_values = [v for v in step_values if v is not None]
                if clean_values:
                    metric_value = sum(clean_values) / len(clean_values)

        if metric_value is None:
            continue
        entries.append({"worker_id": worker_id, metric: metric_value})

    if not entries:
        raise ValueError(f"No worker values found for metric '{metric}' in {results_dir}.")

    entries.sort(key=lambda item: int(item["worker_id"]))
    return entries
