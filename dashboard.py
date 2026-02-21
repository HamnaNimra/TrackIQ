"""Unified dashboard launcher for AutoPerfPy, MiniCluster, and trackiq-compare."""
# Cluster Health Dashboard — the single-pane view for a MiniCluster validation run.
# In production, replace static JSON inputs with a live telemetry sink (PostgreSQL or OpenSearch)
# and refresh on a configurable interval.

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, List, Optional

from autoperfpy.ui.dashboard import AutoPerfDashboard
from minicluster.runner import RunConfig, run_distributed, save_metrics
from minicluster.ui.dashboard import MiniClusterDashboard
from trackiq_compare.ui.dashboard import CompareDashboard
from trackiq_core.schema import Metrics, PlatformInfo, RegressionInfo, TrackiqResult, WorkloadInfo
from trackiq_core.serializer import load_trackiq_result
from trackiq_core.ui import DARK_THEME, ResultBrowser, run_dashboard


def _validate_path(path: Optional[str], label: str) -> str:
    if not path:
        raise SystemExit(f"{label} is required.")
    if not Path(path).exists():
        raise SystemExit(f"{label} does not exist: {path}")
    return path


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


def _load_json_dict(path: str, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"{label} is not valid JSON: {path} ({exc})") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must contain a JSON object: {path}")
    return payload


def _extract_minicluster_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize raw JSON into a minicluster-like payload dictionary."""
    payload = raw.get("tool_payload")
    if isinstance(payload, dict):
        result = dict(payload)
    else:
        result = dict(raw)

    steps = result.get("steps")
    if not isinstance(steps, list):
        per_step = result.get("per_step_metrics")
        if isinstance(per_step, list):
            result["steps"] = per_step

    # Fallbacks for canonical wrapper fields.
    metrics = raw.get("metrics")
    if isinstance(metrics, dict):
        if result.get("average_throughput_samples_per_sec") is None:
            result["average_throughput_samples_per_sec"] = metrics.get("throughput_samples_per_sec")
        if result.get("scaling_efficiency_pct") is None:
            result["scaling_efficiency_pct"] = metrics.get("scaling_efficiency_pct")
    if result.get("run_label") is None and isinstance(raw.get("workload"), dict):
        result["run_label"] = raw["workload"].get("name")

    return result


def _extract_step_rows(payload: dict[str, Any]) -> list[dict[str, float]]:
    """Extract normalized per-step rows for cluster-health charts."""
    source = payload.get("steps")
    if not isinstance(source, list):
        return []
    rows: list[dict[str, float]] = []
    for idx, item in enumerate(source):
        if not isinstance(item, dict):
            continue
        step = _to_float(item.get("step"))
        loss = _to_float(item.get("loss"))
        allreduce_ms = _to_float(item.get("allreduce_time_ms"))
        compute_ms = _to_float(item.get("compute_time_ms"))
        throughput = _to_float(item.get("throughput_samples_per_sec"))
        rows.append(
            {
                "step": float(idx if step is None else step),
                "loss": float(0.0 if loss is None else loss),
                "allreduce_ms": float(0.0 if allreduce_ms is None else allreduce_ms),
                "compute_ms": float(0.0 if compute_ms is None else compute_ms),
                "throughput": float(0.0 if throughput is None else throughput),
            }
        )
    return rows


def _status_from_metric(name: str, value: Any, payload: dict[str, Any]) -> str:
    metric = str(name).lower()
    if metric == "fabric":
        text = str(value or "").strip().lower()
        if "pass" in text or "ok" in text:
            return "PASS"
        if "warn" in text:
            return "WARN"
        if "fail" in text or "error" in text:
            return "FAIL"
        return "WARN"
    numeric = _to_float(value)
    if metric == "workers":
        return "PASS" if isinstance(value, int) and value > 0 else "FAIL"
    if metric == "throughput":
        return "PASS" if numeric is not None and numeric > 0 else "WARN"
    if metric == "p99":
        if numeric is None:
            return "WARN"
        if numeric <= 5.0:
            return "PASS"
        if numeric <= 15.0:
            return "WARN"
        return "FAIL"
    if metric == "scaling":
        if numeric is None:
            return "WARN"
        if numeric >= 90.0:
            return "PASS"
        if numeric >= 75.0:
            return "WARN"
        return "FAIL"
    if metric == "workers_from_config":
        workers_cfg = _to_int(value)
        if workers_cfg is None:
            return "WARN"
        return "PASS" if workers_cfg > 0 else "FAIL"
    return "WARN"


def _badge_html(status: str) -> str:
    label = status.upper().strip()
    if label == "PASS":
        bg = "#16a34a"
    elif label == "FAIL":
        bg = "#dc2626"
    else:
        bg = "#ca8a04"
    return (
        f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;"
        f"background:{bg};color:#fff;font-size:11px;font-weight:700'>{label}</span>"
    )


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


def _build_fault_timeline_figure(fault_report: dict[str, Any]) -> Any | None:
    """Build the fault timeline plotly figure used in cluster-health mode."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        return None

    results = fault_report.get("results", [])
    if not isinstance(results, list):
        results = []

    loss_values = _extract_loss_curve(fault_report)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.22,
        subplot_titles=("Loss Curve with Fault Injection Points", "Fault Detection Summary"),
    )

    if loss_values:
        fig.add_trace(
            go.Scatter(
                x=list(range(len(loss_values))),
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
            y_val = loss_values[detected_step] if loss_values and 0 <= detected_step < len(loss_values) else 0.0
            fig.add_trace(
                go.Scatter(
                    x=[detected_step],
                    y=[y_val],
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
        bar_text.append(f"Detected at step {detected_step}" if detected else "MISSED")

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
    return fig


def _render_cluster_health_page(
    *,
    result_payload: dict[str, Any],
    fault_report_payload: dict[str, Any] | None = None,
    scaling_payloads: list[dict[str, Any]] | None = None,
) -> None:
    """Render the unified cluster-health dashboard layout."""
    import streamlit as st

    try:
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
    except Exception as exc:
        st.error(f"Plotly/pandas are required for cluster-health dashboard rendering: {exc}")
        return

    st.title("Cluster Health Dashboard")
    run_label = result_payload.get("run_label")
    if run_label:
        st.caption(f"Run: {run_label}")

    config = result_payload.get("config")
    config = config if isinstance(config, dict) else {}
    steps = _extract_step_rows(result_payload)
    df = pd.DataFrame(steps) if steps else pd.DataFrame()

    num_workers = _to_int(result_payload.get("num_workers"))
    if num_workers is None:
        num_workers = _to_int(config.get("num_processes"))
    if num_workers is None:
        num_workers = _to_int(config.get("num_workers"))
    if num_workers is None and not df.empty:
        num_workers = 1

    avg_thr = _to_float(result_payload.get("average_throughput_samples_per_sec"))
    p99_allreduce = _to_float(result_payload.get("p99_allreduce_ms"))
    scaling_eff = _to_float(result_payload.get("scaling_efficiency_pct"))
    fabric_status = result_payload.get("fabric_probe_status", "N/A")

    # Row 1: Summary cards with PASS/WARN/FAIL badges.
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Workers", num_workers if num_workers is not None else "N/A")
        st.markdown(_badge_html(_status_from_metric("workers", num_workers, result_payload)), unsafe_allow_html=True)
    with c2:
        st.metric(
            "Avg Throughput (samples/s)",
            f"{avg_thr:.2f}" if isinstance(avg_thr, float) else "N/A",
        )
        st.markdown(_badge_html(_status_from_metric("throughput", avg_thr, result_payload)), unsafe_allow_html=True)
    with c3:
        st.metric(
            "P99 All-Reduce (ms)",
            f"{p99_allreduce:.3f}" if isinstance(p99_allreduce, float) else "N/A",
        )
        st.markdown(_badge_html(_status_from_metric("p99", p99_allreduce, result_payload)), unsafe_allow_html=True)
    with c4:
        st.metric(
            "Scaling Efficiency (%)",
            f"{scaling_eff:.2f}" if isinstance(scaling_eff, float) else "N/A",
        )
        st.markdown(_badge_html(_status_from_metric("scaling", scaling_eff, result_payload)), unsafe_allow_html=True)
    with c5:
        st.metric("Fabric Probe Status", str(fabric_status))
        st.markdown(_badge_html(_status_from_metric("fabric", fabric_status, result_payload)), unsafe_allow_html=True)

    # Row 2: Loss + timing breakdown.
    col_left, col_right = st.columns(2)
    with col_left:
        if not df.empty and "loss" in df.columns:
            fig_loss = px.line(df, x="step", y="loss", title="Loss Curve", labels={"step": "Step", "loss": "Loss"})
            st.plotly_chart(fig_loss, use_container_width=True)
        else:
            st.info("Loss curve data unavailable.")
        st.caption(
            "Loss should trend down or stabilize. Sudden spikes often indicate instability, "
            "bad gradients, or injected faults."
        )
    with col_right:
        if not df.empty:
            fig_timing = go.Figure()
            if "compute_ms" in df.columns:
                fig_timing.add_trace(
                    go.Bar(x=df["step"], y=df["compute_ms"], name="compute_time_ms", marker_color="#2563eb")
                )
            if "allreduce_ms" in df.columns:
                fig_timing.add_trace(
                    go.Bar(x=df["step"], y=df["allreduce_ms"], name="allreduce_time_ms", marker_color="#f97316")
                )
            fig_timing.update_layout(
                title="Step Time Breakdown: Compute vs All-Reduce (ms)",
                xaxis_title="Step",
                yaxis_title="Time (ms)",
                barmode="stack",
                legend_title_text="Time Source",
            )
            st.plotly_chart(fig_timing, use_container_width=True)
        else:
            st.info("Per-step timing data unavailable.")
        st.caption(
            "If all-reduce dominates step time, communication is throttling training even when compute is healthy."
        )

    # Row 3: All-reduce histogram with optional P99 marker.
    if not df.empty and "allreduce_ms" in df.columns:
        allreduce_values = [float(v) for v in df["allreduce_ms"].tolist() if isinstance(v, (int, float))]
        if allreduce_values:
            fig_hist = px.histogram(
                x=allreduce_values,
                nbins=30,
                title="All-Reduce Latency Distribution (ms)",
                labels={"x": "allreduce_time_ms", "y": "Count"},
            )
            if isinstance(p99_allreduce, float):
                fig_hist.add_vline(
                    x=p99_allreduce,
                    line_dash="dash",
                    line_color="#dc2626",
                    annotation_text="P99",
                    annotation_position="top right",
                )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("All-reduce values unavailable for histogram.")
    else:
        st.info("All-reduce values unavailable for histogram.")
    st.caption(
        "A long right tail means one or more workers are taking significantly longer than the median "
        "for gradient synchronization. This is your straggler indicator."
    )

    # Row 4: Optional fault timeline.
    if isinstance(fault_report_payload, dict):
        fig_fault = _build_fault_timeline_figure(fault_report_payload)
        if fig_fault is not None:
            st.plotly_chart(fig_fault, use_container_width=True)
        else:
            st.info("Fault report visualization unavailable (plotly missing).")
        st.caption(
            "Fault injections validate detection coverage: SLOW_WORKER, GRADIENT_SYNC_ANOMALY, and WORKER_TIMEOUT. "
            "Detection latency estimates how long bad steps can slip through in production."
        )

    # Row 5: Optional scaling chart from multiple runs.
    if scaling_payloads:
        records: list[dict[str, Any]] = []
        for idx, item in enumerate(scaling_payloads):
            if not isinstance(item, dict):
                continue
            run = _extract_minicluster_payload(item)
            run_cfg = run.get("config")
            run_cfg = run_cfg if isinstance(run_cfg, dict) else {}
            workers = _to_int(run.get("num_workers"))
            if workers is None:
                workers = _to_int(run_cfg.get("num_processes"))
            if workers is None:
                workers = _to_int(run_cfg.get("num_workers"))
            throughput = _to_float(run.get("average_throughput_samples_per_sec"))
            scaling = _to_float(run.get("scaling_efficiency_pct"))
            label = str(run.get("run_label") or f"run_{idx + 1}")
            if workers is None:
                continue
            records.append(
                {
                    "workers": workers,
                    "scaling_efficiency_pct": scaling,
                    "throughput": throughput,
                    "label": label,
                }
            )

        if records:
            baseline_candidates = [row for row in records if row["workers"] == 1 and isinstance(row["throughput"], float)]
            baseline_thr = baseline_candidates[0]["throughput"] if baseline_candidates else None
            if baseline_thr is None:
                sorted_records = sorted(records, key=lambda row: row["workers"])
                first = sorted_records[0] if sorted_records else None
                baseline_thr = first["throughput"] if first and isinstance(first.get("throughput"), float) else None

            for row in records:
                if row["scaling_efficiency_pct"] is None and isinstance(row["throughput"], float) and baseline_thr:
                    row["scaling_efficiency_pct"] = (row["throughput"] / (row["workers"] * baseline_thr)) * 100.0

            plot_rows = [row for row in records if isinstance(row.get("scaling_efficiency_pct"), float)]
            if plot_rows:
                sdf = pd.DataFrame(plot_rows).sort_values("workers")
                fig_scaling = px.line(
                    sdf,
                    x="workers",
                    y="scaling_efficiency_pct",
                    markers=True,
                    text="label",
                    title="Scaling Efficiency vs Workers",
                    labels={"workers": "Workers", "scaling_efficiency_pct": "Scaling Efficiency (%)"},
                )
                fig_scaling.add_hline(
                    y=90.0,
                    line_dash="dash",
                    line_color="#ca8a04",
                    annotation_text="90% target",
                )
                st.plotly_chart(fig_scaling, use_container_width=True)
                st.caption(
                    "Scaling efficiency near 100% indicates near-linear scaling. "
                    "Sustained drops below 90% usually point to interconnect or memory bottlenecks."
                )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified TrackIQ dashboard launcher")
    parser.add_argument(
        "--tool",
        required=False,
        default="all",
        choices=["all", "autoperfpy", "minicluster", "compare", "cluster-health"],
        help="Tool dashboard to launch (default: all)",
    )
    parser.add_argument("--result", help="Single TrackiqResult JSON path")
    parser.add_argument("--result-a", help="Compare mode: result A path")
    parser.add_argument("--result-b", help="Compare mode: result B path")
    parser.add_argument("--label-a", help="Compare mode: display label A")
    parser.add_argument("--label-b", help="Compare mode: display label B")
    parser.add_argument(
        "--fault-report",
        help="Cluster-health mode: optional fault report JSON path",
    )
    parser.add_argument(
        "--scaling-runs",
        nargs="*",
        default=None,
        help="Cluster-health mode: optional list of scaling-run JSON paths",
    )
    return parser.parse_args(argv)


def _placeholder_result(tool_name: str, workload_type: str = "inference") -> TrackiqResult:
    """Create placeholder result for browser-mode dashboards."""
    from datetime import datetime, timezone

    return TrackiqResult(
        tool_name=tool_name,
        tool_version="browser",
        timestamp=datetime.now(timezone.utc),
        platform=PlatformInfo(
            hardware_name="Unknown",
            os="Unknown",
            framework="unknown",
            framework_version="unknown",
        ),
        workload=WorkloadInfo(
            name="browser_mode",
            workload_type=workload_type,  # type: ignore[arg-type]
            batch_size=0,
            steps=0,
        ),
        metrics=Metrics(
            throughput_samples_per_sec=0.0,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
            memory_utilization_percent=0.0,
            communication_overhead_percent=None,
            power_consumption_watts=None,
        ),
        regression=RegressionInfo(
            baseline_id=None,
            delta_percent=0.0,
            status="pass",
            failed_metrics=[],
        ),
    )


def _run_minicluster_once(config: RunConfig) -> TrackiqResult:
    """Run MiniCluster once and return canonical TrackiqResult."""
    metrics = run_distributed(config)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        out_path = handle.name
    try:
        save_metrics(metrics, out_path)
        return load_trackiq_result(out_path)
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def _run_autoperf_single(
    *,
    device_id: str,
    precision: str,
    batch_size: int,
    duration_seconds: int,
    warmup_runs: int,
    iterations: int,
) -> TrackiqResult:
    """Run one AutoPerfPy benchmark and return canonical TrackiqResult."""
    from autoperfpy.auto_runner import run_single_benchmark
    from autoperfpy.cli import _infer_trackiq_result
    from autoperfpy.device_config import (
        InferenceConfig,
        resolve_device,
        resolve_precision_for_device,
    )

    device = resolve_device(device_id or "cpu_0")
    if device is None:
        raise RuntimeError(f"Device not found: {device_id}")
    effective_precision = resolve_precision_for_device(device, precision)
    config = InferenceConfig(
        precision=effective_precision,
        batch_size=int(batch_size),
        accelerator=device.device_id,
        streams=1,
        warmup_runs=int(warmup_runs),
        iterations=int(iterations),
    )
    payload = run_single_benchmark(
        device,
        config,
        duration_seconds=float(duration_seconds),
        sample_interval_seconds=0.2,
        quiet=True,
    )
    return _infer_trackiq_result(
        payload,
        workload_name=str(payload.get("run_label", "autoperfpy_manual_ui")),
        workload_type="inference",
    )


def _run_autoperf_auto(
    *,
    device_ids: List[str],
    precisions: List[str],
    batch_sizes: List[int],
    duration_seconds: int,
    max_configs_per_device: int,
) -> List[TrackiqResult]:
    """Run AutoPerfPy auto mode and return canonical TrackiqResult list."""
    from autoperfpy.auto_runner import run_auto_benchmarks
    from autoperfpy.cli import _infer_trackiq_result
    from autoperfpy.device_config import get_devices_and_configs_auto

    pairs = get_devices_and_configs_auto(
        precisions=list(precisions),
        batch_sizes=list(batch_sizes),
        max_configs_per_device=int(max_configs_per_device),
        device_ids_filter=list(device_ids) if device_ids else None,
    )
    if not pairs:
        return []
    payloads = run_auto_benchmarks(
        pairs,
        duration_seconds=float(duration_seconds),
        sample_interval_seconds=0.2,
        quiet=True,
    )
    results: List[TrackiqResult] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        results.append(
            _infer_trackiq_result(
                payload,
                workload_name=str(payload.get("run_label", "autoperfpy_auto_ui")),
                workload_type="inference",
            )
        )
    return results


def _render_autoperf_interactive(
    *,
    args: argparse.Namespace,
    key_prefix: str,
    title: Optional[str] = None,
    caption: Optional[str] = None,
) -> int:
    """Render AutoPerfPy with rich run configuration and graph output."""
    import streamlit as st

    from autoperfpy.device_config import (
        DEFAULT_BATCH_SIZES,
        DEFAULT_ITERATIONS,
        DEFAULT_WARMUP_RUNS,
        PRECISION_FP32,
        PRECISIONS,
    )
    from trackiq_core.hardware import get_all_devices

    if title:
        st.title(title)
    if caption:
        st.caption(caption)

    devices = get_all_devices()
    device_ids = [d.device_id for d in devices]
    if not device_ids:
        device_ids = ["cpu_0"]

    with st.sidebar.expander("AutoPerfPy Run Configuration", expanded=True):
        run_mode = st.radio(
            "Run Mode",
            options=["Manual", "Auto"],
            index=0,
            key=f"{key_prefix}_mode",
        )
        duration_seconds = st.number_input(
            "Duration (seconds)",
            min_value=1,
            max_value=300,
            value=10,
            step=1,
            key=f"{key_prefix}_duration",
        )
        warmup_runs = st.number_input(
            "Warmup Runs",
            min_value=0,
            max_value=200,
            value=int(DEFAULT_WARMUP_RUNS),
            step=1,
            key=f"{key_prefix}_warmup",
        )
        iterations = st.number_input(
            "Iterations",
            min_value=1,
            max_value=5000,
            value=int(DEFAULT_ITERATIONS),
            step=1,
            key=f"{key_prefix}_iterations",
        )

        if run_mode == "Manual":
            manual_device = st.selectbox(
                "Device",
                options=device_ids,
                index=0,
                key=f"{key_prefix}_manual_device",
            )
            manual_precision = st.selectbox(
                "Precision",
                options=list(PRECISIONS),
                index=list(PRECISIONS).index(PRECISION_FP32) if PRECISION_FP32 in PRECISIONS else 0,
                key=f"{key_prefix}_manual_precision",
            )
            manual_batch = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=4096,
                value=1,
                step=1,
                key=f"{key_prefix}_manual_batch",
            )
            run_clicked = st.button(
                "Run AutoPerfPy Benchmark",
                use_container_width=True,
                key=f"{key_prefix}_run_manual",
            )
            if run_clicked:
                with st.spinner("Running AutoPerfPy benchmark..."):
                    result = _run_autoperf_single(
                        device_id=str(manual_device),
                        precision=str(manual_precision),
                        batch_size=int(manual_batch),
                        duration_seconds=int(duration_seconds),
                        warmup_runs=int(warmup_runs),
                        iterations=int(iterations),
                    )
                    st.session_state[f"{key_prefix}_results"] = [result]
                    st.session_state[f"{key_prefix}_selected_idx"] = 0
        else:
            selected_devices = st.multiselect(
                "Devices",
                options=device_ids,
                default=device_ids[: min(2, len(device_ids))] or device_ids,
                key=f"{key_prefix}_auto_devices",
            )
            selected_precisions = st.multiselect(
                "Precisions",
                options=list(PRECISIONS),
                default=list(PRECISIONS[:2]) if len(PRECISIONS) >= 2 else list(PRECISIONS),
                key=f"{key_prefix}_auto_precisions",
            )
            auto_batch_sizes = st.multiselect(
                "Batch Sizes",
                options=list(dict.fromkeys(list(DEFAULT_BATCH_SIZES) + [2, 16, 32])),
                default=list(DEFAULT_BATCH_SIZES),
                key=f"{key_prefix}_auto_batches",
            )
            max_configs = st.number_input(
                "Max Configs Per Device",
                min_value=1,
                max_value=20,
                value=6,
                step=1,
                key=f"{key_prefix}_auto_max_cfg",
            )
            run_auto_clicked = st.button(
                "Run AutoPerfPy Auto Benchmarks",
                use_container_width=True,
                key=f"{key_prefix}_run_auto",
            )
            if run_auto_clicked:
                if not selected_devices:
                    st.sidebar.warning("Select at least one device.")
                elif not selected_precisions:
                    st.sidebar.warning("Select at least one precision.")
                elif not auto_batch_sizes:
                    st.sidebar.warning("Select at least one batch size.")
                else:
                    with st.spinner("Running AutoPerfPy auto benchmarks..."):
                        results = _run_autoperf_auto(
                            device_ids=[str(x) for x in selected_devices],
                            precisions=[str(x) for x in selected_precisions],
                            batch_sizes=[int(x) for x in auto_batch_sizes],
                            duration_seconds=int(duration_seconds),
                            max_configs_per_device=int(max_configs),
                        )
                        st.session_state[f"{key_prefix}_results"] = results
                        st.session_state[f"{key_prefix}_selected_idx"] = 0

    with st.sidebar.expander("Load AutoPerfPy Result", expanded=False):
        ResultBrowser(theme=DARK_THEME, allowed_tools=["autoperfpy"]).render()

    results = st.session_state.get(f"{key_prefix}_results")
    selected: Optional[TrackiqResult] = None
    if isinstance(results, list) and results:
        if len(results) > 1:
            labels = [
                f"{idx + 1}. {r.workload.name} | bs{r.workload.batch_size} | {r.platform.hardware_name}"
                for idx, r in enumerate(results)
                if isinstance(r, TrackiqResult)
            ]
            if labels:
                idx = st.sidebar.selectbox(
                    "Displayed Auto Run",
                    options=list(range(len(labels))),
                    index=min(
                        int(st.session_state.get(f"{key_prefix}_selected_idx", 0)),
                        len(labels) - 1,
                    ),
                    format_func=lambda i: labels[i],
                    key=f"{key_prefix}_selected_idx",
                )
                if 0 <= int(idx) < len(results) and isinstance(results[int(idx)], TrackiqResult):
                    selected = results[int(idx)]
        else:
            only = results[0]
            if isinstance(only, TrackiqResult):
                selected = only

    loaded = st.session_state.get("loaded_result")
    if selected is None and isinstance(loaded, TrackiqResult) and str(loaded.tool_name).lower() == "autoperfpy":
        selected = loaded
    if selected is None and args.result and Path(args.result).exists():
        loaded_arg = load_trackiq_result(args.result)
        if str(loaded_arg.tool_name).lower() == "autoperfpy":
            selected = loaded_arg

    if selected is None:
        st.info("Configure and run AutoPerfPy from the sidebar, or load an existing result.")
        return 0

    dashboard = AutoPerfDashboard(result=selected)
    dashboard.apply_theme(dashboard.theme)
    dashboard.render_header()
    dashboard.render_sidebar()
    dashboard.render_body()
    dashboard.render_footer()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for the unified dashboard launcher."""
    args = _parse_args(argv)

    try:
        if args.tool == "all":
            import streamlit as st

            st.set_page_config(
                page_title="TrackIQ Unified Dashboard",
                layout="wide",
                initial_sidebar_state="expanded",
            )
            st.title("TrackIQ Unified Dashboard")
            st.caption("Switch between AutoPerfPy, MiniCluster, and Compare in one app.")

            tool_choice = st.sidebar.selectbox(
                "Tool",
                options=["autoperfpy", "minicluster", "compare", "cluster-health"],
                index=0,
                key="trackiq_unified_tool_choice",
            )

            if tool_choice == "autoperfpy":
                return _render_autoperf_interactive(
                    args=args,
                    key_prefix="trackiq_unified_autoperf",
                )

            if tool_choice == "minicluster":
                with st.sidebar.expander("MiniCluster Run Configuration", expanded=True):
                    workers = st.number_input(
                        "Workers",
                        min_value=1,
                        max_value=16,
                        value=1,
                        step=1,
                        key="trackiq_unified_mini_workers",
                    )
                    backend = st.selectbox(
                        "Collective Backend",
                        options=["gloo", "nccl"],
                        index=0,
                        key="trackiq_unified_mini_backend",
                    )
                    workload = st.selectbox(
                        "Workload",
                        options=["mlp", "transformer", "embedding"],
                        index=0,
                        key="trackiq_unified_mini_workload",
                    )
                    steps = st.number_input(
                        "Steps",
                        min_value=1,
                        max_value=5000,
                        value=100,
                        step=1,
                        key="trackiq_unified_mini_steps",
                    )
                    batch_size = st.number_input(
                        "Batch Size",
                        min_value=1,
                        max_value=4096,
                        value=32,
                        step=1,
                        key="trackiq_unified_mini_batch",
                    )
                    learning_rate = st.number_input(
                        "Learning Rate",
                        min_value=0.0001,
                        max_value=1.0,
                        value=0.01,
                        step=0.0001,
                        format="%.4f",
                        key="trackiq_unified_mini_lr",
                    )
                    seed = st.number_input(
                        "Seed",
                        min_value=0,
                        max_value=2_147_483_647,
                        value=42,
                        step=1,
                        key="trackiq_unified_mini_seed",
                    )
                    tdp_watts = st.number_input(
                        "TDP Watts",
                        min_value=10.0,
                        max_value=1000.0,
                        value=150.0,
                        step=1.0,
                        key="trackiq_unified_mini_tdp",
                    )
                    use_baseline = st.checkbox(
                        "Use Baseline Throughput",
                        value=False,
                        key="trackiq_unified_mini_use_baseline",
                    )
                    baseline_throughput = st.number_input(
                        "Baseline Throughput (samples/s)",
                        min_value=0.0,
                        max_value=1_000_000.0,
                        value=100.0,
                        step=1.0,
                        disabled=not bool(use_baseline),
                        key="trackiq_unified_mini_baseline",
                    )
                    run_clicked = st.button(
                        "Run MiniCluster",
                        use_container_width=True,
                        key="trackiq_unified_mini_run",
                    )
                    quick_clicked = st.button(
                        "Quick Smoke Run (20 steps)",
                        use_container_width=True,
                        key="trackiq_unified_mini_quick_run",
                    )

                with st.sidebar.expander("Load MiniCluster Result", expanded=False):
                    ResultBrowser(theme=DARK_THEME, allowed_tools=["minicluster"]).render()

                if run_clicked or quick_clicked:
                    cfg = RunConfig(
                        num_steps=int(20 if quick_clicked else steps),
                        num_processes=int(1 if quick_clicked else workers),
                        batch_size=int(16 if quick_clicked else batch_size),
                        learning_rate=float(0.01 if quick_clicked else learning_rate),
                        seed=int(42 if quick_clicked else seed),
                        tdp_watts=float(tdp_watts),
                        collective_backend=str(backend),
                        workload=str(workload),
                        baseline_throughput=(float(baseline_throughput) if bool(use_baseline) else None),
                    )
                    with st.spinner("Running MiniCluster..."):
                        st.session_state["trackiq_unified_minicluster_result"] = _run_minicluster_once(cfg)

                selected = st.session_state.get("trackiq_unified_minicluster_result")
                loaded = st.session_state.get("loaded_result")
                if (
                    not isinstance(selected, TrackiqResult)
                    and isinstance(loaded, TrackiqResult)
                    and str(loaded.tool_name).lower() == "minicluster"
                ):
                    selected = loaded
                if not isinstance(selected, TrackiqResult) and args.result and Path(args.result).exists():
                    loaded_arg = load_trackiq_result(args.result)
                    if str(loaded_arg.tool_name).lower() == "minicluster":
                        selected = loaded_arg

                if not isinstance(selected, TrackiqResult):
                    st.info("Use sidebar run configuration to generate a MiniCluster run, or load an existing result.")
                    return 0

                dashboard = MiniClusterDashboard(result=selected)
                dashboard.apply_theme(dashboard.theme)
                dashboard.render_header()
                dashboard.render_sidebar()
                dashboard.render_body()
                dashboard.render_footer()
                return 0

            if tool_choice == "cluster-health":
                st.sidebar.markdown("---")
                st.sidebar.subheader("Cluster Health Inputs")
                result_path = st.sidebar.text_input(
                    "Result JSON",
                    value=str(st.session_state.get("trackiq_cluster_result_path", args.result or "")),
                    key="trackiq_cluster_result_path_input",
                )
                fault_path = st.sidebar.text_input(
                    "Fault Report JSON (optional)",
                    value=str(st.session_state.get("trackiq_cluster_fault_path", args.fault_report or "")),
                    key="trackiq_cluster_fault_path_input",
                )
                scaling_paths_raw = st.sidebar.text_area(
                    "Scaling Run JSON Paths (optional, one per line)",
                    value=str(
                        st.session_state.get(
                            "trackiq_cluster_scaling_paths_raw",
                            "\n".join(args.scaling_runs or []),
                        )
                    ),
                    key="trackiq_cluster_scaling_paths_input",
                )

                if st.sidebar.button("Load Cluster Health", use_container_width=True):
                    st.session_state["trackiq_cluster_result_path"] = result_path.strip()
                    st.session_state["trackiq_cluster_fault_path"] = fault_path.strip()
                    st.session_state["trackiq_cluster_scaling_paths_raw"] = scaling_paths_raw

                selected_result = str(st.session_state.get("trackiq_cluster_result_path", "")).strip()
                if not selected_result:
                    st.info("Provide a MiniCluster result JSON path in the sidebar and click 'Load Cluster Health'.")
                    return 0
                if not Path(selected_result).exists():
                    st.error(f"Result JSON not found: {selected_result}")
                    return 0

                result_raw = _load_json_dict(selected_result, "Result")
                result_payload = _extract_minicluster_payload(result_raw)

                selected_fault = str(st.session_state.get("trackiq_cluster_fault_path", "")).strip()
                fault_payload = _load_json_dict(selected_fault, "Fault report") if selected_fault else None

                scaling_raw = str(st.session_state.get("trackiq_cluster_scaling_paths_raw", "")).strip()
                scaling_payloads: list[dict[str, Any]] = []
                if scaling_raw:
                    scaling_paths = [line.strip() for line in scaling_raw.splitlines() if line.strip()]
                    for idx, path in enumerate(scaling_paths):
                        if not Path(path).exists():
                            st.warning(f"Scaling run path not found (skipped): {path}")
                            continue
                        scaling_payloads.append(_load_json_dict(path, f"Scaling run {idx + 1}"))

                _render_cluster_health_page(
                    result_payload=result_payload,
                    fault_report_payload=fault_payload,
                    scaling_payloads=scaling_payloads or None,
                )
                return 0

            st.sidebar.markdown("---")
            st.sidebar.subheader("Compare Configuration")
            input_mode = st.sidebar.radio(
                "Input Mode",
                options=["Browse discovered results", "Manual paths"],
                index=0,
                key="trackiq_unified_compare_input_mode",
            )
            result_a_path = ""
            result_b_path = ""
            rows = ResultBrowser(theme=DARK_THEME).to_dict()
            if input_mode == "Browse discovered results":
                if not rows:
                    st.sidebar.info("No result files discovered. Switch to manual paths.")
                else:
                    labels = [
                        f"{i + 1}. {row.get('tool_name', '?')} | {row.get('workload_name', '?')} | {row.get('timestamp', '?')}"
                        for i, row in enumerate(rows)
                    ]
                    idx_a = st.sidebar.selectbox(
                        "Result A",
                        options=list(range(len(labels))),
                        format_func=lambda i: labels[i],
                        key="trackiq_unified_compare_a_idx",
                    )
                    remaining = [i for i in range(len(labels)) if i != idx_a] or [idx_a]
                    idx_b = st.sidebar.selectbox(
                        "Result B",
                        options=remaining,
                        format_func=lambda i: labels[i],
                        key="trackiq_unified_compare_b_idx",
                    )
                    result_a_path = str(rows[idx_a]["path"])
                    result_b_path = str(rows[idx_b]["path"])
            else:
                result_a_path = st.sidebar.text_input(
                    "Result A Path",
                    value=args.result_a or "",
                    key="trackiq_unified_result_a",
                )
                result_b_path = st.sidebar.text_input(
                    "Result B Path",
                    value=args.result_b or "",
                    key="trackiq_unified_result_b",
                )

            label_a = st.sidebar.text_input(
                "Label A",
                value=args.label_a or "Result A",
                key="trackiq_unified_label_a",
            )
            label_b = st.sidebar.text_input(
                "Label B",
                value=args.label_b or "Result B",
                key="trackiq_unified_label_b",
            )
            regression_threshold = st.sidebar.slider(
                "Regression Threshold (%)",
                min_value=0.5,
                max_value=50.0,
                value=5.0,
                step=0.5,
                key="trackiq_unified_compare_regression_threshold",
            )
            variance_threshold = st.sidebar.slider(
                "Variance Threshold (%)",
                min_value=1.0,
                max_value=200.0,
                value=25.0,
                step=1.0,
                key="trackiq_unified_compare_variance_threshold",
            )

            if st.sidebar.button("Load Comparison", use_container_width=True):
                if not result_a_path or not Path(result_a_path).exists():
                    st.sidebar.error(f"Result A file not found: {result_a_path}")
                elif not result_b_path or not Path(result_b_path).exists():
                    st.sidebar.error(f"Result B file not found: {result_b_path}")
                else:
                    st.session_state["trackiq_unified_compare_a"] = load_trackiq_result(result_a_path)
                    st.session_state["trackiq_unified_compare_b"] = load_trackiq_result(result_b_path)
                    st.session_state["trackiq_unified_compare_label_a"] = label_a
                    st.session_state["trackiq_unified_compare_label_b"] = label_b
                    st.session_state["trackiq_unified_compare_regression_threshold"] = float(regression_threshold)
                    st.session_state["trackiq_unified_compare_variance_threshold"] = float(variance_threshold)

            result_a = st.session_state.get("trackiq_unified_compare_a")
            result_b = st.session_state.get("trackiq_unified_compare_b")
            if not isinstance(result_a, TrackiqResult) or not isinstance(result_b, TrackiqResult):
                st.info("Configure comparison in the sidebar and click 'Load Comparison'.")
                return 0

            dashboard = CompareDashboard(
                result_a=result_a,
                result_b=result_b,
                label_a=str(st.session_state.get("trackiq_unified_compare_label_a", label_a)),
                label_b=str(st.session_state.get("trackiq_unified_compare_label_b", label_b)),
                regression_threshold_percent=float(
                    st.session_state.get(
                        "trackiq_unified_compare_regression_threshold",
                        regression_threshold,
                    )
                ),
                variance_threshold_percent=float(
                    st.session_state.get(
                        "trackiq_unified_compare_variance_threshold",
                        variance_threshold,
                    )
                ),
            )
            dashboard.apply_theme(dashboard.theme)
            dashboard.render_header()
            dashboard.render_body()
            dashboard.render_footer()
            return 0

        if args.tool == "cluster-health":
            import streamlit as st

            result_path = _validate_path(args.result, "--result")
            fault_path = _validate_path(args.fault_report, "--fault-report") if args.fault_report else None
            scaling_paths = [_validate_path(path, "--scaling-runs") for path in (args.scaling_runs or [])]

            st.set_page_config(
                page_title="Cluster Health Dashboard",
                layout="wide",
                initial_sidebar_state="expanded",
            )

            result_payload = _extract_minicluster_payload(_load_json_dict(result_path, "Result"))
            fault_payload = _load_json_dict(fault_path, "Fault report") if fault_path else None
            scaling_payloads = [_load_json_dict(path, "Scaling run") for path in scaling_paths] if scaling_paths else None
            _render_cluster_health_page(
                result_payload=result_payload,
                fault_report_payload=fault_payload,
                scaling_payloads=scaling_payloads,
            )
            return 0

        if args.tool == "autoperfpy":
            import streamlit as st

            if args.result:
                _validate_path(args.result, "--result")

            st.set_page_config(
                page_title="AutoPerfPy Dashboard",
                layout="wide",
                initial_sidebar_state="expanded",
            )
            return _render_autoperf_interactive(
                args=args,
                key_prefix="trackiq_autoperf_only",
                title="AutoPerfPy Interactive Dashboard",
                caption="Configure benchmark runs and visualize graph-rich results.",
            )

        if args.tool == "minicluster":
            if args.result:
                result_path = _validate_path(args.result, "--result")
                run_dashboard(MiniClusterDashboard, result_path=result_path)
            else:

                class _MiniClusterBrowserDashboard(MiniClusterDashboard):
                    def render_body(self) -> None:
                        import streamlit as st

                        loaded = st.session_state.get("loaded_result")
                        if loaded is None:
                            st.info("Select a result file to begin.")
                            ResultBrowser(theme=self.theme).render()
                            return
                        self.result = loaded
                        super().render_body()

                run_dashboard(
                    _MiniClusterBrowserDashboard,
                    result=_placeholder_result("minicluster", workload_type="training"),
                )
            return 0

        if not args.result_a or not args.result_b:
            from trackiq_compare.ui import streamlit_app

            streamlit_app.main()
            return 0

        result_a_path = _validate_path(args.result_a, "--result-a")
        result_b_path = _validate_path(args.result_b, "--result-b")
        result_a = load_trackiq_result(result_a_path)
        result_b = load_trackiq_result(result_b_path)

        class _CompareDashboardAdapter(CompareDashboard):
            def __init__(self, result, theme, title="TrackIQ Compare Dashboard"):
                if not isinstance(result, list) or len(result) != 2:
                    raise ValueError("Compare dashboard requires exactly two loaded results.")
                super().__init__(
                    result_a=result[0],
                    result_b=result[1],
                    label_a=args.label_a,
                    label_b=args.label_b,
                    theme=theme,
                    title=title,
                )

        run_dashboard(_CompareDashboardAdapter, result=[result_a, result_b])  # type: ignore[arg-type]
        return 0
    except Exception as exc:
        raise SystemExit(f"Failed to launch dashboard: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
