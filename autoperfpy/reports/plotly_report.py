"""Plotly HTML report builder for inference benchmark JSON payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency guard
    go = None
    make_subplots = None
    PLOTLY_AVAILABLE = False


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _safe_get(payload: dict[str, Any], *path: str) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _coerce_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize raw benchmark or TrackiqResult payload into a working payload."""
    if isinstance(data.get("tool_payload"), dict):
        payload = dict(data["tool_payload"])
        payload.setdefault("tool_name", data.get("tool_name"))
        payload.setdefault("platform", data.get("platform"))
        payload.setdefault("workload", data.get("workload"))
        payload.setdefault("metrics", data.get("metrics"))
        return payload
    return data


def _latency_bars(payload: dict[str, Any]) -> tuple[list[str], list[float]]:
    summary_latency = _safe_get(payload, "summary", "latency")
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}

    candidates = [
        ("P50", _to_float((summary_latency or {}).get("p50_ms"))),
        ("P90", _to_float((summary_latency or {}).get("p90_ms"))),
        ("P95", _to_float((summary_latency or {}).get("p95_ms"))),
        ("P99", _to_float((summary_latency or {}).get("p99_ms"))),
        ("P50", _to_float(metrics.get("latency_p50_ms"))),
        ("P95", _to_float(metrics.get("latency_p95_ms"))),
        ("P99", _to_float(metrics.get("latency_p99_ms"))),
    ]
    merged: dict[str, float] = {}
    for label, value in candidates:
        if value is None:
            continue
        merged[label] = value
    if merged:
        order = ["P50", "P90", "P95", "P99"]
        labels = [label for label in order if label in merged]
        return labels, [merged[label] for label in labels]

    llm_candidates = [
        ("Mean TTFT", _to_float(payload.get("mean_ttft_ms"))),
        ("P99 TTFT", _to_float(payload.get("p99_ttft_ms"))),
        ("Mean TPOT", _to_float(payload.get("mean_tpot_ms"))),
        ("P99 TPOT", _to_float(payload.get("p99_tpot_ms"))),
        ("TTFT", _to_float(metrics.get("ttft_ms"))),
        ("TPOT", _to_float(metrics.get("decode_tpt_ms"))),
    ]
    llm_points = [(label, value) for label, value in llm_candidates if value is not None]
    if llm_points:
        return [label for label, _ in llm_points], [value for _, value in llm_points]

    average_latency = _to_float((summary_latency or {}).get("mean_ms"))
    if average_latency is None:
        average_latency = _to_float(payload.get("mean_ttft_ms"))
    if average_latency is None:
        average_latency = _to_float(metrics.get("ttft_ms"))
    if average_latency is not None:
        return ["Average"], [average_latency]
    return [], []


def _throughput_sweep(payload: dict[str, Any]) -> tuple[list[int], list[float]]:
    runs = payload.get("runs")
    if isinstance(runs, list):
        points: list[tuple[int, float]] = []
        for run in runs:
            if not isinstance(run, dict):
                continue
            bs = _to_float(_safe_get(run, "inference_config", "batch_size"))
            thr = _to_float(_safe_get(run, "summary", "throughput", "mean_fps"))
            if bs is None or thr is None:
                continue
            points.append((int(bs), thr))
        if points:
            points.sort(key=lambda row: row[0])
            return [p[0] for p in points], [p[1] for p in points]

    batching = payload.get("batching_results")
    if isinstance(batching, dict):
        batch_sizes = batching.get("batch_size")
        throughput = batching.get("throughput_img_per_sec")
        if isinstance(batch_sizes, list) and isinstance(throughput, list) and len(batch_sizes) == len(throughput):
            xs: list[int] = []
            ys: list[float] = []
            for batch, value in zip(batch_sizes, throughput):
                batch_i = _to_float(batch)
                thr_f = _to_float(value)
                if batch_i is None or thr_f is None:
                    continue
                xs.append(int(batch_i))
                ys.append(thr_f)
            if xs:
                return xs, ys
    return [], []


def _power_efficiency(payload: dict[str, Any]) -> float | None:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    throughput = _to_float(_safe_get(summary, "throughput", "mean_fps"))
    if throughput is None:
        throughput = _to_float(payload.get("throughput_tokens_per_sec"))
    if throughput is None:
        throughput = _to_float(_safe_get(payload, "metrics", "throughput_samples_per_sec"))

    power = _to_float(_safe_get(summary, "power", "mean_w"))
    if power is None:
        power = _to_float(_safe_get(payload, "metrics", "power_consumption_watts"))
    if power is None or power <= 0:
        return None
    if throughput is None:
        return None
    return throughput / power


def _summary_rows(payload: dict[str, Any]) -> list[list[str]]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    platform = payload.get("platform") if isinstance(payload.get("platform"), dict) else {}
    platform_meta = payload.get("platform_metadata") if isinstance(payload.get("platform_metadata"), dict) else {}

    device = (
        platform_meta.get("device_name")
        or platform_meta.get("gpu_model")
        or platform.get("hardware_name")
        or payload.get("model")
        or "N/A"
    )
    duration = _to_float(summary.get("duration_seconds"))
    avg_throughput = _to_float(_safe_get(summary, "throughput", "mean_fps"))
    if avg_throughput is None:
        avg_throughput = _to_float(payload.get("throughput_tokens_per_sec"))
    if avg_throughput is None:
        avg_throughput = _to_float(metrics.get("throughput_samples_per_sec"))

    peak_power = _to_float(_safe_get(summary, "power", "max_w"))
    if peak_power is None:
        peak_power = _to_float(metrics.get("power_consumption_watts"))
    backend = payload.get("backend")
    prompts = _to_float(payload.get("num_prompts"))

    rows: list[list[str]] = [
        ["Device Name", str(device)],
        ["Duration", f"{duration:.2f} s" if duration is not None else "N/A"],
        ["Backend", str(backend) if backend is not None else "N/A"],
        ["Prompts", f"{int(prompts)}" if prompts is not None else "N/A"],
        ["Average Throughput", f"{avg_throughput:.2f}" if avg_throughput is not None else "N/A"],
        ["Peak Power", f"{peak_power:.2f} W" if peak_power is not None else "N/A"],
    ]
    return rows


def _interpretation_rows(payload: dict[str, Any]) -> list[list[str]]:
    """Build plain-English guidance rows for key results in this report."""
    rows: list[list[str]] = []
    latency_labels, _ = _latency_bars(payload)
    if latency_labels:
        rows.append(
            [
                "Latency percentiles",
                "Shows typical and tail response time (P50/P95/P99).",
                "Use P99 as the straggler and SLO risk signal.",
            ]
        )

    sweep_x, sweep_y = _throughput_sweep(payload)
    if sweep_x and sweep_y:
        rows.append(
            [
                "Throughput vs batch size",
                "Capacity trend as batch size changes.",
                "Reveals scaling sweet spots and saturation points.",
            ]
        )

    if _power_efficiency(payload) is not None:
        rows.append(
            [
                "Throughput per watt",
                "Delivered work for each watt consumed.",
                "Tracks operating cost efficiency and thermal headroom.",
            ]
        )

    rows.append(
        [
            "Summary table",
            "Run metadata (device, duration, backend, throughput, peak power).",
            "Use this to validate run comparability before metric deltas.",
        ]
    )
    return rows


def generate_plotly_report(data: dict[str, Any], output_path: str) -> str:
    """Generate AutoPerfPy Inference Benchmark Plotly report HTML."""
    if not PLOTLY_AVAILABLE:  # pragma: no cover
        raise RuntimeError("Plotly is required for Plotly report generation.")

    payload = _coerce_payload(data)
    rows: list[dict[str, Any]] = []

    latency_labels, latency_values = _latency_bars(payload)
    if latency_labels:
        rows.append({"type": "latency"})

    sweep_x, sweep_y = _throughput_sweep(payload)
    if sweep_x and sweep_y:
        rows.append({"type": "throughput"})

    efficiency = _power_efficiency(payload)
    if efficiency is not None:
        rows.append({"type": "efficiency"})

    rows.append({"type": "summary"})
    rows.append({"type": "guide"})

    specs = []
    titles = []
    for row in rows:
        if row["type"] == "summary":
            specs.append([{"type": "table"}])
            titles.append("Summary")
        elif row["type"] == "guide":
            specs.append([{"type": "table"}])
            titles.append("How to Interpret These Results")
        else:
            specs.append([{"type": "xy"}])
            if row["type"] == "latency":
                titles.append("Latency Percentiles (ms)")
            elif row["type"] == "throughput":
                titles.append("Throughput vs Batch Size")
            else:
                titles.append("Power Efficiency")

    fig = make_subplots(rows=len(rows), cols=1, specs=specs, subplot_titles=titles, vertical_spacing=0.12)

    row_index = 1
    for row in rows:
        row_type = row["type"]
        if row_type == "latency":
            fig.add_trace(
                go.Bar(x=latency_labels, y=latency_values, name="Latency (ms)", marker_color="#2563eb"),
                row=row_index,
                col=1,
            )
            fig.update_yaxes(title_text="Milliseconds", row=row_index, col=1)
        elif row_type == "throughput":
            fig.add_trace(
                go.Scatter(
                    x=sweep_x,
                    y=sweep_y,
                    mode="lines+markers",
                    name="Throughput",
                    line=dict(color="#0f766e", width=2),
                ),
                row=row_index,
                col=1,
            )
            fig.update_xaxes(title_text="Batch Size", row=row_index, col=1)
            fig.update_yaxes(title_text="Throughput", row=row_index, col=1)
        elif row_type == "efficiency":
            fig.add_trace(
                go.Bar(x=["Throughput per Watt"], y=[efficiency], name="Efficiency", marker_color="#ca8a04"),
                row=row_index,
                col=1,
            )
            fig.update_yaxes(title_text="Units / W", row=row_index, col=1)
        elif row_type == "summary":
            summary_rows = _summary_rows(payload)
            fig.add_trace(
                go.Table(
                    header=dict(values=["Field", "Value"], fill_color="#1f2937", font=dict(color="white")),
                    cells=dict(values=[[r[0] for r in summary_rows], [r[1] for r in summary_rows]]),
                ),
                row=row_index,
                col=1,
            )
        else:
            guide_rows = _interpretation_rows(payload)
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=["Result", "What it means", "Why it matters"],
                        fill_color="#1f2937",
                        font=dict(color="white"),
                    ),
                    cells=dict(
                        values=[[r[0] for r in guide_rows], [r[1] for r in guide_rows], [r[2] for r in guide_rows]]
                    ),
                ),
                row=row_index,
                col=1,
            )
        row_index += 1

    fig.update_layout(
        title="AutoPerfPy Inference Benchmark Report",
        template="plotly_white",
        height=380 * len(rows) + 120,
        showlegend=False,
        margin=dict(l=48, r=48, t=84, b=48),
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        fig.to_html(
            full_html=True,
            include_plotlyjs="cdn",
            config={"displayModeBar": True, "displaylogo": False, "responsive": True},
        ),
        encoding="utf-8",
    )
    return str(output)
