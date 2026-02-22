"""Shared helpers for building consistent HTML reports from benchmark payloads."""

from __future__ import annotations

import json
from typing import Any


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


def _merge_missing(base: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    """Merge fallback keys into base, only filling missing values."""
    merged = dict(base)
    for key, fallback_value in fallback.items():
        current_value = merged.get(key)
        if isinstance(current_value, dict) and isinstance(fallback_value, dict):
            merged[key] = _merge_missing(current_value, fallback_value)
            continue
        if key not in merged or _is_missing(current_value):
            merged[key] = fallback_value
    return merged


def _format_value(value: Any) -> str:
    """Format values for report tables and metadata."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=True)
    text = str(value).strip()
    return text if text else "-"


def _has_numeric_signal(series: Any) -> bool:
    """Return True when a pandas-like series has any meaningful numeric value."""
    try:
        cleaned = series.dropna()
    except Exception:
        return False
    if len(cleaned) == 0:
        return False
    try:
        numeric = cleaned.astype(float)
    except Exception:
        return True
    return bool((numeric.abs() > 1e-12).any())


def _extract_power_profile_by_step(data: dict[str, Any]) -> dict[int, dict[str, float | None]]:
    """Return power_profile readings keyed by step index."""
    profile = data.get("power_profile")
    if not isinstance(profile, dict):
        return {}
    readings = profile.get("step_readings")
    if not isinstance(readings, list):
        return {}

    mapped: dict[int, dict[str, float | None]] = {}
    for reading in readings:
        if not isinstance(reading, dict):
            continue
        step = reading.get("step")
        if not isinstance(step, int) or step < 0:
            continue
        power = reading.get("power_watts")
        temp = reading.get("temperature_celsius")
        mapped[step] = {
            "power_w": float(power) if isinstance(power, (int, float)) else None,
            "temperature_c": float(temp) if isinstance(temp, (int, float)) else None,
        }
    return mapped


def _fill_dataframe_power_columns(report_df: Any, data: dict[str, Any]) -> None:
    """Backfill per-sample power/temperature columns from power_profile when needed."""
    if report_df is None or len(report_df) == 0:
        return
    by_step = _extract_power_profile_by_step(data)
    if not by_step:
        return

    sample_ids: list[int] = []
    if "meta_sample_index" in report_df.columns:
        for row_idx, sample_id in enumerate(report_df["meta_sample_index"].tolist()):
            if isinstance(sample_id, int):
                sample_ids.append(sample_id)
            else:
                sample_ids.append(row_idx)
    else:
        sample_ids = list(range(len(report_df)))

    mapped_power = [by_step.get(sample_id, {}).get("power_w") for sample_id in sample_ids]
    mapped_temp = [by_step.get(sample_id, {}).get("temperature_c") for sample_id in sample_ids]

    for column, mapped_values in (("power_w", mapped_power), ("temperature_c", mapped_temp)):
        if column not in report_df.columns:
            report_df[column] = mapped_values
            continue
        if not _has_numeric_signal(report_df[column]):
            report_df[column] = mapped_values
            continue
        try:
            report_df[column] = report_df[column].where(report_df[column].notna(), mapped_values)
        except Exception:
            pass


def _enrich_summary_from_power_profile(summary: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """Fill missing power/temperature summary fields from power_profile summary."""
    profile = data.get("power_profile")
    if not isinstance(profile, dict):
        return summary
    profile_summary = profile.get("summary")
    if not isinstance(profile_summary, dict):
        return summary

    merged = dict(summary)
    power = merged.get("power", {})
    if not isinstance(power, dict):
        power = {}
    if _is_missing(power.get("mean_w")) and isinstance(profile_summary.get("mean_power_watts"), (int, float)):
        power["mean_w"] = float(profile_summary["mean_power_watts"])
    if _is_missing(power.get("max_w")) and isinstance(profile_summary.get("peak_power_watts"), (int, float)):
        power["max_w"] = float(profile_summary["peak_power_watts"])
    if _is_missing(power.get("peak_w")) and isinstance(profile_summary.get("peak_power_watts"), (int, float)):
        power["peak_w"] = float(profile_summary["peak_power_watts"])
    if power:
        merged["power"] = power

    temperature = merged.get("temperature", {})
    if not isinstance(temperature, dict):
        temperature = {}
    if _is_missing(temperature.get("mean_c")) and isinstance(
        profile_summary.get("mean_temperature_celsius"),
        (int, float),
    ):
        temperature["mean_c"] = float(profile_summary["mean_temperature_celsius"])
    readings = profile.get("step_readings")
    if _is_missing(temperature.get("max_c")) and isinstance(readings, list):
        max_temp = None
        for reading in readings:
            if not isinstance(reading, dict):
                continue
            temp_value = reading.get("temperature_celsius")
            if not isinstance(temp_value, (int, float)):
                continue
            max_temp = float(temp_value) if max_temp is None else max(max_temp, float(temp_value))
        if max_temp is not None:
            temperature["max_c"] = max_temp
    if temperature:
        merged["temperature"] = temperature

    return merged


def _flatten_mapping(
    payload: dict[str, Any],
    *,
    prefix: str = "",
    max_depth: int = 2,
) -> list[tuple[str, Any]]:
    """Flatten nested mappings into key/value pairs for display."""
    rows: list[tuple[str, Any]] = []

    def _walk(mapping: dict[str, Any], current_prefix: str, depth: int) -> None:
        for key, value in mapping.items():
            key_text = str(key)
            path = f"{current_prefix}.{key_text}" if current_prefix else key_text
            if isinstance(value, dict) and depth < max_depth:
                _walk(value, path, depth + 1)
            else:
                rows.append((path, value))

    _walk(payload, prefix, 1)
    return rows


def _add_run_metadata_tables(
    report: Any,
    data: dict[str, Any],
    merged_summary: dict[str, Any],
) -> None:
    """Add run/platform metadata tables for parity with Streamlit metadata view."""
    section_added = False
    rows: list[list[str]] = []
    overview_pairs = [
        ("Collector", data.get("collector_name")),
        ("Profile", data.get("profile")),
        ("Run Label", data.get("run_label")),
        ("Sample Count", merged_summary.get("sample_count")),
        ("Warmup Samples", merged_summary.get("warmup_samples")),
        ("Duration (s)", merged_summary.get("duration_seconds")),
    ]
    validation = data.get("validation", {})
    if isinstance(validation, dict) and "overall_pass" in validation:
        overview_pairs.append(("Validation Status", "PASS" if validation.get("overall_pass") else "FAIL"))
    for label, value in overview_pairs:
        if not _is_missing(value):
            rows.append([label, _format_value(value)])

    if rows:
        report.add_section("Run Metadata", "Platform, configuration, and run metadata captured for this benchmark.")
        report.add_table("Run Overview", ["Field", "Value"], rows, "Run Metadata")
        section_added = True

    for title, payload in [
        ("Platform Metadata", data.get("platform_metadata")),
        ("Inference Configuration", data.get("inference_config")),
        ("Device Information", data.get("device_info")),
        ("Validation Details", data.get("validation")),
    ]:
        if not isinstance(payload, dict) or not payload:
            continue
        flat_rows = [[key, _format_value(value)] for key, value in _flatten_mapping(payload, max_depth=3)]
        if flat_rows:
            if not section_added:
                report.add_section(
                    "Run Metadata",
                    "Platform, configuration, and run metadata captured for this benchmark.",
                )
                section_added = True
            report.add_table(title, ["Field", "Value"], flat_rows, "Run Metadata")


def _add_detailed_summary_table(report: Any, merged_summary: dict[str, Any]) -> None:
    """Add detailed summary metrics table matching Streamlit metric sections."""
    rows: list[list[str]] = []

    # Include top-level summary fields first.
    for field, label in [
        ("sample_count", "Sample Count"),
        ("warmup_samples", "Warmup Samples"),
        ("duration_seconds", "Duration (s)"),
    ]:
        if field in merged_summary and not _is_missing(merged_summary.get(field)):
            rows.append([label, _format_value(merged_summary.get(field)), ""])

    group_specs = [
        ("latency", "ms"),
        ("throughput", "FPS"),
        ("power", "W"),
        ("temperature", "C"),
        ("cpu", "%"),
        ("gpu", "%"),
        ("memory", "MB"),
    ]
    for group_name, default_unit in group_specs:
        group = merged_summary.get(group_name)
        if not isinstance(group, dict) or not group:
            continue
        for key, value in group.items():
            if _is_missing(value):
                continue
            metric_name = f"{group_name}.{key}"
            unit = default_unit
            if isinstance(key, str) and key.endswith("_percent"):
                unit = "%"
            elif isinstance(key, str) and key.endswith("_fps"):
                unit = "FPS"
            elif isinstance(key, str) and key.endswith("_ms"):
                unit = "ms"
            elif isinstance(key, str) and key.endswith("_w"):
                unit = "W"
            elif isinstance(key, str) and key.endswith("_c"):
                unit = "C"
            rows.append([metric_name, _format_value(value), unit])

    if rows:
        report.add_section("Summary Details", "Expanded summary metrics generated from the run payload.")
        report.add_table("Detailed Summary Metrics", ["Metric", "Value", "Unit"], rows, "Summary Details")


def _add_result_purpose_table(report: Any, merged_summary: dict[str, Any]) -> None:
    """Add plain-English explanation for each key result metric."""
    rows: list[list[str]] = []
    top_level = [
        (
            "sample_count",
            "Total collected measurement points in this run.",
            "Higher sample counts increase confidence in regression and trend decisions.",
        ),
        (
            "warmup_samples",
            "Initial samples excluded from steady-state analysis.",
            "Prevents startup transients from distorting production-latency conclusions.",
        ),
        (
            "duration_seconds",
            "Total measured run time in seconds.",
            "Validates benchmark window length and comparability between runs.",
        ),
    ]
    for metric, purpose, impact in top_level:
        if not _is_missing(merged_summary.get(metric)):
            rows.append([metric, purpose, impact])

    metric_guides = {
        "latency.p50_ms": (
            "Median response latency.",
            "Represents typical user experience and baseline responsiveness.",
        ),
        "latency.p95_ms": (
            "95th percentile latency.",
            "Captures tail latency for near-worst-case requests.",
        ),
        "latency.p99_ms": (
            "99th percentile latency.",
            "Primary straggler signal; spikes indicate queueing, throttling, or contention.",
        ),
        "latency.mean_ms": (
            "Average latency across measured samples.",
            "Useful trend metric, but can hide tail outliers.",
        ),
        "throughput.mean_fps": (
            "Average processed samples per second.",
            "Primary capacity metric for planning and SLO sizing.",
        ),
        "throughput.min_fps": (
            "Lowest observed throughput in steady state.",
            "Highlights instability or periodic slowdowns under load.",
        ),
        "power.mean_w": (
            "Average power draw in watts.",
            "Used for energy-cost modeling and efficiency tracking.",
        ),
        "power.max_w": (
            "Peak power draw in watts.",
            "Helps detect power spikes and validate power envelope constraints.",
        ),
        "temperature.max_c": (
            "Maximum observed device temperature.",
            "High peaks correlate with thermal throttling risk and performance collapse.",
        ),
        "cpu.mean_percent": (
            "Average CPU utilization.",
            "Identifies host bottlenecks, preprocessing pressure, or dataloader saturation.",
        ),
        "gpu.mean_percent": (
            "Average GPU utilization.",
            "Low utilization with high latency usually indicates input or sync bottlenecks.",
        ),
        "memory.mean_mb": (
            "Average memory footprint in MB.",
            "Tracks memory pressure and headroom for larger batch sizes/models.",
        ),
    }

    for group_name, group_values in merged_summary.items():
        if not isinstance(group_values, dict):
            continue
        for metric_key in group_values:
            metric_name = f"{group_name}.{metric_key}"
            guide = metric_guides.get(metric_name)
            if guide is not None:
                rows.append([metric_name, guide[0], guide[1]])

    if rows:
        report.add_section(
            "Result Guide",
            "Plain-English purpose of each key result and how to interpret it for decisions.",
        )
        report.add_table(
            "Metric Purpose Guide",
            ["Metric", "What It Means", "Why It Matters"],
            rows,
            "Result Guide",
        )


def _add_multi_run_column_guide(report: Any) -> None:
    """Explain each consolidated run-overview column for comparison reports."""
    rows = [
        ["Run Label", "Unique identifier for one benchmark run.", "Use this as the join key across reports/artifacts."],
        [
            "Device",
            "Detected target hardware for the run.",
            "Separates CPU/GPU/SOC differences in performance outcomes.",
        ],
        ["Accelerator", "Requested backend/accelerator setting.", "Verifies run intent matches execution environment."],
        [
            "Precision",
            "Numerical precision mode (e.g., fp32/fp16/bf16).",
            "Directly impacts latency, throughput, and quality tradeoffs.",
        ],
        ["Batch", "Batch size used during inference/training.", "Primary lever for throughput and memory behavior."],
        [
            "Streams",
            "Execution/concurrency streams setting.",
            "Higher concurrency can improve throughput or worsen jitter.",
        ],
        [
            "Warmup",
            "Warmup iterations executed before measurement.",
            "Ensures fair steady-state comparison across runs.",
        ],
        ["Iterations", "Measured benchmark iterations.", "Controls run stability and statistical confidence."],
        ["Samples", "Number of recorded samples.", "Low values reduce reliability of percentile metrics."],
        ["Duration (s)", "Measured wall-clock duration.", "Required for consistent cross-run comparison windows."],
        ["P99 (ms)", "99th percentile latency.", "Best single indicator for tail regressions and stragglers."],
        [
            "Mean Throughput (FPS)",
            "Average throughput in samples/sec.",
            "Capacity metric for provisioning and scaling decisions.",
        ],
        ["Mean Power (W)", "Average power draw.", "Enables cost/performance and efficiency analysis."],
        ["Max Temp (C)", "Peak observed temperature.", "Flags thermal headroom issues and throttling risk."],
    ]
    report.add_section("Result Guide", "Purpose of each column in the consolidated comparison output.")
    report.add_table("Run Overview Column Guide", ["Column", "What It Means", "Why It Matters"], rows, "Result Guide")


def _add_raw_data_preview_table(
    report: Any,
    report_df: Any | None,
    *,
    section_name: str = "Raw Data",
    section_description: str = "Preview of collected samples used to compute summary metrics and charts.",
    table_title: str | None = None,
    max_rows: int = 120,
) -> None:
    """Add a bounded raw-data preview table for report/Streamlit parity."""
    if report_df is None or len(report_df) == 0:
        return

    preferred_columns = [
        "elapsed_seconds",
        "latency_ms",
        "cpu_percent",
        "gpu_percent",
        "memory_used_mb",
        "memory_percent",
        "power_w",
        "temperature_c",
        "throughput_fps",
        "is_warmup",
    ]
    columns = [column for column in preferred_columns if column in report_df.columns]
    if not columns:
        columns = list(report_df.columns[: min(10, len(report_df.columns))])
    if not columns:
        return

    preview = report_df[columns].head(max_rows)
    rows: list[list[str]] = []
    for _, sample_row in preview.iterrows():
        rows.append([_format_value(sample_row[column]) for column in columns])

    if rows:
        report.add_section(section_name, section_description)
        report.add_table(
            table_title or f"Sample Preview (first {len(rows)} rows)",
            columns,
            rows,
            section_name,
        )


def _add_run_detail_table(
    report: Any,
    run: dict[str, Any],
    run_name: str,
    merged_summary: dict[str, Any],
    *,
    section_name: str,
) -> None:
    """Add full metadata/config/summary table for one run."""
    rows: list[list[str]] = [["run_label", run_name]]

    for field, value in [
        ("collector_name", run.get("collector_name")),
        ("profile", run.get("profile")),
        ("sample_count", merged_summary.get("sample_count")),
        ("warmup_samples", merged_summary.get("warmup_samples")),
        ("duration_seconds", merged_summary.get("duration_seconds")),
    ]:
        if not _is_missing(value):
            rows.append([field, _format_value(value)])

    for prefix, payload in [
        ("platform_metadata", run.get("platform_metadata")),
        ("inference_config", run.get("inference_config")),
        ("device_info", run.get("device_info")),
        ("validation", run.get("validation")),
        ("summary", merged_summary),
    ]:
        if not isinstance(payload, dict) or not payload:
            continue
        for key, value in _flatten_mapping(payload, prefix=prefix, max_depth=4):
            if _is_missing(value):
                continue
            rows.append([key, _format_value(value)])

    report.add_table(
        title=f"Run Detail Fields: {run_name}",
        headers=["Field", "Value"],
        rows=rows,
        section=section_name,
    )


def _add_prefixed_plotly_charts_for_run(
    report: Any,
    report_df: Any | None,
    merged_summary: dict[str, Any],
    *,
    run_name: str,
) -> None:
    """Add Plotly charts with section names prefixed by run label."""
    if report_df is None or len(report_df) == 0:
        return

    try:
        from autoperfpy.reports import charts as shared_charts
    except Exception:
        return
    if not shared_charts.is_available():
        return

    sections = shared_charts.build_all_charts(report_df, merged_summary)
    if not sections:
        return

    plot_counter = [0]

    def _next_plotly_id() -> str:
        plot_counter[0] += 1
        return f"plotly_run_{plot_counter[0]}"

    for section_label, charts in sections.items():
        section_name = f"{run_name} | {section_label}"
        report.add_section(section_name, f"{section_label} charts for run {run_name}.")
        for caption, fig in charts:
            shared_charts.apply_report_figure_style(fig)
            fig.update_layout(
                autosize=True,
                height=380,
                margin=dict(l=50, r=50, t=50, b=50),
            )
            html = fig.to_html(
                full_html=False,
                include_plotlyjs=False,
                div_id=_next_plotly_id(),
                config={
                    "responsive": True,
                    "displayModeBar": True,
                    "displaylogo": False,
                    "scrollZoom": True,
                },
            )
            report.add_html_figure(html, caption=caption, section=section_name)


def prepare_report_dataframe_and_summary(
    data: dict[str, Any],
    *,
    df: Any | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    """Build a report dataframe and a complete summary from payload data."""
    from autoperfpy.reports import charts as shared_charts

    summary_raw = data.get("summary", {})
    summary = dict(summary_raw) if isinstance(summary_raw, dict) else {}
    samples = data.get("samples", [])

    report_df = df
    if report_df is None and isinstance(samples, list) and samples:
        report_df = shared_charts.samples_to_dataframe(samples)

    if report_df is not None:
        if "throughput" in report_df.columns and "throughput_fps" not in report_df.columns:
            report_df["throughput_fps"] = report_df["throughput"]
        _fill_dataframe_power_columns(report_df, data)
        shared_charts.ensure_throughput_column(report_df)
        computed_summary = shared_charts.compute_summary_from_dataframe(report_df)
        summary = _merge_missing(summary, computed_summary)
    summary = _enrich_summary_from_power_profile(summary, data)

    if _is_missing(summary.get("sample_count")):
        sample_count = data.get("sample_count")
        if _is_missing(sample_count):
            sample_count = len(report_df) if report_df is not None else len(samples) if isinstance(samples, list) else 0
        summary["sample_count"] = sample_count

    if _is_missing(summary.get("warmup_samples")):
        warmup_samples = data.get("warmup_samples")
        if _is_missing(warmup_samples) and report_df is not None and "is_warmup" in report_df.columns:
            warmup_series = report_df["is_warmup"].fillna(False)
            warmup_samples = int(warmup_series.astype(bool).sum())
        if not _is_missing(warmup_samples):
            summary["warmup_samples"] = warmup_samples

    if _is_missing(summary.get("duration_seconds")):
        start_time = data.get("start_time")
        end_time = data.get("end_time")
        duration_seconds = None
        if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
            duration_seconds = max(0.0, float(end_time) - float(start_time))
        elif report_df is not None and "elapsed_seconds" in report_df.columns and len(report_df) > 0:
            try:
                duration_seconds = float(report_df["elapsed_seconds"].max())
            except (TypeError, ValueError):
                duration_seconds = None
        if duration_seconds is not None:
            summary["duration_seconds"] = duration_seconds

    return report_df, summary


def populate_standard_html_report(
    report: Any,
    data: dict[str, Any],
    *,
    data_source: str | None = None,
    df: Any | None = None,
    summary: dict[str, Any] | None = None,
    add_charts: bool = True,
    chart_engine: str = "chartjs",
) -> tuple[Any | None, dict[str, Any]]:
    """Populate metadata, summary cards, and charts for benchmark-style reports."""
    from autoperfpy.reports import charts as shared_charts

    report_df, merged_summary = prepare_report_dataframe_and_summary(data, df=df)
    if isinstance(summary, dict) and summary:
        merged_summary = _merge_missing(summary, merged_summary)

    if data_source:
        report.add_metadata("Data Source", data_source)

    collector = data.get("collector_name")
    if collector:
        report.add_metadata("Collector", collector)

    if data.get("profile"):
        report.add_metadata("Profile", data["profile"])
    if data.get("run_label"):
        report.add_metadata("Run Label", data["run_label"])

    platform_meta = data.get("platform_metadata", {}) if isinstance(data, dict) else {}
    device_info = data.get("device_info", {}) if isinstance(data, dict) else {}
    inference_cfg = data.get("inference_config", {}) if isinstance(data, dict) else {}

    device_name = None
    if isinstance(device_info, dict):
        device_name = device_info.get("device_name")
    if _is_missing(device_name) and isinstance(platform_meta, dict):
        device_name = platform_meta.get("device_name")
    if _is_missing(device_name) and isinstance(inference_cfg, dict):
        device_name = inference_cfg.get("accelerator")
    if device_name:
        report.add_metadata("Device", device_name)

    if isinstance(inference_cfg, dict):
        if inference_cfg.get("precision"):
            report.add_metadata("Precision", inference_cfg["precision"])
        if inference_cfg.get("batch_size") is not None:
            report.add_metadata("Batch Size", str(inference_cfg["batch_size"]))

    def _add_meta_if_present(label: str, value: Any) -> None:
        if _is_missing(value):
            return
        report.add_metadata(label, _format_value(value))

    if isinstance(platform_meta, dict):
        _add_meta_if_present("GPU", platform_meta.get("gpu_model") or platform_meta.get("gpu"))
        _add_meta_if_present("CPU", platform_meta.get("cpu_model") or platform_meta.get("cpu"))
        _add_meta_if_present("SoC", platform_meta.get("soc"))
        _add_meta_if_present("Power Mode", platform_meta.get("power_mode"))
        _add_meta_if_present("OS", platform_meta.get("os"))

    if isinstance(inference_cfg, dict):
        _add_meta_if_present("Accelerator", inference_cfg.get("accelerator"))
        _add_meta_if_present("Streams", inference_cfg.get("streams"))
        _add_meta_if_present("Warmup Runs", inference_cfg.get("warmup_runs"))
        _add_meta_if_present("Iterations", inference_cfg.get("iterations"))

    sample_count = merged_summary.get("sample_count") or data.get("sample_count") or len(data.get("samples", []))
    _add_meta_if_present("Samples", sample_count)
    _add_meta_if_present("Duration (s)", merged_summary.get("duration_seconds"))
    report.add_summary_item("Samples", sample_count, "", "neutral")

    latency = merged_summary.get("latency", {})
    if isinstance(latency, dict) and latency:
        report.add_summary_item("P99 Latency", f"{latency.get('p99_ms', 0):.2f}", "ms", "neutral")
        if latency.get("p95_ms") is not None:
            report.add_summary_item("P95 Latency", f"{latency.get('p95_ms', 0):.2f}", "ms", "neutral")
        report.add_summary_item("P50 Latency", f"{latency.get('p50_ms', 0):.2f}", "ms", "neutral")
        report.add_summary_item("Mean Latency", f"{latency.get('mean_ms', 0):.2f}", "ms", "neutral")

    throughput = merged_summary.get("throughput", {})
    if isinstance(throughput, dict) and throughput:
        report.add_summary_item("Mean Throughput", f"{throughput.get('mean_fps', 0):.1f}", "FPS", "neutral")
        if throughput.get("min_fps") is not None:
            report.add_summary_item("Min Throughput", f"{throughput.get('min_fps', 0):.1f}", "FPS", "neutral")

    power = merged_summary.get("power", {})
    if isinstance(power, dict) and power.get("mean_w") is not None:
        report.add_summary_item("Mean Power", f"{power.get('mean_w', 0):.1f}", "W", "neutral")
    if isinstance(power, dict) and power.get("max_w") is not None:
        report.add_summary_item("Max Power", f"{power.get('max_w', 0):.1f}", "W", "neutral")

    gpu = merged_summary.get("gpu", {})
    if isinstance(gpu, dict) and gpu.get("mean_percent") is not None:
        report.add_summary_item("Avg GPU", f"{gpu.get('mean_percent', 0):.1f}", "%", "neutral")

    cpu = merged_summary.get("cpu", {})
    if isinstance(cpu, dict) and cpu.get("mean_percent") is not None:
        report.add_summary_item("Avg CPU", f"{cpu.get('mean_percent', 0):.1f}", "%", "neutral")

    memory = merged_summary.get("memory", {})
    if isinstance(memory, dict) and memory.get("mean_mb") is not None:
        report.add_summary_item("Mean Memory", f"{memory.get('mean_mb', 0):.0f}", "MB", "neutral")

    temperature = merged_summary.get("temperature", {})
    if isinstance(temperature, dict) and temperature.get("max_c") is not None:
        report.add_summary_item("Max Temp", f"{temperature.get('max_c', 0):.1f}", "C", "neutral")

    if merged_summary.get("duration_seconds") is not None:
        report.add_summary_item("Duration", f"{float(merged_summary.get('duration_seconds', 0)):.1f}", "s", "neutral")

    validation = data.get("validation", {})
    if isinstance(validation, dict) and validation:
        status = "good" if validation.get("overall_pass") else "critical"
        report.add_summary_item("Run Status", "PASS" if validation.get("overall_pass") else "FAIL", "", status)

    _add_run_metadata_tables(report, data, merged_summary)
    _add_detailed_summary_table(report, merged_summary)
    _add_result_purpose_table(report, merged_summary)

    if add_charts and report_df is not None and len(report_df) > 0:
        shared_charts.add_charts_to_html_report(report, report_df, merged_summary, chart_engine=chart_engine)
    _add_raw_data_preview_table(report, report_df)

    return report_df, merged_summary


def populate_multi_run_html_report(
    report: Any,
    runs: list[dict[str, Any]],
    *,
    data_source: str | None = None,
    include_run_details: bool = False,
    chart_engine: str = "chartjs",
) -> None:
    """Populate a consolidated multi-run report view with all run labels."""
    valid_runs = [run for run in runs if isinstance(run, dict) and run]
    if not valid_runs:
        return

    run_names = [
        run.get("run_label") or run.get("collector_name") or f"Run {idx + 1}" for idx, run in enumerate(valid_runs)
    ]

    if data_source:
        report.add_metadata("Data Source", data_source)
    report.add_metadata("Run Count", str(len(valid_runs)))
    report.add_metadata("Run Labels", ", ".join(str(name) for name in run_names))

    total_samples = 0
    best_run_name = None
    best_run_throughput = None
    device_names: set[str] = set()
    accelerator_names: set[str] = set()
    precision_names: set[str] = set()
    overview_rows: list[list[str]] = []
    metadata_rows: list[list[str]] = []
    for idx, run in enumerate(valid_runs):
        run_name = str(run_names[idx])
        summary = run.get("summary", {}) if isinstance(run, dict) else {}
        latency = summary.get("latency", {}) if isinstance(summary, dict) else {}
        throughput = summary.get("throughput", {}) if isinstance(summary, dict) else {}
        power = summary.get("power", {}) if isinstance(summary, dict) else {}
        temperature = summary.get("temperature", {}) if isinstance(summary, dict) else {}
        platform = run.get("platform_metadata", {}) if isinstance(run, dict) else {}
        inference = run.get("inference_config", {}) if isinstance(run, dict) else {}

        samples = summary.get("sample_count") if isinstance(summary, dict) else None
        try:
            sample_count = int(samples) if samples is not None else int(len(run.get("samples", [])))
        except (TypeError, ValueError):
            sample_count = int(len(run.get("samples", [])))
        total_samples += sample_count

        try:
            mean_fps = float(throughput.get("mean_fps")) if throughput.get("mean_fps") is not None else None
        except (TypeError, ValueError):
            mean_fps = None

        if mean_fps is not None and (best_run_throughput is None or mean_fps > best_run_throughput):
            best_run_throughput = mean_fps
            best_run_name = run_name

        p99_value = latency.get("p99_ms") if isinstance(latency, dict) else None
        p99_display = f"{float(p99_value):.2f}" if isinstance(p99_value, (int, float)) else "-"
        thr_display = f"{mean_fps:.2f}" if mean_fps is not None else "-"
        pwr_value = power.get("mean_w") if isinstance(power, dict) else None
        pwr_display = f"{float(pwr_value):.2f}" if isinstance(pwr_value, (int, float)) else "-"
        temp_value = temperature.get("max_c") if isinstance(temperature, dict) else None
        temp_display = f"{float(temp_value):.1f}" if isinstance(temp_value, (int, float)) else "-"
        duration_value = summary.get("duration_seconds") if isinstance(summary, dict) else None
        duration_display = f"{float(duration_value):.1f}" if isinstance(duration_value, (int, float)) else "-"

        device = ""
        if isinstance(platform, dict):
            device = str(platform.get("device_name") or "")
        if _is_missing(device) and isinstance(inference, dict):
            device = str(inference.get("accelerator") or "")
        if _is_missing(device):
            device = "-"
        if device != "-":
            device_names.add(device)

        precision = "-"
        batch_size = "-"
        streams = "-"
        warmup_runs = "-"
        iterations = "-"
        accelerator = "-"
        if isinstance(inference, dict):
            if not _is_missing(inference.get("precision")):
                precision = str(inference.get("precision"))
                precision_names.add(precision)
            if inference.get("batch_size") is not None and not _is_missing(inference.get("batch_size")):
                batch_size = str(inference.get("batch_size"))
            if inference.get("streams") is not None and not _is_missing(inference.get("streams")):
                streams = str(inference.get("streams"))
            if inference.get("warmup_runs") is not None and not _is_missing(inference.get("warmup_runs")):
                warmup_runs = str(inference.get("warmup_runs"))
            if inference.get("iterations") is not None and not _is_missing(inference.get("iterations")):
                iterations = str(inference.get("iterations"))
            if not _is_missing(inference.get("accelerator")):
                accelerator = str(inference.get("accelerator"))
                accelerator_names.add(accelerator)

        overview_rows.append(
            [
                run_name,
                device,
                accelerator,
                precision,
                batch_size,
                streams,
                warmup_runs,
                iterations,
                str(sample_count),
                duration_display,
                p99_display,
                thr_display,
                pwr_display,
                temp_display,
            ]
        )

        for source_name, payload in [
            ("platform_metadata", platform),
            ("inference_config", inference),
        ]:
            if not isinstance(payload, dict) or not payload:
                continue
            for key, value in _flatten_mapping(payload, max_depth=3):
                metadata_rows.append([run_name, f"{source_name}.{key}", _format_value(value)])

    report.add_summary_item("Runs", len(valid_runs), "", "neutral")
    report.add_summary_item("Total Samples", total_samples, "", "neutral")
    if device_names:
        report.add_metadata("Devices", ", ".join(sorted(device_names)))
    if accelerator_names:
        report.add_metadata("Accelerators", ", ".join(sorted(accelerator_names)))
    if precision_names:
        report.add_metadata("Precisions", ", ".join(sorted(precision_names)))
    if best_run_name is not None and best_run_throughput is not None:
        report.add_summary_item("Best Throughput Run", best_run_name, "", "good")
        report.add_summary_item("Best Throughput", f"{best_run_throughput:.2f}", "FPS", "good")

    report.add_section("Run Overview", "All run labels and key metrics in one consolidated table.")
    report.add_table(
        "Run Overview Table",
        [
            "Run Label",
            "Device",
            "Accelerator",
            "Precision",
            "Batch",
            "Streams",
            "Warmup",
            "Iterations",
            "Samples",
            "Duration (s)",
            "P99 (ms)",
            "Mean Throughput (FPS)",
            "Mean Power (W)",
            "Max Temp (C)",
        ],
        overview_rows,
        "Run Overview",
    )
    _add_multi_run_column_guide(report)
    if metadata_rows:
        report.add_section("Run Metadata", "Per-run platform and inference metadata from each input result.")
        report.add_table(
            "Run Metadata Details",
            ["Run Label", "Field", "Value"],
            metadata_rows,
            "Run Metadata",
        )
    report.add_multi_run_comparison(
        valid_runs,
        run_names=[str(name) for name in run_names],
        section="Run Comparison",
        description="Cross-run latency and throughput comparison with summary table.",
    )

    if include_run_details:
        for idx, run in enumerate(valid_runs):
            run_name = str(run_names[idx])
            section_name = f"Run Details: {run_name}"
            report.add_section(
                section_name,
                "Complete metadata and sample coverage for this run (config, summary, raw sample preview, charts).",
            )
            report_df, merged_summary = prepare_report_dataframe_and_summary(run)
            _add_run_detail_table(report, run, run_name, merged_summary, section_name=section_name)
            _add_raw_data_preview_table(
                report,
                report_df,
                section_name=section_name,
                section_description="",
                table_title=f"Raw Sample Preview: {run_name}",
                max_rows=80,
            )
            if str(chart_engine).lower().strip() == "plotly":
                _add_prefixed_plotly_charts_for_run(
                    report,
                    report_df,
                    merged_summary,
                    run_name=run_name,
                )


__all__ = [
    "prepare_report_dataframe_and_summary",
    "populate_standard_html_report",
    "populate_multi_run_html_report",
]
