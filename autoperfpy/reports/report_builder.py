"""Shared helpers for building consistent HTML reports from benchmark payloads."""

from __future__ import annotations

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
        shared_charts.ensure_throughput_column(report_df)
        computed_summary = shared_charts.compute_summary_from_dataframe(report_df)
        summary = _merge_missing(summary, computed_summary)

    if _is_missing(summary.get("sample_count")):
        sample_count = data.get("sample_count")
        if _is_missing(sample_count):
            sample_count = len(report_df) if report_df is not None else len(samples) if isinstance(samples, list) else 0
        summary["sample_count"] = sample_count

    return report_df, summary


def populate_standard_html_report(
    report: Any,
    data: dict[str, Any],
    *,
    data_source: str | None = None,
    df: Any | None = None,
    summary: dict[str, Any] | None = None,
    add_charts: bool = True,
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

    sample_count = merged_summary.get("sample_count") or data.get("sample_count") or len(data.get("samples", []))
    report.add_summary_item("Samples", sample_count, "", "neutral")

    latency = merged_summary.get("latency", {})
    if isinstance(latency, dict) and latency:
        report.add_summary_item("P99 Latency", f"{latency.get('p99_ms', 0):.2f}", "ms", "neutral")
        report.add_summary_item("P50 Latency", f"{latency.get('p50_ms', 0):.2f}", "ms", "neutral")
        report.add_summary_item("Mean Latency", f"{latency.get('mean_ms', 0):.2f}", "ms", "neutral")

    throughput = merged_summary.get("throughput", {})
    if isinstance(throughput, dict) and throughput:
        report.add_summary_item("Mean Throughput", f"{throughput.get('mean_fps', 0):.1f}", "FPS", "neutral")

    power = merged_summary.get("power", {})
    if isinstance(power, dict) and power.get("mean_w") is not None:
        report.add_summary_item("Mean Power", f"{power.get('mean_w', 0):.1f}", "W", "neutral")

    gpu = merged_summary.get("gpu", {})
    if isinstance(gpu, dict) and gpu.get("mean_percent") is not None:
        report.add_summary_item("Avg GPU", f"{gpu.get('mean_percent', 0):.1f}", "%", "neutral")

    cpu = merged_summary.get("cpu", {})
    if isinstance(cpu, dict) and cpu.get("mean_percent") is not None:
        report.add_summary_item("Avg CPU", f"{cpu.get('mean_percent', 0):.1f}", "%", "neutral")

    memory = merged_summary.get("memory", {})
    if isinstance(memory, dict) and memory.get("mean_mb") is not None:
        report.add_summary_item("Mean Memory", f"{memory.get('mean_mb', 0):.0f}", "MB", "neutral")

    validation = data.get("validation", {})
    if isinstance(validation, dict) and validation:
        status = "good" if validation.get("overall_pass") else "critical"
        report.add_summary_item("Run Status", "PASS" if validation.get("overall_pass") else "FAIL", "", status)

    if add_charts and report_df is not None and len(report_df) > 0:
        shared_charts.add_charts_to_html_report(report, report_df, merged_summary)

    return report_df, merged_summary


__all__ = [
    "prepare_report_dataframe_and_summary",
    "populate_standard_html_report",
]
