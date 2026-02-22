"""Tests for shared HTML report builder helpers."""

import pytest

from autoperfpy.reporting import HTMLReportGenerator
from autoperfpy.reports.report_builder import (
    populate_multi_run_html_report,
    populate_standard_html_report,
    prepare_report_dataframe_and_summary,
)

try:
    from autoperfpy.reports import charts as shared_charts

    CHARTS_AVAILABLE = shared_charts.is_available()
except ImportError:
    CHARTS_AVAILABLE = False


@pytest.mark.skipif(not CHARTS_AVAILABLE, reason="trackiq.reporting.charts (pandas/plotly) not available")
def test_populate_standard_html_report_adds_consistent_fields() -> None:
    """Builder should add common metadata, summary cards, and charts."""
    samples = [
        {
            "timestamp": 1.0,
            "metrics": {
                "latency_ms": 20.0,
                "cpu_percent": 45.0,
                "gpu_percent": 62.0,
                "memory_used_mb": 1024.0,
                "memory_total_mb": 8192.0,
                "power_w": 70.0,
            },
        },
        {
            "timestamp": 2.0,
            "metrics": {
                "latency_ms": 22.0,
                "cpu_percent": 50.0,
                "gpu_percent": 68.0,
                "memory_used_mb": 1100.0,
                "memory_total_mb": 8192.0,
                "power_w": 74.0,
            },
        },
    ]
    data = {
        "collector_name": "SyntheticCollector",
        "profile": "ci_smoke",
        "run_label": "cpu_0_fp32_bs1",
        "platform_metadata": {"device_name": "CPU 0"},
        "inference_config": {"precision": "fp32", "batch_size": 1},
        "summary": {},
        "samples": samples,
    }
    report = HTMLReportGenerator()

    df, summary = populate_standard_html_report(report, data, data_source="unit-test")

    assert df is not None
    assert summary.get("latency")
    assert report.metadata["Data Source"] == "unit-test"
    assert report.metadata["Collector"] == "SyntheticCollector"
    assert report.metadata["Profile"] == "ci_smoke"
    assert report.metadata["Run Label"] == "cpu_0_fp32_bs1"
    assert report.metadata["Device"] == "CPU 0"
    assert report.metadata["Precision"] == "fp32"
    assert report.metadata["Batch Size"] == "1"

    summary_labels = {item["label"] for item in report.summary_items}
    for label in [
        "Samples",
        "P99 Latency",
        "P50 Latency",
        "Mean Latency",
        "Mean Throughput",
        "Mean Power",
        "Avg GPU",
        "Avg CPU",
        "Mean Memory",
    ]:
        assert label in summary_labels

    assert len(report.interactive_charts) > 0


@pytest.mark.skipif(not CHARTS_AVAILABLE, reason="trackiq.reporting.charts (pandas/plotly) not available")
def test_prepare_report_dataframe_and_summary_keeps_existing_values() -> None:
    """Explicit summary values should be preserved while missing keys are backfilled."""
    data = {
        "summary": {
            "latency": {
                "p99_ms": 999.0,
            }
        },
        "samples": [
            {"timestamp": 1.0, "metrics": {"latency_ms": 20.0}},
            {"timestamp": 2.0, "metrics": {"latency_ms": 22.0}},
        ],
    }

    _df, summary = prepare_report_dataframe_and_summary(data)

    assert summary["latency"]["p99_ms"] == 999.0
    assert summary["latency"]["p50_ms"] > 0
    assert summary["sample_count"] == 2


@pytest.mark.skipif(not CHARTS_AVAILABLE, reason="trackiq.reporting.charts (pandas/plotly) not available")
def test_prepare_report_dataframe_and_summary_backfills_power_from_power_profile() -> None:
    """Missing sample power/temperature should be backfilled from power_profile step readings."""
    data = {
        "samples": [
            {
                "timestamp": 1.0,
                "metrics": {"latency_ms": 10.0},
                "metadata": {"sample_index": 0},
            },
            {
                "timestamp": 2.0,
                "metrics": {"latency_ms": 11.0},
                "metadata": {"sample_index": 1},
            },
        ],
        "summary": {},
        "power_profile": {
            "step_readings": [
                {"step": -1, "power_watts": 99.0, "temperature_celsius": 60.0},
                {"step": 0, "power_watts": 45.0, "temperature_celsius": 50.0},
                {"step": 1, "power_watts": 47.0, "temperature_celsius": 52.0},
            ],
            "summary": {
                "mean_power_watts": 46.0,
                "peak_power_watts": 47.0,
                "mean_temperature_celsius": 51.0,
            },
        },
    }

    df, summary = prepare_report_dataframe_and_summary(data)

    assert df is not None
    assert "power_w" in df.columns
    assert list(df["power_w"].astype(float)) == [45.0, 47.0]
    assert "temperature_c" in df.columns
    assert list(df["temperature_c"].astype(float)) == [50.0, 52.0]
    assert summary.get("power", {}).get("mean_w") == pytest.approx(46.0)
    assert summary.get("power", {}).get("max_w") == pytest.approx(47.0)
    assert summary.get("temperature", {}).get("mean_c") == pytest.approx(51.0)
    assert summary.get("temperature", {}).get("max_c") == pytest.approx(52.0)


def test_populate_multi_run_html_report_includes_all_run_labels() -> None:
    """Multi-run report should include every run label in overview metadata/table."""
    report = HTMLReportGenerator()
    runs = [
        {
            "run_label": "cpu_0_fp32_bs1",
            "platform_metadata": {"device_name": "CPU"},
            "inference_config": {"precision": "fp32", "batch_size": 1},
            "summary": {
                "sample_count": 10,
                "latency": {"p99_ms": 24.0},
                "throughput": {"mean_fps": 40.0},
                "power": {"mean_w": 55.0},
            },
        },
        {
            "run_label": "nvidia_0_fp16_bs4",
            "platform_metadata": {"device_name": "NVIDIA"},
            "inference_config": {"precision": "fp16", "batch_size": 4},
            "summary": {
                "sample_count": 12,
                "latency": {"p99_ms": 12.0},
                "throughput": {"mean_fps": 90.0},
                "power": {"mean_w": 110.0},
            },
        },
    ]

    populate_multi_run_html_report(report, runs, data_source="unit-test-multi")

    assert report.metadata["Data Source"] == "unit-test-multi"
    assert report.metadata["Run Count"] == "2"
    assert "cpu_0_fp32_bs1" in report.metadata["Run Labels"]
    assert "nvidia_0_fp16_bs4" in report.metadata["Run Labels"]
    table = next(item for item in report.tables if item["title"] == "Run Overview Table")
    labels = {row[0] for row in table["rows"]}
    assert "cpu_0_fp32_bs1" in labels
    assert "nvidia_0_fp16_bs4" in labels
    metadata_table = next(item for item in report.tables if item["title"] == "Run Metadata Details")
    metadata_rows = metadata_table["rows"]
    assert any(row[0] == "cpu_0_fp32_bs1" and row[1] == "platform_metadata.device_name" for row in metadata_rows)
    assert any(row[0] == "nvidia_0_fp16_bs4" and row[1] == "inference_config.precision" for row in metadata_rows)


def test_populate_standard_html_report_adds_metadata_and_detailed_summary_tables() -> None:
    """Single-run reports should include detailed run metadata and expanded summary rows."""
    report = HTMLReportGenerator()
    data = {
        "collector_name": "PsutilCollector",
        "run_label": "cpu_0_fp32_bs2",
        "platform_metadata": {
            "device_name": "CPU 0",
            "cpu": "AMD Ryzen",
            "gpu": "N/A",
            "soc": "N/A",
            "power_mode": "balanced",
        },
        "inference_config": {
            "accelerator": "cpu_0",
            "precision": "fp32",
            "batch_size": 2,
        },
        "summary": {
            "sample_count": 8,
            "warmup_samples": 2,
            "duration_seconds": 1.5,
            "latency": {
                "min_ms": 20.0,
                "p50_ms": 25.0,
                "mean_ms": 27.0,
                "p95_ms": 35.0,
                "p99_ms": 40.0,
                "max_ms": 45.0,
            },
            "throughput": {"mean_fps": 30.0, "min_fps": 22.0},
            "power": {"mean_w": 50.0, "max_w": 65.0},
            "temperature": {"mean_c": 51.0, "max_c": 61.0},
        },
        "samples": [
            {"timestamp": 1.0, "metrics": {"latency_ms": 22.0, "throughput_fps": 30.0}},
            {"timestamp": 2.0, "metrics": {"latency_ms": 28.0, "throughput_fps": 26.0}},
        ],
    }

    populate_standard_html_report(report, data, data_source="unit-test")

    run_meta_tables = [table for table in report.tables if table["section"] == "Run Metadata"]
    assert any(table["title"] == "Run Overview" for table in run_meta_tables)
    assert any(table["title"] == "Platform Metadata" for table in run_meta_tables)
    assert any(table["title"] == "Inference Configuration" for table in run_meta_tables)

    platform_table = next(table for table in run_meta_tables if table["title"] == "Platform Metadata")
    platform_keys = {row[0] for row in platform_table["rows"]}
    assert "device_name" in platform_keys
    assert "cpu" in platform_keys
    assert "power_mode" in platform_keys

    detail_table = next(table for table in report.tables if table["title"] == "Detailed Summary Metrics")
    metric_names = {row[0] for row in detail_table["rows"]}
    assert "latency.p95_ms" in metric_names
    assert "power.max_w" in metric_names
    assert "temperature.max_c" in metric_names
    purpose_table = next(table for table in report.tables if table["title"] == "Metric Purpose Guide")
    purpose_metrics = {row[0] for row in purpose_table["rows"]}
    assert "latency.p99_ms" in purpose_metrics
    assert "throughput.mean_fps" in purpose_metrics
    assert any(table["section"] == "Raw Data" for table in report.tables)


@pytest.mark.skipif(not CHARTS_AVAILABLE, reason="trackiq.reporting.charts (pandas/plotly) not available")
def test_populate_multi_run_html_report_with_run_details_includes_per_run_sections() -> None:
    """Multi-run detail mode should include per-run config tables and chart sections."""
    report = HTMLReportGenerator()
    runs = [
        {
            "run_label": "cpu_0_fp32_bs1",
            "platform_metadata": {"device_name": "CPU"},
            "inference_config": {
                "precision": "fp32",
                "batch_size": 1,
                "accelerator": "cpu_0",
                "warmup_runs": 3,
                "iterations": 20,
                "streams": 1,
            },
            "summary": {},
            "samples": [
                {"timestamp": 1.0, "metrics": {"latency_ms": 20.0, "throughput_fps": 50.0}},
                {"timestamp": 2.0, "metrics": {"latency_ms": 22.0, "throughput_fps": 45.0}},
            ],
        },
        {
            "run_label": "cpu_0_fp16_bs4",
            "platform_metadata": {"device_name": "CPU"},
            "inference_config": {
                "precision": "fp16",
                "batch_size": 4,
                "accelerator": "cpu_0",
                "warmup_runs": 3,
                "iterations": 20,
                "streams": 1,
            },
            "summary": {},
            "samples": [
                {"timestamp": 1.0, "metrics": {"latency_ms": 12.0, "throughput_fps": 80.0}},
                {"timestamp": 2.0, "metrics": {"latency_ms": 13.0, "throughput_fps": 76.0}},
            ],
        },
    ]

    populate_multi_run_html_report(
        report,
        runs,
        include_run_details=True,
        chart_engine="plotly",
    )

    section_names = {section["name"] for section in report.sections}
    assert "Run Details: cpu_0_fp32_bs1" in section_names
    assert any(name.startswith("cpu_0_fp32_bs1 | ") for name in section_names)
    assert len(report.html_figures) > 0

    overview_table = next(item for item in report.tables if item["title"] == "Run Overview Table")
    headers = overview_table["headers"]
    assert "Precision" in headers
    assert "Batch" in headers
    assert "Warmup" in headers
    assert "Iterations" in headers
    assert "Streams" in headers

    detail_table = next(item for item in report.tables if item["title"] == "Run Detail Fields: cpu_0_fp32_bs1")
    detail_rows = {row[0] for row in detail_table["rows"]}
    assert "inference_config.precision" in detail_rows
    assert "inference_config.batch_size" in detail_rows
    run_guide = next(item for item in report.tables if item["title"] == "Run Overview Column Guide")
    guide_columns = {row[0] for row in run_guide["rows"]}
    assert "P99 (ms)" in guide_columns
    assert "Mean Throughput (FPS)" in guide_columns
