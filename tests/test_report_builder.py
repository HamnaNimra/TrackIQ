"""Tests for shared HTML report builder helpers."""

import pytest

from autoperfpy.reporting import HTMLReportGenerator
from autoperfpy.reports.report_builder import (
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
