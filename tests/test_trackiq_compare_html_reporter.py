"""Unit tests for compare HTML reporter visualization parity sections."""

from datetime import datetime

from trackiq_compare.comparator import MetricComparator, SummaryGenerator
from trackiq_compare.reporters import HtmlReporter
from trackiq_core.schema import (
    Metrics,
    PlatformInfo,
    RegressionInfo,
    TrackiqResult,
    WorkloadInfo,
)


def _result(
    tool_name: str = "tool",
    hardware: str = "CPU",
    *,
    metric_overrides: dict[str, float | None] | None = None,
) -> TrackiqResult:
    metrics = Metrics(
        throughput_samples_per_sec=100.0,
        latency_p50_ms=5.0,
        latency_p95_ms=7.0,
        latency_p99_ms=8.0,
        memory_utilization_percent=45.0,
        communication_overhead_percent=1.0,
        power_consumption_watts=80.0,
        energy_per_step_joules=1.2,
        performance_per_watt=1.25,
        temperature_celsius=55.0,
    )
    if metric_overrides:
        for key, value in metric_overrides.items():
            setattr(metrics, key, value)

    return TrackiqResult(
        tool_name=tool_name,
        tool_version="0.1.0",
        timestamp=datetime(2026, 2, 21, 12, 0, 0),
        platform=PlatformInfo(
            hardware_name=hardware,
            os="Linux",
            framework="pytorch",
            framework_version="2.7.0",
        ),
        workload=WorkloadInfo(
            name="demo",
            workload_type="inference",  # type: ignore[arg-type]
            batch_size=4,
            steps=5,
        ),
        metrics=metrics,
        regression=RegressionInfo(
            baseline_id=None,
            delta_percent=1.0,
            status="pass",
            failed_metrics=[],
        ),
    )


def test_compare_html_report_includes_visualization_parity_sections(tmp_path) -> None:
    """Report should include the same compare MVP visualization sections as dashboard."""
    left = _result(tool_name="autoperfpy", hardware="HW-A")
    right = _result(
        tool_name="minicluster",
        hardware="HW-B",
        metric_overrides={
            "throughput_samples_per_sec": 130.0,
            "performance_per_watt": 1.8,
            "latency_p50_ms": 6.0,
            "latency_p95_ms": 8.0,
            "latency_p99_ms": 10.0,
            "power_consumption_watts": 70.0,
        },
    )

    comparison = MetricComparator("A", "B").compare(left, right)
    summary = SummaryGenerator().generate(comparison)
    out = tmp_path / "compare_report.html"
    HtmlReporter().generate(str(out), comparison, summary, left, right)
    html = out.read_text(encoding="utf-8")

    assert "Visual Overview" in html
    assert "Top Normalized Deltas" in html
    assert "Winner Distribution" in html
    assert "Confidence Distribution" in html
    assert "conic-gradient(" in html
    assert "Normalized Metric Deltas" in html
    assert "Metric Family Delta Waterfall" in html
    assert "Metric Availability Confidence Matrix" in html
    assert "throughput_samples_per_sec" in html
    assert "latency_p95_ms" in html
    assert "performance" in html


def test_compare_html_report_family_deltas_preserve_directional_winner_logic() -> None:
    """Family aggregation should preserve normalized winner direction semantics."""
    left = _result(tool_name="left", hardware="HW-A")
    right = _result(
        tool_name="right",
        hardware="HW-B",
        metric_overrides={
            "throughput_samples_per_sec": 130.0,  # right advantage
            "performance_per_watt": 1.8,  # right advantage
            "latency_p50_ms": 6.0,  # right disadvantage
            "latency_p95_ms": 8.0,  # right disadvantage
            "latency_p99_ms": 10.0,  # right disadvantage
            "power_consumption_watts": 70.0,  # right advantage (lower is better)
        },
    )

    comparison = MetricComparator("A", "B").compare(left, right)
    reporter = HtmlReporter()
    normalized_rows = reporter._normalized_metric_delta_rows(comparison)
    family_rows = {
        row["family"]: row
        for row in reporter._metric_family_delta_rows(normalized_rows, comparison.label_a, comparison.label_b)
    }

    assert float(family_rows["performance"]["normalized_delta_percent"]) > 0
    assert family_rows["performance"]["winner"] == "B"
    assert float(family_rows["latency"]["normalized_delta_percent"]) < 0
    assert family_rows["latency"]["winner"] == "A"
    assert float(family_rows["efficiency"]["normalized_delta_percent"]) > 0
    assert family_rows["efficiency"]["winner"] == "B"


def test_compare_html_report_confidence_rows_flag_missing_metrics() -> None:
    """Confidence model should mark partially available metrics as insufficient."""
    left = _result(
        tool_name="left",
        hardware="HW-A",
        metric_overrides={
            "communication_overhead_percent": None,
        },
    )
    right = _result(
        tool_name="right",
        hardware="HW-B",
        metric_overrides={
            "communication_overhead_percent": 3.2,
            "power_consumption_watts": None,
        },
    )

    comparison = MetricComparator("A", "B").compare(left, right)
    reporter = HtmlReporter()
    confidence_rows = {row["metric"]: row for row in reporter._metric_confidence_rows(comparison)}

    assert confidence_rows["throughput_samples_per_sec"]["confidence"] == "strong"
    assert confidence_rows["communication_overhead_percent"]["confidence"] == "insufficient"
    assert confidence_rows["power_consumption_watts"]["confidence"] == "insufficient"
