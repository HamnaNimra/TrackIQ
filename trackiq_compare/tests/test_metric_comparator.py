"""Tests for metric comparator behavior."""

from datetime import datetime

from trackiq_compare.comparator.metric_comparator import MetricComparator
from trackiq_compare.deps import TrackiqResult
from trackiq_core.schema import Metrics, PlatformInfo, RegressionInfo, WorkloadInfo


def make_result(
    throughput: float = 100.0,
    p50: float = 10.0,
    p95: float = 12.0,
    p99: float = 14.0,
    mem: float = 60.0,
    comm=None,
    power=None,
) -> TrackiqResult:
    """Create a TrackiqResult for tests."""
    return TrackiqResult(
        tool_name="tool",
        tool_version="1.0",
        timestamp=datetime(2026, 2, 21, 10, 0, 0),
        platform=PlatformInfo(
            hardware_name="HW",
            os="Linux",
            framework="pytorch",
            framework_version="2.7.0",
        ),
        workload=WorkloadInfo(
            name="w",
            workload_type="inference",
            batch_size=1,
            steps=100,
        ),
        metrics=Metrics(
            throughput_samples_per_sec=throughput,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            latency_p99_ms=p99,
            memory_utilization_percent=mem,
            communication_overhead_percent=comm,
            power_consumption_watts=power,
        ),
        regression=RegressionInfo(
            baseline_id=None, delta_percent=0.0, status="pass", failed_metrics=[]
        ),
    )


def test_identical_results_produce_zero_deltas() -> None:
    """Comparing identical results should produce zero deltas for comparable metrics."""
    result_a = make_result()
    result_b = make_result()
    comparison = MetricComparator("A", "B").compare(result_a, result_b)
    for metric in comparison.comparable_metrics:
        assert metric.abs_delta == 0.0
        assert metric.percent_delta == 0.0


def test_winner_identification_per_metric() -> None:
    """Comparator should identify metric winners based on direction."""
    result_a = make_result(throughput=100.0, p99=15.0)
    result_b = make_result(throughput=120.0, p99=10.0)
    comparison = MetricComparator("A", "B").compare(result_a, result_b)
    assert comparison.metrics["throughput_samples_per_sec"].winner == "B"
    assert comparison.metrics["latency_p99_ms"].winner == "B"


def test_null_metric_handling_not_comparable() -> None:
    """Null metrics in one result should be marked not comparable."""
    result_a = make_result(comm=8.0)
    result_b = make_result(comm=None)
    comparison = MetricComparator("A", "B").compare(result_a, result_b)
    metric = comparison.metrics["communication_overhead_percent"]
    assert metric.comparable is False
    assert metric.winner == "not_comparable"

