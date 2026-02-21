"""Tests for summary generator and CLI run command."""

from datetime import datetime

from trackiq_compare.cli import main as cli_main
from trackiq_compare.comparator.metric_comparator import MetricComparator
from trackiq_compare.comparator.summary_generator import SummaryGenerator
from trackiq_compare.deps import (
    Metrics,
    PlatformInfo,
    RegressionInfo,
    TrackiqResult,
    WorkloadInfo,
    save_trackiq_result,
)
from trackiq_compare.tests.synthetic_result_generator import write_synthetic_pair


def _result(throughput: float, p99: float) -> TrackiqResult:
    return TrackiqResult(
        tool_name="tool",
        tool_version="1.0",
        timestamp=datetime(2026, 2, 21, 9, 0, 0),
        platform=PlatformInfo(
            hardware_name="Device",
            os="Linux",
            framework="pytorch",
            framework_version="2.7.0",
        ),
        workload=WorkloadInfo(
            name="resnet50",
            workload_type="inference",
            batch_size=4,
            steps=100,
        ),
        metrics=Metrics(
            throughput_samples_per_sec=throughput,
            latency_p50_ms=10.0,
            latency_p95_ms=12.0,
            latency_p99_ms=p99,
            memory_utilization_percent=70.0,
            communication_overhead_percent=None,
            power_consumption_watts=None,
        ),
        regression=RegressionInfo(
            baseline_id=None, delta_percent=0.0, status="pass", failed_metrics=[]
        ),
    )


def test_summary_non_empty_output() -> None:
    """Summary text should be non-empty."""
    comparison = MetricComparator("A", "B").compare(_result(100.0, 15.0), _result(110.0, 11.0))
    summary = SummaryGenerator(regression_threshold_percent=5.0).generate(comparison)
    assert summary.text.strip()


def test_threshold_flagging() -> None:
    """Regressions beyond threshold should be flagged."""
    # B is worse in latency p99 by 30% (10 -> 13) and threshold is 5%.
    comparison = MetricComparator("A", "B").compare(_result(100.0, 10.0), _result(90.0, 13.0))
    summary = SummaryGenerator(regression_threshold_percent=5.0).generate(comparison)
    assert len(summary.flagged_regressions) >= 1


def test_cli_run_subcommand_executes_without_errors(tmp_path) -> None:
    """CLI run subcommand should complete successfully."""
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    save_trackiq_result(_result(100.0, 15.0), path_a)
    save_trackiq_result(_result(102.0, 14.0), path_b)
    rc = cli_main(["run", str(path_a), str(path_b), "--label-a", "A", "--label-b", "B"])
    assert rc == 0


def test_cli_custom_labels_show_in_output(tmp_path, capsys) -> None:
    """CLI output should include requested custom labels."""
    path_a = tmp_path / "amd.json"
    path_b = tmp_path / "nvidia.json"
    write_synthetic_pair(path_a, path_b)
    rc = cli_main(
        [
            "run",
            str(path_a),
            str(path_b),
            "--label-a",
            "AMD MI300X",
            "--label-b",
            "NVIDIA A100",
        ]
    )
    assert rc == 0
    output = capsys.readouterr().out
    assert "AMD MI300X" in output
    assert "NVIDIA A100" in output
