"""Synthetic TrackiqResult generator for deterministic comparison tests."""

from datetime import datetime
from pathlib import Path

from trackiq_compare.deps import (
    Metrics,
    PlatformInfo,
    RegressionInfo,
    TrackiqResult,
    WorkloadInfo,
    save_trackiq_result,
)


def make_synthetic_result(
    tool_name: str,
    hardware_name: str,
    throughput: float,
    p50: float,
    p95: float,
    p99: float,
    memory_percent: float,
    communication_overhead_percent=None,
    power_watts=None,
) -> TrackiqResult:
    """Create a controlled TrackiqResult with known metric values."""
    return TrackiqResult(
        tool_name=tool_name,
        tool_version="0.1.0",
        timestamp=datetime(2026, 2, 21, 12, 0, 0),
        platform=PlatformInfo(
            hardware_name=hardware_name,
            os="Linux 6.8",
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
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            latency_p99_ms=p99,
            memory_utilization_percent=memory_percent,
            communication_overhead_percent=communication_overhead_percent,
            power_consumption_watts=power_watts,
        ),
        regression=RegressionInfo(
            baseline_id=None,
            delta_percent=0.0,
            status="pass",
            failed_metrics=[],
        ),
    )


def write_synthetic_pair(path_a: Path, path_b: Path) -> tuple[TrackiqResult, TrackiqResult]:
    """Write two synthetic result files with known deltas for tests/demos."""
    result_a = make_synthetic_result(
        tool_name="autoperfpy",
        hardware_name="AMD MI300X",
        throughput=100.0,
        p50=10.0,
        p95=12.0,
        p99=14.0,
        memory_percent=60.0,
        communication_overhead_percent=None,
        power_watts=280.0,
    )
    result_b = make_synthetic_result(
        tool_name="autoperfpy",
        hardware_name="NVIDIA A100",
        throughput=108.0,
        p50=9.5,
        p95=11.5,
        p99=13.0,
        memory_percent=58.0,
        communication_overhead_percent=None,
        power_watts=270.0,
    )
    save_trackiq_result(result_a, str(path_a))
    save_trackiq_result(result_b, str(path_b))
    return result_a, result_b
