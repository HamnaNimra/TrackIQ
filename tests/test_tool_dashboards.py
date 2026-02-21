"""Smoke tests for tool-specific dashboard classes and launcher errors."""

from datetime import datetime

import pytest

from autoperfpy.ui.dashboard import AutoPerfDashboard
from minicluster.ui.dashboard import MiniClusterDashboard
from trackiq_compare.ui.dashboard import CompareDashboard
from trackiq_core.schema import (
    Metrics,
    PlatformInfo,
    RegressionInfo,
    TrackiqResult,
    WorkloadInfo,
)
import dashboard as root_dashboard


def _result(
    tool_name: str = "tool",
    workload_type: str = "inference",
    hardware: str = "CPU",
    tool_payload=None,
) -> TrackiqResult:
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
            workload_type=workload_type,  # type: ignore[arg-type]
            batch_size=4,
            steps=5,
        ),
        metrics=Metrics(
            throughput_samples_per_sec=100.0,
            latency_p50_ms=5.0,
            latency_p95_ms=7.0,
            latency_p99_ms=8.0,
            memory_utilization_percent=45.0,
            communication_overhead_percent=None,
            power_consumption_watts=80.0,
            energy_per_step_joules=1.2,
            performance_per_watt=1.25,
            temperature_celsius=55.0,
        ),
        regression=RegressionInfo(
            baseline_id=None,
            delta_percent=1.0,
            status="pass",
            failed_metrics=[],
        ),
        tool_payload=tool_payload,
    )


def test_autoperf_dashboard_component_smoke_to_dict() -> None:
    """AutoPerfDashboard components should provide serializable payloads."""
    dash = AutoPerfDashboard(result=_result(tool_name="autoperfpy"))
    components = dash.build_components()
    for component in components.values():
        component.to_dict()


def test_minicluster_dashboard_component_smoke_to_dict() -> None:
    """MiniClusterDashboard components should provide serializable payloads."""
    payload = {
        "workers": [
            {
                "worker_id": "w0",
                "throughput": 123.0,
                "allreduce_time_ms": 2.1,
                "status": "healthy",
                "loss": 1.1,
            }
        ],
        "steps": [
            {"step": 0, "loss": 1.1},
            {"step": 1, "loss": 1.0},
        ],
        "faults_detected": {"slow_workers": []},
    }
    dash = MiniClusterDashboard(
        result=_result(tool_name="minicluster", workload_type="training", tool_payload=payload)
    )
    components = dash.build_components()
    for component in components.values():
        component.to_dict()


def test_compare_dashboard_component_smoke_to_dict() -> None:
    """CompareDashboard components should provide serializable payloads."""
    left = _result(tool_name="autoperfpy", hardware="HW-A")
    right = _result(tool_name="minicluster", workload_type="training", hardware="HW-B")
    dash = CompareDashboard(result_a=left, result_b=right, label_a="A", label_b="B")
    components = dash.build_components()
    for component in components.values():
        component.to_dict()


def test_dashboard_launcher_missing_file_raises_clear_system_exit() -> None:
    """Launcher should fail clearly when input result file path does not exist."""
    with pytest.raises(SystemExit, match="--result does not exist"):
        root_dashboard.main(["--tool", "autoperfpy", "--result", "missing-result.json"])

