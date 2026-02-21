"""Smoke tests for tool-specific dashboard classes and launcher errors."""

from datetime import datetime

import pytest

import dashboard as root_dashboard
from autoperfpy.ui.dashboard import AutoPerfDashboard
from minicluster.ui.dashboard import MiniClusterDashboard
from trackiq_compare.ui.dashboard import (
    VENDOR_COLORS,
    CompareDashboard,
    detect_platform_vendor,
)
from trackiq_core.schema import (
    Metrics,
    PlatformInfo,
    RegressionInfo,
    TrackiqResult,
    WorkloadInfo,
)


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
    dash = MiniClusterDashboard(result=_result(tool_name="minicluster", workload_type="training", tool_payload=payload))
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


def test_cluster_health_launcher_requires_result_argument() -> None:
    """Cluster-health mode should require --result path."""
    with pytest.raises(SystemExit, match="--result is required"):
        root_dashboard.main(["--tool", "cluster-health"])


def test_cluster_health_launcher_missing_result_file_raises_clear_system_exit() -> None:
    """Cluster-health mode should fail clearly when --result does not exist."""
    with pytest.raises(SystemExit, match="--result does not exist"):
        root_dashboard.main(["--tool", "cluster-health", "--result", "missing-minicluster.json"])


def test_dashboard_parser_accepts_cluster_health_optional_inputs() -> None:
    """Root parser should accept cluster-health fault and scaling arguments."""
    args = root_dashboard._parse_args(  # type: ignore[attr-defined]
        [
            "--tool",
            "cluster-health",
            "--result",
            "result.json",
            "--fault-report",
            "fault.json",
            "--scaling-runs",
            "run1.json",
            "run2.json",
        ]
    )
    assert args.tool == "cluster-health"
    assert args.result == "result.json"
    assert args.fault_report == "fault.json"
    assert args.scaling_runs == ["run1.json", "run2.json"]


def test_extract_minicluster_payload_unwraps_tool_payload_and_steps_alias() -> None:
    """Cluster-health payload extraction should support wrapped and per_step keys."""
    raw = {
        "tool_payload": {
            "per_step_metrics": [
                {"step": 0, "loss": 1.2, "allreduce_time_ms": 1.0, "compute_time_ms": 2.0},
            ],
            "average_throughput_samples_per_sec": 120.0,
        },
        "metrics": {"throughput_samples_per_sec": 121.0},
    }
    payload = root_dashboard._extract_minicluster_payload(raw)  # type: ignore[attr-defined]
    assert "steps" in payload
    assert isinstance(payload["steps"], list)
    assert payload["average_throughput_samples_per_sec"] == 120.0


def test_extract_step_rows_normalizes_minicluster_step_data() -> None:
    """Cluster-health step-row extraction should normalize numeric fields."""
    rows = root_dashboard._extract_step_rows(  # type: ignore[attr-defined]
        {
            "steps": [
                {
                    "step": 5,
                    "loss": 0.9,
                    "allreduce_time_ms": 1.5,
                    "compute_time_ms": 2.5,
                    "throughput_samples_per_sec": 99.0,
                }
            ]
        }
    )
    assert len(rows) == 1
    assert rows[0]["step"] == 5.0
    assert rows[0]["loss"] == 0.9
    assert rows[0]["allreduce_ms"] == 1.5
    assert rows[0]["compute_ms"] == 2.5


def test_detect_platform_vendor_cases() -> None:
    """Vendor detection should normalize known platform names."""
    assert detect_platform_vendor("AMD MI300X") == "AMD"
    assert detect_platform_vendor("NVIDIA A100") == "NVIDIA"
    assert detect_platform_vendor("Intel Arc A770") == "Intel"
    assert detect_platform_vendor("Apple M2 Pro") == "Apple"
    assert detect_platform_vendor("Qualcomm Snapdragon 8 Gen 3") == "Qualcomm"
    assert detect_platform_vendor("Intel Core i9") in {"CPU", "Intel"}
    assert detect_platform_vendor("Unknown Device XYZ") == "Unknown"


def test_platform_comparison_mode_activation_differs_by_vendor() -> None:
    """Platform comparison mode should activate only when vendors differ."""
    a = _result(tool_name="a", hardware="AMD MI300X")
    b = _result(tool_name="b", hardware="NVIDIA A100")
    dash = CompareDashboard(result_a=a, result_b=b)
    assert dash.is_platform_comparison_mode() is True


def test_platform_comparison_mode_not_active_for_same_vendor() -> None:
    """Platform comparison mode should stay off when vendors match."""
    a = _result(tool_name="a", hardware="NVIDIA A100")
    b = _result(tool_name="b", hardware="NVIDIA H100")
    dash = CompareDashboard(result_a=a, result_b=b)
    assert dash.is_platform_comparison_mode() is False


def test_vendor_color_map_contains_expected_entries() -> None:
    """Vendor color map should define all required vendor keys."""
    for key in ["AMD", "NVIDIA", "Intel", "Apple", "Qualcomm", "CPU", "Unknown"]:
        assert key in VENDOR_COLORS


def test_platform_export_filename_format() -> None:
    """Platform export filename should include both vendors and timestamp."""
    out = CompareDashboard.platform_export_filename("AMD", "NVIDIA", "20260221_235959")
    assert out == "amd_vs_nvidia_comparison_20260221_235959.html"


def test_compare_dashboard_family_delta_rows_show_directional_advantage() -> None:
    """Normalized family deltas should indicate which side has the advantage."""
    left = _result(tool_name="left", hardware="HW-A")
    right = _result(tool_name="right", hardware="HW-B")
    right.metrics.throughput_samples_per_sec = 130.0  # better for right
    right.metrics.performance_per_watt = 1.8  # better for right
    right.metrics.latency_p99_ms = 10.0  # worse for right
    right.metrics.latency_p95_ms = 8.0  # worse for right
    right.metrics.latency_p50_ms = 6.0  # worse for right
    right.metrics.power_consumption_watts = 70.0  # better for right (lower)

    dash = CompareDashboard(result_a=left, result_b=right, label_a="A", label_b="B")
    rows = dash._competitive_metric_rows()
    families = {row["family"]: row for row in dash._metric_family_delta_rows(rows)}

    assert "performance" in families
    assert "latency" in families
    assert "efficiency" in families
    assert float(families["performance"]["normalized_delta_percent"]) > 0
    assert float(families["latency"]["normalized_delta_percent"]) < 0
    assert float(families["efficiency"]["normalized_delta_percent"]) > 0


def test_compare_dashboard_confidence_rows_reflect_missing_metrics() -> None:
    """Confidence rows should flag missing values as insufficient."""
    left = _result(tool_name="left", hardware="HW-A")
    right = _result(tool_name="right", hardware="HW-B")
    right.metrics.communication_overhead_percent = 3.2
    left.metrics.communication_overhead_percent = None
    right.metrics.power_consumption_watts = None

    dash = CompareDashboard(result_a=left, result_b=right, label_a="A", label_b="B")
    confidence_rows = {row["metric"]: row for row in dash._metric_confidence_rows()}

    assert confidence_rows["throughput_samples_per_sec"]["confidence"] == "strong"
    assert confidence_rows["communication_overhead_percent"]["confidence"] == "insufficient"
    assert confidence_rows["power_consumption_watts"]["confidence"] == "insufficient"


def test_compare_dashboard_download_html_uses_compare_report_builder() -> None:
    """Compare dashboard HTML download path should match compare reporter content."""
    left = _result(tool_name="autoperfpy", hardware="HW-A")
    right = _result(tool_name="minicluster", workload_type="training", hardware="HW-B")
    dash = CompareDashboard(result_a=left, result_b=right, label_a="A", label_b="B")
    html = dash._build_html_report(left)

    assert "TrackIQ Comparison Report" in html
    assert "Metric Comparison" in html
    assert "Visual Overview" in html


def test_minicluster_dashboard_download_html_uses_minicluster_report_builder() -> None:
    """MiniCluster dashboard HTML download path should match minicluster HTML reporter."""
    payload = {
        "config": {
            "num_processes": 1,
            "num_steps": 2,
            "batch_size": 4,
            "learning_rate": 0.01,
            "hidden_size": 128,
            "num_layers": 2,
            "collective_backend": "gloo",
            "workload": "mlp",
            "baseline_throughput": 50.0,
            "seed": 42,
            "tdp_watts": 150.0,
        },
        "steps": [
            {
                "step": 0,
                "loss": 1.1,
                "throughput_samples_per_sec": 100.0,
                "allreduce_time_ms": 1.0,
                "compute_time_ms": 2.0,
            },
            {
                "step": 1,
                "loss": 1.0,
                "throughput_samples_per_sec": 102.0,
                "allreduce_time_ms": 1.2,
                "compute_time_ms": 2.2,
            },
        ],
        "total_time_sec": 0.2,
        "final_loss": 1.0,
        "p99_allreduce_ms": 1.2,
        "scaling_efficiency_pct": 95.0,
    }
    result = _result(tool_name="minicluster", workload_type="training", tool_payload=payload)
    dash = MiniClusterDashboard(result=result)
    html = dash._build_html_report(result)

    assert "MiniCluster Performance Report" in html
    assert "Training Graphs" in html
    assert "Scaling Efficiency (%)" in html
    assert "collective_backend" in html
    assert ("plotly-graph-div" in html) or ("<svg" in html)
