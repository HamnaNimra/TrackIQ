"""Tests for launch_dashboard convenience wrapper."""

from __future__ import annotations

from launch_dashboard import _build_parser, _build_streamlit_command


def test_launch_dashboard_parser_accepts_cluster_health_args() -> None:
    """Parser should accept cluster-health options exposed by root dashboard."""
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--tool",
            "cluster-health",
            "--result",
            "mini.json",
            "--fault-report",
            "fault.json",
            "--scaling-runs",
            "s1.json",
            "s2.json",
        ]
    )
    assert args.tool == "cluster-health"
    assert args.result == "mini.json"
    assert args.fault_report == "fault.json"
    assert args.scaling_runs == ["s1.json", "s2.json"]


def test_launch_dashboard_build_command_includes_cluster_health_args() -> None:
    """Streamlit command should forward cluster-health inputs to dashboard.py."""
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--tool",
            "cluster-health",
            "--result",
            "mini.json",
            "--fault-report",
            "fault.json",
            "--scaling-runs",
            "s1.json",
            "s2.json",
        ]
    )
    command = _build_streamlit_command(args)

    assert "--tool" in command and "cluster-health" in command
    assert "--result" in command and "mini.json" in command
    assert "--fault-report" in command and "fault.json" in command
    assert "--scaling-runs" in command
    scaling_idx = command.index("--scaling-runs")
    assert command[scaling_idx + 1 : scaling_idx + 3] == ["s1.json", "s2.json"]
