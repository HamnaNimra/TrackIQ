"""Tests for minicluster Plotly report generators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minicluster.cli import cmd_report_fault_timeline, cmd_report_heatmap
from minicluster.reporting.plotly_report import (
    generate_cluster_heatmap,
    generate_fault_timeline,
    load_worker_results_from_dir,
)


def test_generate_cluster_heatmap_writes_html_with_straggler_annotation(tmp_path: Path) -> None:
    """Heatmap report should include STRAGGLER annotation for high outlier worker."""
    output = tmp_path / "heatmap.html"
    results = [
        {"worker_id": 0, "allreduce_time_ms": 10.0},
        {"worker_id": 1, "allreduce_time_ms": 9.5},
        {"worker_id": 2, "allreduce_time_ms": 32.0},
    ]
    generate_cluster_heatmap(results, metric="allreduce_time_ms", output_path=str(output))
    html = output.read_text(encoding="utf-8")
    assert "Cluster Heatmap: allreduce_time_ms" in html
    assert "STRAGGLER" in html


def test_generate_fault_timeline_writes_html_with_detection_rate(tmp_path: Path) -> None:
    """Fault timeline report should include title and detection-rate subtitle."""
    output = tmp_path / "fault_timeline.html"
    report = {
        "num_faults": 3,
        "num_detected": 2,
        "results": [
            {
                "fault_type": "slow_worker",
                "was_detected": True,
                "injection_step": 3,
                "detected_step": 5,
            },
            {
                "fault_type": "gradient_sync_anomaly",
                "was_detected": True,
                "injection_step": 8,
                "detected_step": 10,
            },
            {
                "fault_type": "worker_timeout",
                "was_detected": False,
                "injection_step": 12,
            },
        ],
        "loss_curve": [1.0, 0.95, 0.91, 0.88, 0.85, 0.83, 0.81, 0.8, 0.79, 0.78, 0.77, 0.76, 0.75],
    }
    generate_fault_timeline(report, output_path=str(output))
    html = output.read_text(encoding="utf-8")
    assert "Fault Injection Validation Report" in html
    assert "Detection Rate: 2/3 faults caught" in html
    assert "MISSED" in html


def test_load_worker_results_from_dir_extracts_metrics_from_tool_payload(tmp_path: Path) -> None:
    """Loader should extract selected metric values from tool_payload content."""
    worker_a = {
        "tool_payload": {
            "worker_id": 0,
            "p99_allreduce_ms": 1.4,
            "steps": [{"allreduce_time_ms": 1.0}],
        }
    }
    worker_b = {
        "tool_payload": {
            "worker_id": 1,
            "p99_allreduce_ms": 2.2,
            "steps": [{"allreduce_time_ms": 2.0}],
        }
    }
    (tmp_path / "worker_0.json").write_text(json.dumps(worker_a), encoding="utf-8")
    (tmp_path / "worker_1.json").write_text(json.dumps(worker_b), encoding="utf-8")

    rows = load_worker_results_from_dir(str(tmp_path), "p99_allreduce_ms")
    assert rows == [
        {"worker_id": 0, "p99_allreduce_ms": 1.4},
        {"worker_id": 1, "p99_allreduce_ms": 2.2},
    ]


def test_cmd_report_heatmap_generates_file(tmp_path: Path) -> None:
    """CLI heatmap command should generate report from results directory."""
    (tmp_path / "w0.json").write_text(
        json.dumps({"worker_id": 0, "allreduce_time_ms": 1.0}),
        encoding="utf-8",
    )
    (tmp_path / "w1.json").write_text(
        json.dumps({"worker_id": 1, "allreduce_time_ms": 1.2}),
        encoding="utf-8",
    )
    output = tmp_path / "heatmap.html"
    args = argparse.Namespace(
        results_dir=str(tmp_path),
        metric="allreduce_time_ms",
        output=str(output),
    )
    cmd_report_heatmap(args)
    assert output.exists()
    assert "Cluster Heatmap: allreduce_time_ms" in output.read_text(encoding="utf-8")


def test_cmd_report_fault_timeline_generates_file(tmp_path: Path) -> None:
    """CLI fault-timeline command should generate report from fault JSON."""
    report = {
        "num_faults": 1,
        "num_detected": 1,
        "results": [
            {
                "fault_type": "slow_worker",
                "was_detected": True,
                "injection_step": 2,
                "detected_step": 4,
            }
        ],
        "loss_curve": [1.0, 0.9, 0.8, 0.75, 0.72],
    }
    report_path = tmp_path / "fault_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    output = tmp_path / "fault_timeline.html"
    args = argparse.Namespace(json=str(report_path), output=str(output))
    cmd_report_fault_timeline(args)
    assert output.exists()
    assert "Fault Injection Validation Report" in output.read_text(encoding="utf-8")
