"""End-to-end integration test for MiniCluster health monitoring."""

from __future__ import annotations

import json

import pytest
import torch

from minicluster.monitor.anomaly_detector import AnomalyDetector
from minicluster.monitor.health_reader import HealthReader
from minicluster.monitor.health_reporter import HealthReporter
from minicluster.runner import RunConfig, run_distributed, save_metrics


@pytest.mark.skipif(
    not torch.distributed.is_available(),
    reason="torch.distributed is unavailable in this environment",
)
def test_health_monitor_end_to_end(tmp_path) -> None:
    """Run a monitored training job and validate end-to-end monitor outputs."""
    checkpoint_path = tmp_path / "health.json"
    result_path = tmp_path / "result.json"
    html_report_path = tmp_path / "health_report.html"

    config = RunConfig(num_steps=20, num_processes=2, seed=42)
    metrics = run_distributed(config, health_checkpoint_path=str(checkpoint_path))
    save_metrics(metrics, str(result_path))

    reader = HealthReader(str(checkpoint_path))
    checkpoint = reader.read()
    assert checkpoint is not None
    assert checkpoint.is_complete is True
    assert checkpoint.completed_steps == 20

    detector = AnomalyDetector()
    anomalies = detector.detect(checkpoint)
    assert isinstance(anomalies, list)

    reporter = HealthReporter()
    json_report = reporter.generate_json_report(checkpoint, anomalies)
    assert json_report["status"] in {"healthy", "degraded", "critical"}

    reporter.generate_html_report(checkpoint, anomalies, str(html_report_path))
    assert html_report_path.exists()
    assert html_report_path.stat().st_size > 0

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    worker_snapshots = payload.get("tool_payload", {}).get("worker_snapshots")
    assert isinstance(worker_snapshots, list)
    assert len(worker_snapshots) > 0

