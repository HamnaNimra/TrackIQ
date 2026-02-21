"""Tests for minicluster monitor package."""

import json
from pathlib import Path

from minicluster.monitor.anomaly_detector import Anomaly, AnomalyDetector
from minicluster.monitor.health_reader import HealthReader
from minicluster.monitor.health_reporter import HealthReporter
from minicluster.monitor.live_dashboard import LiveDashboard
from minicluster.runner.distributed_runner import HealthCheckpoint, WorkerSnapshot


def _checkpoint(
    workers,
    completed_steps: int = 1,
    total_steps: int = 10,
    is_complete: bool = False,
) -> HealthCheckpoint:
    return HealthCheckpoint(
        run_id="run-1",
        total_steps=total_steps,
        completed_steps=completed_steps,
        workers=workers,
        timestamp="2026-02-21T00:00:00",
        is_complete=is_complete,
    )


def test_health_reader_read_returns_none_when_missing(tmp_path) -> None:
    reader = HealthReader(str(tmp_path / "missing.json"), timeout_seconds=0.1)
    assert reader.read() is None


def test_health_reader_read_deserializes_valid_checkpoint(tmp_path) -> None:
    path = tmp_path / "health.json"
    payload = {
        "run_id": "run-1",
        "total_steps": 10,
        "completed_steps": 3,
        "workers": [
            {
                "worker_id": 0,
                "step": 3,
                "loss": 1.2,
                "throughput_samples_per_sec": 100.0,
                "allreduce_time_ms": 1.0,
                "compute_time_ms": 1.0,
                "status": "healthy",
                "timestamp": "2026-02-21T00:00:00",
            }
        ],
        "timestamp": "2026-02-21T00:00:00",
        "is_complete": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    reader = HealthReader(str(path))
    checkpoint = reader.read()
    assert checkpoint is not None
    assert checkpoint.completed_steps == 3
    assert checkpoint.workers[0].worker_id == 0


def test_health_reader_is_run_complete_when_complete_checkpoint(tmp_path) -> None:
    path = tmp_path / "health.json"
    path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "total_steps": 10,
                "completed_steps": 10,
                "workers": [],
                "timestamp": "2026-02-21T00:00:00",
                "is_complete": True,
            }
        ),
        encoding="utf-8",
    )
    reader = HealthReader(str(path))
    _ = reader.read()
    assert reader.is_run_complete() is True


def test_anomaly_detector_detects_slow_worker() -> None:
    detector = AnomalyDetector()
    workers = [
        WorkerSnapshot(0, 1, 1.0, 100.0, 1.0, 1.0, "healthy", "2026-02-21T00:00:00"),
        WorkerSnapshot(1, 1, 1.0, 50.0, 1.0, 1.0, "healthy", "2026-02-21T00:00:00"),
        WorkerSnapshot(2, 1, 1.0, 100.0, 1.0, 1.0, "healthy", "2026-02-21T00:00:00"),
    ]
    anomalies = detector.detect(_checkpoint(workers))
    assert any(a.anomaly_type == "slow_worker" and a.worker_id == 1 for a in anomalies)


def test_anomaly_detector_detects_loss_divergence() -> None:
    detector = AnomalyDetector()
    workers = [
        WorkerSnapshot(i, 1, 1.0, 100.0, 1.0, 1.0, "healthy", "2026-02-21T00:00:00")
        for i in range(9)
    ]
    workers.append(
        WorkerSnapshot(9, 1, 10.0, 100.0, 1.0, 1.0, "healthy", "2026-02-21T00:00:00")
    )
    anomalies = detector.detect(_checkpoint(workers))
    assert any(a.anomaly_type == "loss_divergence" and a.worker_id == 9 for a in anomalies)


def test_anomaly_detector_detects_allreduce_spike() -> None:
    detector = AnomalyDetector()
    workers = [
        WorkerSnapshot(0, 1, 1.0, 100.0, 1.0, 1.0, "healthy", "2026-02-21T00:00:00"),
        WorkerSnapshot(1, 1, 1.0, 100.0, 1.0, 1.0, "healthy", "2026-02-21T00:00:00"),
        WorkerSnapshot(2, 1, 1.0, 100.0, 1.0, 1.0, "healthy", "2026-02-21T00:00:00"),
        WorkerSnapshot(3, 1, 1.0, 100.0, 10.0, 1.0, "healthy", "2026-02-21T00:00:00"),
    ]
    anomalies = detector.detect(_checkpoint(workers))
    assert any(a.anomaly_type == "allreduce_spike" and a.worker_id == 3 for a in anomalies)


def test_anomaly_detector_returns_empty_for_healthy_checkpoint() -> None:
    detector = AnomalyDetector()
    workers = [
        WorkerSnapshot(0, 1, 1.0, 100.0, 1.0, 1.0, "healthy", "2026-02-21T00:00:00"),
        WorkerSnapshot(1, 1, 1.1, 101.0, 1.1, 1.0, "healthy", "2026-02-21T00:00:00"),
        WorkerSnapshot(2, 1, 0.9, 99.0, 0.9, 1.0, "healthy", "2026-02-21T00:00:00"),
    ]
    anomalies = detector.detect(_checkpoint(workers))
    assert anomalies == []


def test_anomaly_detector_summarize_counts() -> None:
    detector = AnomalyDetector()
    anomalies = [
        Anomaly(worker_id=0, step=1, anomaly_type="slow_worker", severity="warning", description=""),
        Anomaly(worker_id=0, step=2, anomaly_type="failed_worker", severity="critical", description=""),
        Anomaly(worker_id=1, step=2, anomaly_type="slow_worker", severity="warning", description=""),
    ]
    summary = detector.summarize(anomalies)
    assert summary["by_severity"]["warning"] == 2
    assert summary["by_severity"]["critical"] == 1
    assert summary["by_type"]["slow_worker"] == 2


def test_health_reporter_generate_summary_contains_verdict_word() -> None:
    reporter = HealthReporter()
    workers = [WorkerSnapshot(0, 1, 1.0, 100.0, 1.0, 1.0, "healthy", "2026-02-21T00:00:00")]
    summary = reporter.generate_summary(_checkpoint(workers), [])
    lowered = summary.lower()
    assert "healthy" in lowered or "degraded" in lowered or "critical" in lowered


def test_health_reporter_generate_json_report_status_healthy() -> None:
    reporter = HealthReporter()
    workers = [WorkerSnapshot(0, 1, 1.0, 100.0, 1.0, 1.0, "healthy", "2026-02-21T00:00:00")]
    payload = reporter.generate_json_report(_checkpoint(workers), [])
    assert payload["status"] == "healthy"


def test_health_reporter_generate_json_report_status_critical_for_failed_worker() -> None:
    reporter = HealthReporter()
    workers = [WorkerSnapshot(0, 1, 1.0, 0.0, 1.0, 1.0, "failed", "2026-02-21T00:00:00")]
    anomalies = [
        Anomaly(
            worker_id=0,
            step=1,
            anomaly_type="failed_worker",
            severity="critical",
            description="zero throughput",
        )
    ]
    payload = reporter.generate_json_report(_checkpoint(workers), anomalies)
    assert payload["status"] == "critical"


def test_live_dashboard_instantiates_without_error() -> None:
    dashboard = LiveDashboard("./minicluster_results/health.json")
    assert dashboard is not None
