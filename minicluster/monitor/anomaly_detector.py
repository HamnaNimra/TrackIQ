"""Anomaly detection for MiniCluster worker health snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import TYPE_CHECKING, Any, Dict, List, Literal

if TYPE_CHECKING:
    from minicluster.runner.distributed_runner import HealthCheckpoint


@dataclass
class Anomaly:
    """Detected anomaly for a worker/step."""

    worker_id: int
    step: int
    anomaly_type: str
    severity: Literal["warning", "critical"]
    description: str


class AnomalyDetector:
    """Detect throughput, loss, and communication anomalies from checkpoints."""

    def __init__(self) -> None:
        self._last_step_by_worker: Dict[int, int] = {}
        self._stalled_counts: Dict[int, int] = {}

    def detect(self, checkpoint: HealthCheckpoint) -> List[Anomaly]:
        """Detect anomalies for the current checkpoint."""
        anomalies: List[Anomaly] = []
        workers = checkpoint.workers
        if not workers:
            return anomalies

        throughputs = [w.throughput_samples_per_sec for w in workers]
        losses = [w.loss for w in workers]
        allreduce = [w.allreduce_time_ms for w in workers]

        mean_thr = mean(throughputs)
        mean_loss = mean(losses)
        loss_std = pstdev(losses) if len(losses) > 1 else 0.0
        mean_allreduce = mean(allreduce)

        for worker in workers:
            if worker.throughput_samples_per_sec == 0:
                anomalies.append(
                    Anomaly(
                        worker_id=worker.worker_id,
                        step=worker.step,
                        anomaly_type="failed_worker",
                        severity="critical",
                        description="Worker throughput is zero.",
                    )
                )
            elif mean_thr > 0 and worker.throughput_samples_per_sec < (0.7 * mean_thr):
                anomalies.append(
                    Anomaly(
                        worker_id=worker.worker_id,
                        step=worker.step,
                        anomaly_type="slow_worker",
                        severity="warning",
                        description="Worker throughput is below 70% of cluster mean.",
                    )
                )

            if loss_std > 0 and worker.loss > (mean_loss + (2.0 * loss_std)):
                anomalies.append(
                    Anomaly(
                        worker_id=worker.worker_id,
                        step=worker.step,
                        anomaly_type="loss_divergence",
                        severity="warning",
                        description="Worker loss is >2 standard deviations above mean.",
                    )
                )

            if mean_allreduce > 0 and worker.allreduce_time_ms > (3.0 * mean_allreduce):
                anomalies.append(
                    Anomaly(
                        worker_id=worker.worker_id,
                        step=worker.step,
                        anomaly_type="allreduce_spike",
                        severity="warning",
                        description="Worker allreduce time is >3x cluster mean.",
                    )
                )

            prev_step = self._last_step_by_worker.get(worker.worker_id)
            if prev_step is not None and prev_step == worker.step:
                self._stalled_counts[worker.worker_id] = self._stalled_counts.get(worker.worker_id, 0) + 1
            else:
                self._stalled_counts[worker.worker_id] = 0
            self._last_step_by_worker[worker.worker_id] = worker.step

            if self._stalled_counts.get(worker.worker_id, 0) >= 3:
                anomalies.append(
                    Anomaly(
                        worker_id=worker.worker_id,
                        step=worker.step,
                        anomaly_type="stalled_worker",
                        severity="critical",
                        description="Worker has not progressed for 3 consecutive checkpoints.",
                    )
                )

        return anomalies

    def summarize(self, anomalies: List[Anomaly]) -> Dict[str, Any]:
        """Summarize anomaly list with counts and hot workers."""
        by_severity: Dict[str, int] = {"warning": 0, "critical": 0}
        by_type: Dict[str, int] = {}
        by_worker: Dict[int, int] = {}
        for anomaly in anomalies:
            by_severity[anomaly.severity] = by_severity.get(anomaly.severity, 0) + 1
            by_type[anomaly.anomaly_type] = by_type.get(anomaly.anomaly_type, 0) + 1
            by_worker[anomaly.worker_id] = by_worker.get(anomaly.worker_id, 0) + 1

        hot_workers = [
            worker for worker, _ in sorted(by_worker.items(), key=lambda item: item[1], reverse=True)
        ]
        return {
            "total": len(anomalies),
            "by_severity": by_severity,
            "by_type": by_type,
            "most_affected_workers": hot_workers,
        }
