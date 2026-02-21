"""Live Streamlit dashboard for MiniCluster health monitoring."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from minicluster.monitor.anomaly_detector import AnomalyDetector
from minicluster.monitor.health_reader import HealthReader
from minicluster.runner.distributed_runner import Metrics
from trackiq_core.schema import (
    PlatformInfo,
    RegressionInfo,
    TrackiqResult,
    WorkloadInfo,
)
from trackiq_core.ui import DARK_THEME, TrackiqDashboard, TrackiqTheme
from trackiq_core.ui.components import LossChart, WorkerGrid


class LiveDashboard(TrackiqDashboard):
    """Live monitor view that streams checkpoint updates during training."""

    def __init__(self, health_checkpoint_path: str, theme: TrackiqTheme = DARK_THEME):
        self.health_checkpoint_path = health_checkpoint_path
        self.reader = HealthReader(health_checkpoint_path)
        self.detector = AnomalyDetector()
        self._history: Dict[int, List[Dict[str, float]]] = {}
        self._start_time = time.time()

        placeholder_result = TrackiqResult(
            tool_name="minicluster-monitor",
            tool_version="0.1.0",
            timestamp=datetime.now(timezone.utc),
            platform=PlatformInfo(
                hardware_name="monitor",
                os="unknown",
                framework="pytorch",
                framework_version="unknown",
            ),
            workload=WorkloadInfo(
                name="live_monitor",
                workload_type="training",
                batch_size=0,
                steps=0,
            ),
            metrics=Metrics(
                throughput_samples_per_sec=0.0,
                latency_p50_ms=0.0,
                latency_p95_ms=0.0,
                latency_p99_ms=0.0,
                memory_utilization_percent=0.0,
                communication_overhead_percent=None,
                power_consumption_watts=None,
            ),
            regression=RegressionInfo(
                baseline_id=None, delta_percent=0.0, status="pass", failed_metrics=[]
            ),
        )
        super().__init__(result=placeholder_result, theme=theme, title="MiniCluster Live Monitor")

    def _estimate_eta_seconds(self, completed_steps: int, total_steps: int) -> float:
        elapsed = max(0.0, time.time() - self._start_time)
        if completed_steps <= 0:
            return 0.0
        step_rate = completed_steps / elapsed if elapsed > 0 else 0.0
        if step_rate <= 0:
            return 0.0
        remaining = max(0, total_steps - completed_steps)
        return remaining / step_rate

    def render_body(self) -> None:
        """Render live checkpoint-driven monitoring panels."""
        import streamlit as st

        progress_slot = st.empty()
        worker_slot = st.empty()
        loss_slot = st.empty()
        anomaly_slot = st.empty()
        verdict_slot = st.empty()
        complete_slot = st.empty()

        while True:
            checkpoint = self.reader.read()
            if checkpoint is None:
                progress_slot.info("Waiting for checkpoint file...")
                time.sleep(1)
                if self.reader.is_run_complete():
                    break
                continue

            anomalies = self.detector.detect(checkpoint)
            for worker in checkpoint.workers:
                self._history.setdefault(worker.worker_id, []).append(
                    {"step": float(worker.step), "loss": float(worker.loss)}
                )

            eta_sec = self._estimate_eta_seconds(checkpoint.completed_steps, checkpoint.total_steps)
            elapsed = time.time() - self._start_time

            with progress_slot.container():
                st.subheader("Run Progress")
                st.write(
                    f"Run ID: `{checkpoint.run_id}` | "
                    f"Completed: {checkpoint.completed_steps}/{checkpoint.total_steps} | "
                    f"Elapsed: {elapsed:.1f}s | ETA: {eta_sec:.1f}s"
                )
                frac = (
                    checkpoint.completed_steps / checkpoint.total_steps
                    if checkpoint.total_steps > 0
                    else 0.0
                )
                st.progress(max(0.0, min(1.0, frac)))

            with worker_slot.container():
                workers_payload = [
                    {
                        "worker_id": w.worker_id,
                        "throughput": w.throughput_samples_per_sec,
                        "allreduce_time_ms": w.allreduce_time_ms,
                        "status": w.status,
                        "loss": w.loss,
                    }
                    for w in checkpoint.workers
                ]
                WorkerGrid(workers=workers_payload, theme=self.theme).render()

            with loss_slot.container():
                st.subheader("Loss Curves")
                if checkpoint.workers:
                    chart_series = {}
                    for worker_id, points in self._history.items():
                        chart_series[f"worker_{worker_id}"] = [p["loss"] for p in points]
                    st.line_chart(chart_series)
                else:
                    LossChart(steps=[], loss_values=[], theme=self.theme).render()

            with anomaly_slot.container():
                st.subheader("Anomaly Feed")
                recent = sorted(anomalies, key=lambda x: x.step, reverse=True)[:10]
                if not recent:
                    st.info("No anomalies detected.")
                for anomaly in recent:
                    msg = (
                        f"worker={anomaly.worker_id} step={anomaly.step} "
                        f"type={anomaly.anomaly_type} - {anomaly.description}"
                    )
                    if anomaly.severity == "critical":
                        st.error(msg)
                    else:
                        st.warning(msg)

            with verdict_slot.container():
                if any(a.severity == "critical" for a in anomalies):
                    verdict = "critical"
                    color = self.theme.fail_color
                elif anomalies:
                    verdict = "degraded"
                    color = self.theme.warning_color
                else:
                    verdict = "healthy"
                    color = self.theme.pass_color
                st.markdown(
                    f"""
                    <div style="padding:12px;border-radius:8px;background:{color};color:white;font-weight:700;">
                    Cluster Health: {verdict.upper()}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if checkpoint.is_complete:
                complete_slot.success("Run complete. Monitoring stopped.")
                break
            time.sleep(1)


def main() -> None:
    """Entrypoint for streamlit run live_dashboard.py -- --checkpoint ..."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args, _ = parser.parse_known_args()
    dashboard = LiveDashboard(args.checkpoint)
    dashboard.configure_page()
    dashboard.apply_theme(dashboard.theme)
    dashboard.render_body()


if __name__ == "__main__":
    main()

