"""Health reporting utilities for MiniCluster monitoring."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING, Any, Dict, List

from trackiq_core.configs.config_io import ensure_parent_dir
from minicluster.monitor.anomaly_detector import Anomaly

if TYPE_CHECKING:
    from minicluster.runner.distributed_runner import HealthCheckpoint


class HealthReporter:
    """Generate summary, HTML, and JSON health reports."""

    @staticmethod
    def _build_bar_svg(labels: List[str], values: List[float], title: str) -> str:
        """Build a simple inline SVG bar chart for fully self-contained HTML."""
        if not labels or not values or len(labels) != len(values):
            return f"<p>No data available for {escape(title)}.</p>"

        width = 820
        height = 260
        margin_left = 50
        margin_bottom = 45
        chart_w = width - margin_left - 20
        chart_h = height - 40 - margin_bottom
        max_value = max(values) if max(values) > 0 else 1.0
        bar_w = max(12.0, chart_w / max(1, len(values)) - 8.0)

        bars: List[str] = []
        for idx, value in enumerate(values):
            scaled = (value / max_value) * chart_h
            x = margin_left + idx * (bar_w + 8.0)
            y = 20 + (chart_h - scaled)
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{scaled:.1f}" fill="#cc3333" />'
            )
            bars.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{height - 20}" text-anchor="middle" '
                f'font-size="11" fill="#444">{escape(labels[idx])}</text>'
            )
            bars.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{max(12.0, y - 4.0):.1f}" '
                f'text-anchor="middle" font-size="10" fill="#222">{value:.2f}</text>'
            )

        return (
            f"<h3>{escape(title)}</h3>"
            f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            f'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(title)}">'
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />'
            f'<line x1="{margin_left}" y1="20" x2="{margin_left}" y2="{20 + chart_h}" stroke="#666" />'
            f'<line x1="{margin_left}" y1="{20 + chart_h}" x2="{margin_left + chart_w}" '
            f'y2="{20 + chart_h}" stroke="#666" />'
            + "".join(bars)
            + "</svg>"
        )

    def generate_summary(self, checkpoint: HealthCheckpoint, anomalies: List[Anomaly]) -> str:
        """Generate plain-English cluster health paragraph."""
        healthy_workers = sum(1 for w in checkpoint.workers if w.status == "healthy")
        total_workers = len(checkpoint.workers)
        if any(a.severity == "critical" for a in anomalies):
            verdict = "critical"
        elif anomalies:
            verdict = "degraded"
        else:
            verdict = "healthy"
        worst = (
            next((a.anomaly_type for a in anomalies if a.severity == "critical"), anomalies[0].anomaly_type)
            if anomalies
            else "none"
        )
        return (
            f"Run {checkpoint.run_id} has {healthy_workers}/{total_workers} healthy workers. "
            f"Detected {len(anomalies)} anomalies; worst type is {worst}. "
            f"Overall cluster health is {verdict}."
        )

    def generate_html_report(
        self, checkpoint: HealthCheckpoint, anomalies: List[Anomaly], output_path: str
    ) -> None:
        """Write self-contained HTML health report."""
        ensure_parent_dir(output_path)
        summary = self.generate_summary(checkpoint, anomalies)
        workers_rows = "".join(
            "<tr>"
            f"<td>{w.worker_id}</td><td>{w.step}</td><td>{w.loss:.6f}</td>"
            f"<td>{w.throughput_samples_per_sec:.3f}</td><td>{w.allreduce_time_ms:.3f}</td>"
            f"<td>{escape(w.status)}</td>"
            "</tr>"
            for w in checkpoint.workers
        )
        anomaly_rows = "".join(
            "<tr>"
            f"<td>{a.severity}</td><td>{a.step}</td><td>{a.worker_id}</td>"
            f"<td>{escape(a.anomaly_type)}</td><td>{escape(a.description)}</td>"
            "</tr>"
            for a in sorted(anomalies, key=lambda x: (0 if x.severity == "critical" else 1, x.step))
        )

        throughput_series = [w.throughput_samples_per_sec for w in checkpoint.workers]
        loss_series = [w.loss for w in checkpoint.workers]
        labels = [str(w.worker_id) for w in checkpoint.workers]

        throughput_svg = self._build_bar_svg(labels, throughput_series, "Per-Worker Throughput")
        loss_svg = self._build_bar_svg(labels, loss_series, "Per-Worker Loss")

        html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>MiniCluster Health Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; }}
    th {{ background: #f5f5f5; }}
    .critical {{ color: #b91c1c; font-weight: 700; }}
    .warning {{ color: #a16207; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>MiniCluster Health Report</h1>
  <p>{escape(summary)}</p>
  <h2>Worker Status</h2>
  <table>
    <thead><tr><th>Worker</th><th>Step</th><th>Loss</th><th>Throughput</th><th>AllReduce ms</th><th>Status</th></tr></thead>
    <tbody>{workers_rows}</tbody>
  </table>
  <h2>Anomalies</h2>
  <table>
    <thead><tr><th>Severity</th><th>Step</th><th>Worker</th><th>Type</th><th>Description</th></tr></thead>
    <tbody>{anomaly_rows}</tbody>
  </table>
  {throughput_svg}
  {loss_svg}
</body>
</html>"""
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(html)

    def generate_json_report(
        self, checkpoint: HealthCheckpoint, anomalies: List[Anomaly]
    ) -> Dict[str, Any]:
        """Build machine-readable health report payload."""
        if any(a.severity == "critical" for a in anomalies):
            status = "critical"
        elif anomalies:
            status = "degraded"
        else:
            status = "healthy"

        counts_by_type: Dict[str, int] = {}
        counts_by_severity: Dict[str, int] = {"warning": 0, "critical": 0}
        for anomaly in anomalies:
            counts_by_type[anomaly.anomaly_type] = counts_by_type.get(anomaly.anomaly_type, 0) + 1
            counts_by_severity[anomaly.severity] = counts_by_severity.get(anomaly.severity, 0) + 1

        recommendations: List[str] = []
        if counts_by_type.get("failed_worker", 0) > 0:
            recommendations.append("Inspect worker process logs and node health for failed workers.")
        if counts_by_type.get("slow_worker", 0) > 0:
            recommendations.append("Check resource contention and network saturation for slow workers.")
        if counts_by_type.get("loss_divergence", 0) > 0:
            recommendations.append("Validate data consistency and optimizer state across workers.")
        if counts_by_type.get("allreduce_spike", 0) > 0:
            recommendations.append("Investigate collective communication latency and fabric performance.")
        if counts_by_type.get("stalled_worker", 0) > 0:
            recommendations.append("Check stalled workers for deadlocks or scheduling starvation.")
        if not recommendations:
            recommendations.append("No action required; cluster health is stable.")

        return {
            "status": status,
            "run_id": checkpoint.run_id,
            "completed_steps": checkpoint.completed_steps,
            "total_steps": checkpoint.total_steps,
            "anomaly_counts": {
                "total": len(anomalies),
                "by_severity": counts_by_severity,
                "by_type": counts_by_type,
            },
            "worker_statuses": [
                {
                    "worker_id": w.worker_id,
                    "step": w.step,
                    "status": w.status,
                    "throughput_samples_per_sec": w.throughput_samples_per_sec,
                    "loss": w.loss,
                }
                for w in checkpoint.workers
            ],
            "recommendations": recommendations,
        }
