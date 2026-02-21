"""CLI helpers for MiniCluster health monitoring."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from minicluster.monitor.anomaly_detector import AnomalyDetector
from minicluster.monitor.health_reader import HealthReader
from minicluster.monitor.health_reporter import HealthReporter


def _status_from_anomalies(anomalies) -> str:
    if any(a.severity == "critical" for a in anomalies):
        return "CRITICAL"
    if anomalies:
        return "DEGRADED"
    return "HEALTHY"


def cmd_monitor_watch(args: argparse.Namespace) -> None:
    """Launch live monitor dashboard once checkpoint path appears."""
    checkpoint = Path(args.checkpoint)
    deadline = time.time() + args.timeout
    while not checkpoint.exists():
        if time.time() >= deadline:
            print(
                f"Checkpoint file not found before timeout: {args.checkpoint}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(
            f"Checkpoint file not found yet: {args.checkpoint}. Retrying in 2s...",
        )
        time.sleep(2)

    module_path = str(Path(__file__).with_name("live_dashboard.py"))
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        module_path,
        "--",
        "--checkpoint",
        args.checkpoint,
    ]
    raise SystemExit(subprocess.call(cmd))


def cmd_monitor_report(args: argparse.Namespace) -> None:
    """Generate HTML or JSON report from current checkpoint file."""
    reader = HealthReader(args.checkpoint)
    checkpoint = reader.read()
    if checkpoint is None:
        print(f"Checkpoint not found or unreadable: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    detector = AnomalyDetector()
    anomalies = detector.detect(checkpoint)
    reporter = HealthReporter()
    if args.json:
        print(json.dumps(reporter.generate_json_report(checkpoint, anomalies), indent=2))
        return
    reporter.generate_html_report(checkpoint, anomalies, args.output)
    print(f"Health report written to: {args.output}")


def cmd_monitor_status(args: argparse.Namespace) -> None:
    """Print one-line health status for CI usage."""
    reader = HealthReader(args.checkpoint)
    checkpoint = reader.read()
    if checkpoint is None:
        print("CRITICAL anomalies=unknown checkpoint_unavailable")
        sys.exit(1)
    detector = AnomalyDetector()
    anomalies = detector.detect(checkpoint)
    status = _status_from_anomalies(anomalies)
    print(f"{status} anomalies={len(anomalies)}")


def register_monitor_subcommand(subparsers) -> None:
    """Register monitor command group under the existing minicluster CLI."""
    monitor = subparsers.add_parser(
        "monitor",
        help="Live health monitoring and reporting for in-progress runs",
    )
    monitor_sub = monitor.add_subparsers(dest="monitor_cmd")

    watch = monitor_sub.add_parser("watch", help="Launch live dashboard monitor")
    watch.add_argument(
        "--checkpoint",
        default="./minicluster_results/health.json",
    )
    watch.add_argument("--timeout", type=float, default=60.0)
    watch.set_defaults(func=cmd_monitor_watch)

    report = monitor_sub.add_parser("report", help="Generate health report")
    report.add_argument("--checkpoint", default="./minicluster_results/health.json")
    report.add_argument("--output", default="health_report.html")
    report.add_argument("--json", action="store_true")
    report.set_defaults(func=cmd_monitor_report)

    status = monitor_sub.add_parser("status", help="Print one-line cluster status")
    status.add_argument("--checkpoint", default="./minicluster_results/health.json")
    status.set_defaults(func=cmd_monitor_status)

