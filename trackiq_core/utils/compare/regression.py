"""
Performance regression detection and baseline comparison.

Provides RegressionDetector, RegressionThreshold, and MetricComparison
for comparing current metrics against stored baselines.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class RegressionThreshold:
    """Threshold configuration for regression detection."""

    latency_percent: float = 5.0
    throughput_percent: float = 5.0
    p99_percent: float = 10.0


@dataclass
class MetricComparison:
    """Result of comparing two metrics."""

    metric_name: str
    baseline_value: float
    current_value: float
    percent_change: float
    is_regression: bool
    threshold: float


class RegressionDetector:
    """Detect performance regressions against baseline metrics."""

    def __init__(self, baseline_dir: str = ".trackiq/baselines"):
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)

    def save_baseline(self, name: str, metrics: Dict[str, Any]) -> None:
        """Save metrics as baseline."""
        baseline_file = self.baseline_dir / f"{name}.json"
        baseline_data = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }
        with open(baseline_file, "w") as f:
            json.dump(baseline_data, f, indent=2)

    def load_baseline(self, name: str) -> Dict[str, Any]:
        """Load baseline metrics."""
        baseline_file = self.baseline_dir / f"{name}.json"
        if not baseline_file.exists():
            raise FileNotFoundError(f"Baseline not found: {name}")

        with open(baseline_file, "r") as f:
            data = json.load(f)
        return data["metrics"]

    def list_baselines(self) -> List[str]:
        """List all available baselines."""
        return [f.stem for f in self.baseline_dir.glob("*.json")]

    def compare_metrics(
        self,
        baseline_metrics: Dict[str, float],
        current_metrics: Dict[str, float],
        thresholds: Optional[RegressionThreshold] = None,
    ) -> Dict[str, MetricComparison]:
        """Compare current metrics against baseline."""
        if thresholds is None:
            thresholds = RegressionThreshold()

        comparisons = {}

        for metric_name, baseline_value in baseline_metrics.items():
            if metric_name not in current_metrics:
                continue

            current_value = current_metrics[metric_name]

            is_latency = any(
                x in metric_name.lower()
                for x in ["latency", "time", "p99", "p95", "p50", "mean"]
            )

            if is_latency:
                threshold = (
                    thresholds.p99_percent
                    if "p99" in metric_name
                    else thresholds.latency_percent
                )
                if baseline_value == 0:
                    percent_change = float("inf") if current_value > 0 else 0.0
                    is_regression = current_value > 0
                else:
                    percent_change = (
                        (current_value - baseline_value) / baseline_value
                    ) * 100
                    is_regression = percent_change > threshold
            else:
                threshold = thresholds.throughput_percent
                if baseline_value == 0:
                    percent_change = 0.0
                    is_regression = False
                else:
                    percent_change = (
                        (baseline_value - current_value) / baseline_value
                    ) * 100
                    is_regression = percent_change > threshold

            comparisons[metric_name] = MetricComparison(
                metric_name=metric_name,
                baseline_value=baseline_value,
                current_value=current_value,
                percent_change=percent_change,
                is_regression=is_regression,
                threshold=threshold,
            )

        return comparisons

    def detect_regressions(
        self,
        baseline_name: str,
        current_metrics: Dict[str, float],
        thresholds: Optional[RegressionThreshold] = None,
    ) -> Dict[str, Any]:
        """Detect regressions in current metrics."""
        baseline_metrics = self.load_baseline(baseline_name)
        comparisons = self.compare_metrics(
            baseline_metrics, current_metrics, thresholds
        )

        regressions = {k: v for k, v in comparisons.items() if v.is_regression}
        improvements = {
            k: v
            for k, v in comparisons.items()
            if not v.is_regression and v.percent_change != 0
        }

        return {
            "baseline": baseline_name,
            "timestamp": datetime.now().isoformat(),
            "total_metrics": len(comparisons),
            "regressions": {k: asdict(v) for k, v in regressions.items()},
            "improvements": {k: asdict(v) for k, v in improvements.items()},
            "has_regressions": len(regressions) > 0,
            "comparisons": {k: asdict(v) for k, v in comparisons.items()},
        }

    def generate_report(
        self,
        baseline_name: str,
        current_metrics: Dict[str, float],
        thresholds: Optional[RegressionThreshold] = None,
    ) -> str:
        """Generate a human-readable regression report."""
        result = self.detect_regressions(baseline_name, current_metrics, thresholds)

        lines = [
            "=" * 70,
            "PERFORMANCE REGRESSION REPORT",
            "=" * 70,
            f"Baseline: {result['baseline']}",
            f"Timestamp: {result['timestamp']}",
            "",
        ]

        if result["regressions"]:
            lines.append("REGRESSIONS DETECTED:")
            lines.append("-" * 70)
            for metric_name, comp in result["regressions"].items():
                lines.append(
                    f"  {metric_name:30} | "
                    f"Baseline: {comp['baseline_value']:10.2f} -> "
                    f"Current: {comp['current_value']:10.2f} | "
                    f"Change: {comp['percent_change']:+.2f}% (Threshold: {comp['threshold']:.1f}%)"
                )
            lines.append("")
        else:
            lines.append("NO REGRESSIONS DETECTED")
            lines.append("")

        if result["improvements"]:
            lines.append("IMPROVEMENTS:")
            lines.append("-" * 70)
            for metric_name, comp in result["improvements"].items():
                lines.append(
                    f"  {metric_name:30} | "
                    f"Baseline: {comp['baseline_value']:10.2f} -> "
                    f"Current: {comp['current_value']:10.2f} | "
                    f"Change: {comp['percent_change']:+.2f}%"
                )
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)
