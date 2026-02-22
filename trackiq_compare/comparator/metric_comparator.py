"""Metric comparator for canonical TrackIQ results."""

import statistics
from dataclasses import asdict, dataclass, field

from trackiq_compare.deps import TrackiqResult

LOWER_IS_BETTER_METRICS = {
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "ttft_ms",
    "decode_tpt_ms",
    "memory_utilization_percent",
    "communication_overhead_percent",
    "power_consumption_watts",
    "energy_per_step_joules",
}

POWER_METRIC_FIELDS = {
    "power_consumption_watts",
    "energy_per_step_joules",
    "performance_per_watt",
}


@dataclass
class MetricComparison:
    """Comparison details for a single metric."""

    metric_name: str
    value_a: float | None
    value_b: float | None
    comparable: bool
    abs_delta: float | None
    percent_delta: float | None
    winner: str
    winner_margin_percent: float | None
    reason: str = ""


@dataclass
class ConsistencyFinding:
    """Consistency regression finding derived from per-step all-reduce variance."""

    code: str
    label: str
    status: str
    stddev_a_ms: float
    stddev_b_ms: float
    increase_percent: float
    threshold_percent: float
    reason: str


@dataclass
class ComparisonResult:
    """Structured metric comparison output."""

    label_a: str
    label_b: str
    metrics: dict[str, MetricComparison] = field(default_factory=dict)
    consistency_findings: list[ConsistencyFinding] = field(default_factory=list)

    @property
    def comparable_metrics(self) -> list[MetricComparison]:
        """Return comparable metric comparisons."""
        return [item for item in self.metrics.values() if item.comparable]

    @property
    def non_comparable_metrics(self) -> list[MetricComparison]:
        """Return non-comparable metric comparisons."""
        return [item for item in self.metrics.values() if not item.comparable]


class MetricComparator:
    """Compare two TrackiqResult objects metric-by-metric."""

    def __init__(
        self,
        label_a: str = "Result A",
        label_b: str = "Result B",
        variance_threshold_percent: float = 25.0,
    ):
        self.label_a = label_a
        self.label_b = label_b
        self.variance_threshold_percent = float(variance_threshold_percent)

    def compare(self, result_a: TrackiqResult, result_b: TrackiqResult) -> ComparisonResult:
        """Compare all shared metric fields between two results."""
        metrics_a = asdict(result_a.metrics)
        metrics_b = asdict(result_b.metrics)
        all_metric_names = sorted(set(metrics_a.keys()) | set(metrics_b.keys()))
        if all(metrics_a.get(name) is None and metrics_b.get(name) is None for name in POWER_METRIC_FIELDS):
            all_metric_names = [name for name in all_metric_names if name not in POWER_METRIC_FIELDS]

        output = ComparisonResult(label_a=self.label_a, label_b=self.label_b)

        for name in all_metric_names:
            value_a = metrics_a.get(name)
            value_b = metrics_b.get(name)
            output.metrics[name] = self._compare_metric(name, value_a, value_b)

        output.consistency_findings = self._consistency_findings(result_a, result_b)
        return output

    def _compare_metric(self, name: str, value_a: float | None, value_b: float | None) -> MetricComparison:
        """Compare an individual metric with null-safe handling."""
        if value_a is None or value_b is None:
            return MetricComparison(
                metric_name=name,
                value_a=value_a,
                value_b=value_b,
                comparable=False,
                abs_delta=None,
                percent_delta=None,
                winner="not_comparable",
                winner_margin_percent=None,
                reason="Metric missing/null in one result",
            )

        delta = float(value_b) - float(value_a)
        abs_delta = abs(delta)
        if value_a == 0:
            percent_delta = 0.0 if value_b == 0 else float("inf")
        else:
            percent_delta = (delta / float(value_a)) * 100.0

        lower_is_better = name in LOWER_IS_BETTER_METRICS
        margin: float | None
        if delta == 0:
            winner = "tie"
            margin = 0.0
        else:
            if lower_is_better:
                winner = self.label_b if value_b < value_a else self.label_a
            else:
                winner = self.label_b if value_b > value_a else self.label_a
            margin = abs(percent_delta) if percent_delta != float("inf") else None

        return MetricComparison(
            metric_name=name,
            value_a=float(value_a),
            value_b=float(value_b),
            comparable=True,
            abs_delta=abs_delta,
            percent_delta=percent_delta,
            winner=winner,
            winner_margin_percent=margin,
        )

    @staticmethod
    def _extract_allreduce_series(result: TrackiqResult) -> list[float]:
        """Extract all-reduce per-step values from canonical tool payload."""
        payload = result.tool_payload if isinstance(result.tool_payload, dict) else {}
        if not payload:
            return []

        explicit = payload.get("allreduce_time_ms")
        if isinstance(explicit, list):
            explicit_values = [float(v) for v in explicit if isinstance(v, (int, float))]
            if explicit_values:
                return explicit_values

        steps = payload.get("steps")
        if not isinstance(steps, list):
            return []
        values: list[float] = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            raw_value = step.get("allreduce_time_ms")
            if isinstance(raw_value, (int, float)):
                values.append(float(raw_value))
        return values

    def _consistency_findings(self, result_a: TrackiqResult, result_b: TrackiqResult) -> list[ConsistencyFinding]:
        """Detect variance regressions in all-reduce consistency across runs."""
        series_a = self._extract_allreduce_series(result_a)
        series_b = self._extract_allreduce_series(result_b)
        if len(series_a) < 2 or len(series_b) < 2:
            return []

        stdev_a = float(statistics.stdev(series_a))
        stdev_b = float(statistics.stdev(series_b))
        if stdev_a == 0.0:
            increase_percent = float("inf") if stdev_b > 0.0 else 0.0
        else:
            increase_percent = ((stdev_b - stdev_a) / stdev_a) * 100.0

        if stdev_b > stdev_a and increase_percent > self.variance_threshold_percent:
            return [
                ConsistencyFinding(
                    code="VARIANCE_REGRESSION",
                    label="All-Reduce Consistency Degraded",
                    status="regression",
                    stddev_a_ms=stdev_a,
                    stddev_b_ms=stdev_b,
                    increase_percent=increase_percent,
                    threshold_percent=self.variance_threshold_percent,
                    reason=(
                        "All-reduce step-time variance increased beyond threshold. "
                        "This can indicate an emerging communication straggler."
                    ),
                )
            ]
        return []
