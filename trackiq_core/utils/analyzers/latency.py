"""Latency and performance analyzers."""

from typing import Any

from trackiq_core.schemas import AnalysisResult
from trackiq_core.utils.analysis_utils import DataLoader, LatencyStats
from trackiq_core.utils.base import BaseAnalyzer


class PercentileLatencyAnalyzer(BaseAnalyzer):
    """Analyze percentile latencies from benchmark data."""

    def __init__(self, config=None):
        """Initialize analyzer.

        Args:
            config: Optional configuration object
        """
        super().__init__("PercentileLatencyAnalyzer")
        self.config = config

    def analyze(self, csv_filepath: str) -> AnalysisResult:
        """Analyze benchmark CSV file.

        Args:
            csv_filepath: Path to benchmark CSV

        Returns:
            AnalysisResult with percentile statistics
        """
        # Load and validate data
        df = DataLoader.load_csv(csv_filepath)
        required_cols = ["timestamp", "workload", "batch_size", "latency_ms", "power_w"]
        DataLoader.validate_columns(df, required_cols)

        # Calculate percentiles per workload
        metrics = {}
        for workload in df["workload"].unique():
            workload_df = df[df["workload"] == workload]
            for batch_size in workload_df["batch_size"].unique():
                batch_df = workload_df[workload_df["batch_size"] == batch_size]
                latencies = batch_df["latency_ms"].tolist()

                key = f"{workload}_batch{batch_size}"
                metrics[key] = LatencyStats.calculate_percentiles(latencies)
                metrics[key]["num_samples"] = len(latencies)
                metrics[key]["power_mean"] = batch_df["power_w"].mean()

        result = AnalysisResult(name="Percentile Latency Analysis", metrics=metrics, raw_data=df)
        self.add_result(result)
        return result

    def summarize(self) -> dict[str, Any]:
        """Summarize all analysis results.

        Returns:
            Summary dictionary
        """
        summary = {
            "total_analyses": len(self.results),
            "workloads_analyzed": set(),
            "best_latency": float("inf"),
            "worst_latency": 0,
        }

        for result in self.results:
            for key, metrics in result.metrics.items():
                if "p99" in metrics:
                    summary["workloads_analyzed"].add(key)
                    summary["best_latency"] = min(summary["best_latency"], metrics.get("min", 0))
                    summary["worst_latency"] = max(summary["worst_latency"], metrics.get("max", 0))

        summary["workloads_analyzed"] = list(summary["workloads_analyzed"])
        return summary


class LogAnalyzer(BaseAnalyzer):
    """Analyze performance logs for latency spikes."""

    def __init__(self, config=None):
        """Initialize analyzer.

        Args:
            config: Optional configuration object
        """
        super().__init__("LogAnalyzer")
        self.config = config

    def analyze(self, log_filepath: str, threshold_ms: float = 50.0) -> AnalysisResult:
        """Analyze performance log.

        Args:
            log_filepath: Path to log file
            threshold_ms: Latency threshold in milliseconds

        Returns:
            AnalysisResult with spike events
        """
        events = []
        total_events = 0

        try:
            with open(log_filepath) as f:
                for line in f:
                    if "Frame" in line and "E2E" in line:
                        total_events += 1
                        # Parse latency from line
                        try:
                            if "E2E: " in line:
                                e2e_str = line.split("E2E: ")[1].split("ms")[0]
                                e2e_latency = float(e2e_str)

                                if e2e_latency > threshold_ms:
                                    events.append(
                                        {"line": line.strip(), "latency_ms": e2e_latency, "exceeds_threshold": True}
                                    )
                        except (IndexError, ValueError):
                            continue
        except FileNotFoundError:
            raise FileNotFoundError(f"Log file not found: {log_filepath}")

        metrics = {
            "threshold_ms": threshold_ms,
            "total_events": total_events,
            "spike_events": len(events),
            "spike_percentage": (len(events) / total_events * 100) if total_events > 0 else 0,
            "events": events[:10],  # Store first 10 spikes
        }

        result = AnalysisResult(name="Log Analysis", metrics=metrics, raw_data={"events": events})
        self.add_result(result)
        return result


__all__ = ["PercentileLatencyAnalyzer", "LogAnalyzer"]
