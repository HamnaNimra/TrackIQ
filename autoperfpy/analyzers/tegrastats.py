"""Tegrastats analyzer for NVIDIA DriveOS/Jetson platforms.

This analyzer processes tegrastats output to provide insights into:
- Per-core CPU utilization (8+ cores on Orin/Thor)
- GR3D GPU frequency and utilization
- EMC (memory controller) frequency/bandwidth
- Thermal zone temperatures (CPU, GPU, AO, Tdiode, tj)
- RAM usage breakdown (lfb - largest free block)

Users can analyze tegrastats logs to assess system health, identify thermal throttling,
and monitor memory pressure conditions.

It leverages the TegrastatsParser and TegrastatsCalculator from the core module
to parse raw output and compute aggregate statistics.

Example usage:
    analyzer = TegrastatsAnalyzer()
    result = analyzer.analyze("tegrastats_output.log")
    summary = analyzer.summarize()


Authors:
    Hamna Nimra
"""

from typing import Dict, Any, Optional, List
from ..core import BaseAnalyzer, AnalysisResult
from ..core.tegrastats import (
    TegrastatsParser,
    TegrastatsCalculator,
    TegrastatsSnapshot,
)


class TegrastatsAnalyzer(BaseAnalyzer):
    """Analyze tegrastats output from NVIDIA DriveOS/Jetson platforms.

    This analyzer parses tegrastats output and computes aggregate metrics
    for CPU, GPU, memory, and thermal monitoring. It's designed for
    NVIDIA DRIVE platforms (Orin, Thor) and Jetson devices.

    Example usage:
        analyzer = TegrastatsAnalyzer()

        # Analyze from file
        result = analyzer.analyze("tegrastats_output.log")

        # Analyze from raw lines
        result = analyzer.analyze_lines(tegrastats_lines)

        # Get summary
        summary = analyzer.summarize()
    """

    # Default thermal throttling threshold (Celsius)
    DEFAULT_THROTTLE_TEMP_C = 85.0

    # Default memory pressure threshold (percent)
    DEFAULT_MEMORY_PRESSURE_PERCENT = 90.0

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize analyzer.

        Args:
            config: Optional configuration with:
                - throttle_temp_c: Temperature for throttling detection (default: 85.0)
                - memory_pressure_percent: RAM threshold for pressure detection (default: 90.0)
        """
        super().__init__("TegrastatsAnalyzer")
        self.config = config or {}
        self.throttle_temp_c = self.config.get(
            "throttle_temp_c", self.DEFAULT_THROTTLE_TEMP_C
        )
        self.memory_pressure_percent = self.config.get(
            "memory_pressure_percent", self.DEFAULT_MEMORY_PRESSURE_PERCENT
        )

    def analyze(self, filepath: str) -> AnalysisResult:
        """Analyze tegrastats output from a file.

        Args:
            filepath: Path to file containing tegrastats output lines

        Returns:
            AnalysisResult with system metrics
        """
        snapshots = TegrastatsParser.parse_file(filepath)
        return self._analyze_snapshots(snapshots, source=filepath)

    def analyze_lines(self, lines: List[str]) -> AnalysisResult:
        """Analyze tegrastats from a list of output lines.

        Args:
            lines: List of tegrastats output lines

        Returns:
            AnalysisResult with system metrics
        """
        snapshots = []
        for line in lines:
            line = line.strip()
            if line and "RAM" in line:
                try:
                    snapshot = TegrastatsParser.parse_line(line)
                    snapshots.append(snapshot)
                except Exception:
                    continue

        return self._analyze_snapshots(snapshots, source="lines")

    def analyze_snapshot(self, snapshot: TegrastatsSnapshot) -> Dict[str, Any]:
        """Analyze a single tegrastats snapshot.

        Args:
            snapshot: TegrastatsSnapshot object

        Returns:
            Dictionary with metrics from single snapshot
        """
        return snapshot.to_dict()

    def _analyze_snapshots(
        self,
        snapshots: List[TegrastatsSnapshot],
        source: str = "unknown",
    ) -> AnalysisResult:
        """Internal method to analyze a list of snapshots.

        Args:
            snapshots: List of TegrastatsSnapshot objects
            source: Source identifier for the data

        Returns:
            AnalysisResult with comprehensive metrics
        """
        if not snapshots:
            metrics = {
                "error": "No valid tegrastats data found",
                "num_samples": 0,
            }
            result = AnalysisResult(
                name="Tegrastats Analysis",
                metrics=metrics,
                raw_data=[],
            )
            self.add_result(result)
            return result

        # Calculate aggregate statistics
        aggregates = TegrastatsCalculator.calculate_aggregates(snapshots)

        # Detect issues
        thermal_throttling = TegrastatsCalculator.detect_thermal_throttling(
            snapshots, self.throttle_temp_c
        )
        memory_pressure = TegrastatsCalculator.detect_memory_pressure(
            snapshots, self.memory_pressure_percent
        )

        # Build metrics dictionary
        metrics = {
            "source": source,
            "num_samples": aggregates.num_samples,
            "duration_seconds": aggregates.duration_seconds,
            # CPU metrics
            "cpu": {
                "num_cores": len(aggregates.cpu_per_core_avg),
                "avg_utilization_percent": aggregates.cpu_avg_utilization,
                "max_utilization_percent": aggregates.cpu_max_utilization,
                "min_utilization_percent": aggregates.cpu_min_utilization,
                "per_core_avg_utilization": aggregates.cpu_per_core_avg,
            },
            # GPU (GR3D) metrics
            "gpu": {
                "avg_utilization_percent": aggregates.gpu_avg_utilization,
                "max_utilization_percent": aggregates.gpu_max_utilization,
                "min_utilization_percent": aggregates.gpu_min_utilization,
                "avg_frequency_mhz": aggregates.gpu_avg_frequency_mhz,
            },
            # Memory metrics
            "memory": {
                "total_mb": aggregates.ram_total_mb,
                "avg_used_mb": aggregates.ram_avg_used_mb,
                "max_used_mb": aggregates.ram_max_used_mb,
                "avg_utilization_percent": aggregates.ram_avg_utilization,
                "emc_avg_frequency_mhz": aggregates.emc_avg_frequency_mhz,
            },
            # Thermal metrics
            "thermal": {
                "avg_temps_c": aggregates.thermal_avg_temps,
                "max_temps_c": aggregates.thermal_max_temps,
                "max_observed_c": aggregates.thermal_max_observed,
            },
            # Issue detection
            "thermal_throttling": thermal_throttling,
            "memory_pressure": memory_pressure,
        }

        # Add health assessment
        metrics["health"] = self._assess_health(metrics)

        result = AnalysisResult(
            name="Tegrastats Analysis",
            metrics=metrics,
            raw_data=[s.to_dict() for s in snapshots],
        )
        self.add_result(result)
        return result

    def _assess_health(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall system health based on metrics.

        Args:
            metrics: Computed metrics dictionary

        Returns:
            Health assessment dictionary
        """
        issues = []
        warnings = []

        # Check thermal
        thermal = metrics.get("thermal_throttling", {})
        if thermal.get("throttle_percentage", 0) > 0:
            issues.append(
                f"Thermal throttling detected: {thermal['throttle_percentage']:.1f}% of samples"
            )

        max_temp = metrics.get("thermal", {}).get("max_observed_c", 0)
        if max_temp > 80:
            warnings.append(f"High temperature observed: {max_temp:.1f}C")

        # Check memory
        memory_pressure = metrics.get("memory_pressure", {})
        if memory_pressure.get("pressure_percentage", 0) > 0:
            warnings.append(
                f"Memory pressure detected: {memory_pressure['pressure_percentage']:.1f}% of samples"
            )

        # Check GPU utilization
        gpu_max = metrics.get("gpu", {}).get("max_utilization_percent", 0)
        if gpu_max > 95:
            warnings.append(f"GPU saturation detected: {gpu_max:.1f}% max utilization")

        # Check CPU utilization
        cpu_max = metrics.get("cpu", {}).get("max_utilization_percent", 0)
        if cpu_max > 95:
            warnings.append(f"CPU saturation detected: {cpu_max:.1f}% max utilization")

        # Determine overall status
        if issues:
            status = "critical"
        elif warnings:
            status = "warning"
        else:
            status = "healthy"

        return {
            "status": status,
            "issues": issues,
            "warnings": warnings,
        }

    def summarize(self) -> Dict[str, Any]:
        """Summarize all tegrastats analyses.

        Returns:
            Summary with aggregate statistics across all analyses
        """
        if not self.results:
            return {"total_analyses": 0}

        summary = {
            "total_analyses": len(self.results),
            "total_samples": 0,
            "total_duration_seconds": 0.0,
            "cpu_utilization_range": {"min": float("inf"), "max": 0.0},
            "gpu_utilization_range": {"min": float("inf"), "max": 0.0},
            "thermal_max_observed_c": 0.0,
            "throttle_events_total": 0,
            "memory_pressure_events_total": 0,
            "health_summary": {"healthy": 0, "warning": 0, "critical": 0},
        }

        for result in self.results:
            m = result.metrics
            summary["total_samples"] += m.get("num_samples", 0)
            summary["total_duration_seconds"] += m.get("duration_seconds", 0.0)

            # CPU range
            cpu = m.get("cpu", {})
            summary["cpu_utilization_range"]["min"] = min(
                summary["cpu_utilization_range"]["min"],
                cpu.get("min_utilization_percent", float("inf")),
            )
            summary["cpu_utilization_range"]["max"] = max(
                summary["cpu_utilization_range"]["max"],
                cpu.get("max_utilization_percent", 0),
            )

            # GPU range
            gpu = m.get("gpu", {})
            summary["gpu_utilization_range"]["min"] = min(
                summary["gpu_utilization_range"]["min"],
                gpu.get("min_utilization_percent", float("inf")),
            )
            summary["gpu_utilization_range"]["max"] = max(
                summary["gpu_utilization_range"]["max"],
                gpu.get("max_utilization_percent", 0),
            )

            # Thermal max
            thermal = m.get("thermal", {})
            summary["thermal_max_observed_c"] = max(
                summary["thermal_max_observed_c"], thermal.get("max_observed_c", 0)
            )

            # Issues
            throttle = m.get("thermal_throttling", {})
            summary["throttle_events_total"] += throttle.get("throttle_events", 0)

            pressure = m.get("memory_pressure", {})
            summary["memory_pressure_events_total"] += pressure.get(
                "pressure_events", 0
            )

            # Health
            health = m.get("health", {})
            status = health.get("status", "healthy")
            summary["health_summary"][status] = (
                summary["health_summary"].get(status, 0) + 1
            )

        # Handle edge case where no valid data
        if summary["cpu_utilization_range"]["min"] == float("inf"):
            summary["cpu_utilization_range"]["min"] = 0.0
        if summary["gpu_utilization_range"]["min"] == float("inf"):
            summary["gpu_utilization_range"]["min"] = 0.0

        return summary

    def compare_runs(
        self,
        baseline_metrics: Dict[str, Any],
        current_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare tegrastats metrics between two runs.

        Args:
            baseline_metrics: Metrics from baseline run
            current_metrics: Metrics from current run

        Returns:
            Comparison with deltas and improvement indicators
        """
        from trackiq_core import safe_get as _safe_get

        def _get(d: Dict, *keys, default=0.0):
            return _safe_get(d, *keys, default=default)

        baseline_cpu = _get(baseline_metrics, "cpu", "avg_utilization_percent")
        current_cpu = _get(current_metrics, "cpu", "avg_utilization_percent")

        baseline_gpu = _get(baseline_metrics, "gpu", "avg_utilization_percent")
        current_gpu = _get(current_metrics, "gpu", "avg_utilization_percent")

        baseline_mem = _get(baseline_metrics, "memory", "avg_utilization_percent")
        current_mem = _get(current_metrics, "memory", "avg_utilization_percent")

        baseline_temp = _get(baseline_metrics, "thermal", "max_observed_c")
        current_temp = _get(current_metrics, "thermal", "max_observed_c")

        return {
            "cpu_utilization_delta": current_cpu - baseline_cpu,
            "gpu_utilization_delta": current_gpu - baseline_gpu,
            "memory_utilization_delta": current_mem - baseline_mem,
            "thermal_max_delta_c": current_temp - baseline_temp,
            "cpu_more_efficient": current_cpu < baseline_cpu,
            "gpu_more_efficient": current_gpu < baseline_gpu,
            "memory_more_efficient": current_mem < baseline_mem,
            "runs_cooler": current_temp < baseline_temp,
            "baseline": {
                "cpu_avg": baseline_cpu,
                "gpu_avg": baseline_gpu,
                "memory_avg": baseline_mem,
                "thermal_max": baseline_temp,
            },
            "current": {
                "cpu_avg": current_cpu,
                "gpu_avg": current_gpu,
                "memory_avg": current_mem,
                "thermal_max": current_temp,
            },
        }


__all__ = ["TegrastatsAnalyzer"]
