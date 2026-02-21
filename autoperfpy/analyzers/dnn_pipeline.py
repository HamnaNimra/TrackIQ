"""DNN Pipeline analyzer for TensorRT/DriveWorks performance analysis.

This analyzer processes DNN inference profiling data to provide insights into:
- Layer-by-layer inference timing
- DLA (Deep Learning Accelerator) vs GPU execution split
- TensorRT engine optimization metrics
- Batch inference throughput on DLA cores
- Memory copy overhead (host ↔ device)

Example usage:
    analyzer = DNNPipelineAnalyzer()
    # Analyze from profiler output
    result = analyzer.analyze_profiler_output(profiler_text)
    # Analyze from layer timing CSV
    result = analyzer.analyze_layer_csv("layer_times.csv", batch_size=4)
    # Analyze from InferenceRun objects
    result = analyzer.analyze_runs(inference_runs)
    # Get summary
    summary = analyzer.summarize()
    # Compare two engine configurations
    analyzer.compare_engines(baseline_metrics, optimized_metrics)

Classes:
- DNNPipelineAnalyzer: Main analyzer class for DNN pipeline performance.
- DNNPipelineParser: Parses profiler outputs and CSV files into structured data.
- DNNPipelineCalculator: Computes aggregate metrics and comparisons.
- InferenceRun: Data class representing a single inference run with layer timings.
- LayerTiming: Data class for individual layer timing information.
- MemoryTransfer: Data class for memory transfer timings.
- EngineOptimizationMetrics: Data class for TensorRT engine optimization stats.

Authors:
    Hamna Nimra
"""

from typing import Any

from ..core import AnalysisResult, BaseAnalyzer
from ..core.dnn_pipeline import (
    DNNPipelineCalculator,
    DNNPipelineParser,
    InferenceRun,
    LayerTiming,
    MemoryTransfer,
)


class DNNPipelineAnalyzer(BaseAnalyzer):
    """Analyze DNN inference pipeline performance.

    This analyzer processes TensorRT/DriveWorks profiling data to provide
    comprehensive insights into neural network inference performance on
    NVIDIA platforms with GPU and DLA accelerators.

    Example usage:
        analyzer = DNNPipelineAnalyzer()

        # Analyze from profiler output
        result = analyzer.analyze_profiler_output(profiler_text)

        # Analyze from layer timing CSV
        result = analyzer.analyze_layer_csv("layer_times.csv", batch_size=4)

        # Analyze from InferenceRun objects
        result = analyzer.analyze_runs(inference_runs)

        # Get summary
        summary = analyzer.summarize()
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize analyzer.

        Args:
            config: Optional configuration with:
                - top_n_layers: Number of slowest layers to report (default: 5)
                - memory_overhead_threshold: Warning threshold for memory overhead % (default: 20.0)
        """
        super().__init__("DNNPipelineAnalyzer")
        self.config = config or {}
        self.top_n_layers = self.config.get("top_n_layers", 5)
        self.memory_overhead_threshold = self.config.get("memory_overhead_threshold", 20.0)

    def analyze(self, data: Any) -> AnalysisResult:
        """Perform analysis on DNN pipeline data.

        This method serves as the generic entry point for analysis, supporting
        multiple data formats:
        - str: Profiler output text
        - List[InferenceRun]: List of inference runs
        - List[Dict]: List of layer timing dictionaries

        Args:
            data: Input data (profiler output string, InferenceRun list, or dict list)

        Returns:
            AnalysisResult with inference metrics
        """
        if isinstance(data, str):
            return self.analyze_profiler_output(data)
        elif isinstance(data, list):
            if len(data) == 0:
                return self.analyze_runs([])
            if isinstance(data[0], InferenceRun):
                return self.analyze_runs(data)
            elif isinstance(data[0], dict):
                return self.analyze_from_data(data)
        raise ValueError(f"Unsupported data type: {type(data)}")

    def analyze_profiler_output(self, content: str) -> AnalysisResult:
        """Analyze raw profiler output text.

        Args:
            content: Raw profiler output containing layer timings

        Returns:
            AnalysisResult with inference metrics
        """
        run = DNNPipelineParser.parse_profiler_output(content)
        return self._analyze_single_run(run, source="profiler_output")

    def analyze_layer_csv(
        self,
        filepath: str,
        batch_size: int = 1,
    ) -> AnalysisResult:
        """Analyze layer timing from CSV file.

        Expected CSV columns: layer_name, layer_type, time_ms, device

        Args:
            filepath: Path to CSV file with layer timings
            batch_size: Batch size used for inference

        Returns:
            AnalysisResult with layer-level metrics
        """
        run = DNNPipelineParser.parse_csv_layer_timing(filepath, batch_size)
        return self._analyze_single_run(run, source=filepath)

    def analyze_runs(
        self,
        runs: list[InferenceRun],
        name: str = "inference_analysis",
    ) -> AnalysisResult:
        """Analyze multiple inference runs.

        Args:
            runs: List of InferenceRun objects
            name: Name for this analysis

        Returns:
            AnalysisResult with aggregate metrics
        """
        if not runs:
            metrics = {
                "error": "No inference runs provided",
                "num_runs": 0,
            }
            result = AnalysisResult(
                name="DNN Pipeline Analysis",
                metrics=metrics,
                raw_data=[],
            )
            self.add_result(result)
            return result

        # Calculate aggregates
        aggregates = DNNPipelineCalculator.calculate_aggregates(runs, top_n_layers=self.top_n_layers)

        # Batch scaling analysis
        batch_analysis = DNNPipelineCalculator.analyze_batch_scaling(runs)

        # DLA vs GPU comparison
        dla_gpu_comparison = DNNPipelineCalculator.compare_dla_vs_gpu(runs)

        metrics = {
            "name": name,
            "num_runs": aggregates.num_runs,
            "total_inferences": aggregates.total_inferences,
            "batch_sizes_used": aggregates.batch_sizes_used,
            # Timing metrics
            "timing": {
                "avg_total_ms": aggregates.avg_total_time_ms,
                "min_total_ms": aggregates.min_total_time_ms,
                "max_total_ms": aggregates.max_total_time_ms,
                "std_total_ms": aggregates.std_total_time_ms,
            },
            # Throughput metrics
            "throughput": {
                "avg_fps": aggregates.avg_throughput_fps,
                "max_fps": aggregates.max_throughput_fps,
            },
            # Device split (DLA vs GPU)
            "device_split": {
                "avg_gpu_time_ms": aggregates.avg_gpu_time_ms,
                "avg_dla_time_ms": aggregates.avg_dla_time_ms,
                "dla_percentage": aggregates.avg_dla_percentage,
                "gpu_percentage": 100 - aggregates.avg_dla_percentage,
            },
            # Memory overhead
            "memory_overhead": {
                "avg_h2d_time_ms": aggregates.avg_h2d_time_ms,
                "avg_d2h_time_ms": aggregates.avg_d2h_time_ms,
                "avg_total_overhead_ms": aggregates.avg_memory_overhead_ms,
                "overhead_percentage": aggregates.memory_overhead_percentage,
            },
            # Layer breakdown
            "slowest_layers": aggregates.slowest_layers,
            # Batch analysis
            "batch_scaling": batch_analysis,
            # DLA vs GPU
            "dla_gpu_comparison": dla_gpu_comparison,
        }

        # Add recommendations
        metrics["recommendations"] = self._generate_recommendations(metrics)

        result = AnalysisResult(
            name="DNN Pipeline Analysis",
            metrics=metrics,
            raw_data=[r.to_dict() for r in runs],
        )
        self.add_result(result)
        return result

    def analyze_from_data(
        self,
        layer_timings: list[dict[str, Any]],
        memory_transfers: list[dict[str, Any]] | None = None,
        batch_size: int = 1,
        total_time_ms: float | None = None,
    ) -> AnalysisResult:
        """Analyze from raw timing data dictionaries.

        Args:
            layer_timings: List of layer timing dicts with keys:
                - name: Layer name
                - layer_type: Layer type (e.g., "Conv", "Pool")
                - execution_time_ms: Execution time
                - device: "GPU" or "DLA0", "DLA1", etc.
            memory_transfers: Optional list of memory transfer dicts with keys:
                - transfer_type: "H2D" or "D2H"
                - size_bytes: Transfer size
                - duration_ms: Transfer time
            batch_size: Batch size used
            total_time_ms: Optional total time (computed from layers if not provided)

        Returns:
            AnalysisResult with metrics
        """
        from datetime import datetime

        layers = [
            LayerTiming(
                name=lt["name"],
                layer_type=lt.get("layer_type", "unknown"),
                execution_time_ms=lt["execution_time_ms"],
                device=lt.get("device", "GPU"),
            )
            for lt in layer_timings
        ]

        transfers = []
        if memory_transfers:
            transfers = [
                MemoryTransfer(
                    transfer_type=mt["transfer_type"],
                    size_bytes=mt["size_bytes"],
                    duration_ms=mt["duration_ms"],
                )
                for mt in memory_transfers
            ]

        computed_total = sum(layer.execution_time_ms for layer in layers)
        if transfers:
            computed_total += sum(t.duration_ms for t in transfers)

        run = InferenceRun(
            timestamp=datetime.now(),
            batch_size=batch_size,
            total_time_ms=total_time_ms if total_time_ms else computed_total,
            layers=layers,
            memory_transfers=transfers,
        )

        return self._analyze_single_run(run, source="data")

    def _analyze_single_run(
        self,
        run: InferenceRun,
        source: str = "unknown",
    ) -> AnalysisResult:
        """Internal method to analyze a single inference run.

        Args:
            run: InferenceRun object
            source: Source identifier

        Returns:
            AnalysisResult with metrics
        """
        # Group layers by device
        gpu_layers = [layer for layer in run.layers if layer.device == "GPU"]
        dla_layers = [layer for layer in run.layers if layer.device.startswith("DLA")]

        # Find slowest layers
        sorted_layers = sorted(run.layers, key=lambda layer: layer.execution_time_ms, reverse=True)
        slowest = [
            {
                "name": layer.name,
                "type": layer.layer_type,
                "time_ms": layer.execution_time_ms,
                "device": layer.device,
            }
            for layer in sorted_layers[: self.top_n_layers]
        ]

        metrics = {
            "source": source,
            "batch_size": run.batch_size,
            "num_layers": run.num_layers,
            # Timing breakdown
            "timing": {
                "total_time_ms": run.total_time_ms,
                "compute_time_ms": run.compute_time_ms,
                "gpu_time_ms": run.gpu_time_ms,
                "dla_time_ms": run.dla_time_ms,
            },
            # Throughput
            "throughput_fps": run.throughput_fps,
            # Device split
            "device_split": {
                "gpu_layers": len(gpu_layers),
                "dla_layers": len(dla_layers),
                "gpu_time_ms": run.gpu_time_ms,
                "dla_time_ms": run.dla_time_ms,
                "gpu_percentage": run.gpu_percentage,
                "dla_percentage": run.dla_percentage,
            },
            # Memory overhead
            "memory_overhead": {
                "h2d_time_ms": run.h2d_time_ms,
                "d2h_time_ms": run.d2h_time_ms,
                "total_overhead_ms": run.memory_overhead_ms,
                "overhead_percentage": (
                    (run.memory_overhead_ms / run.total_time_ms * 100) if run.total_time_ms > 0 else 0.0
                ),
                "num_transfers": len(run.memory_transfers),
            },
            # Layer analysis
            "slowest_layers": slowest,
        }

        # Add recommendations
        metrics["recommendations"] = self._generate_recommendations(metrics)

        result = AnalysisResult(
            name="DNN Pipeline Analysis",
            metrics=metrics,
            raw_data=run.to_dict(),
        )
        self.add_result(result)
        return result

    def _generate_recommendations(self, metrics: dict[str, Any]) -> list[str]:
        """Generate optimization recommendations based on metrics.

        Args:
            metrics: Computed metrics dictionary

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Check memory overhead
        mem_overhead = metrics.get("memory_overhead", {})
        overhead_pct = mem_overhead.get("overhead_percentage", 0)
        if overhead_pct > self.memory_overhead_threshold:
            recommendations.append(
                f"High memory transfer overhead ({overhead_pct:.1f}%). "
                "Consider using pinned memory or reducing transfer frequency."
            )

        # Check DLA utilization
        device_split = metrics.get("device_split", {})
        dla_pct = device_split.get("dla_percentage", 0)
        if dla_pct < 30 and dla_pct > 0:
            recommendations.append(
                f"Low DLA utilization ({dla_pct:.1f}%). " "Check for unsupported layers causing GPU fallback."
            )

        # Check for dominant layers
        slowest = metrics.get("slowest_layers", [])
        if slowest:
            timing = metrics.get("timing", {})
            total_time = timing.get("total_time_ms", timing.get("avg_total_ms", 1))
            if total_time > 0:
                top_layer_time = slowest[0].get("time_ms", slowest[0].get("avg_time_ms", 0))
                top_layer_pct = (top_layer_time / total_time) * 100
                if top_layer_pct > 30:
                    recommendations.append(
                        f"Layer '{slowest[0]['name']}' dominates execution ({top_layer_pct:.1f}%). "
                        "Consider layer fusion or optimization."
                    )

        # Check batch scaling
        batch_scaling = metrics.get("batch_scaling", {})
        if "batch_metrics" in batch_scaling and len(batch_scaling["batch_metrics"]) > 1:
            optimal_throughput = batch_scaling.get("optimal_for_throughput")
            optimal_latency = batch_scaling.get("optimal_for_latency")
            if optimal_throughput != optimal_latency:
                recommendations.append(
                    f"Trade-off detected: batch {optimal_latency} for latency, "
                    f"batch {optimal_throughput} for throughput."
                )

        if not recommendations:
            recommendations.append("No significant optimization opportunities detected.")

        return recommendations

    def summarize(self) -> dict[str, Any]:
        """Summarize all DNN pipeline analyses.

        Returns:
            Summary with aggregate statistics across all analyses
        """
        if not self.results:
            return {"total_analyses": 0}

        summary = {
            "total_analyses": len(self.results),
            "total_runs": 0,
            "avg_throughput_fps": 0.0,
            "max_throughput_fps": 0.0,
            "avg_dla_percentage": 0.0,
            "avg_memory_overhead_percentage": 0.0,
            "all_recommendations": [],
        }

        throughputs = []
        dla_percentages = []
        memory_overheads = []

        for result in self.results:
            m = result.metrics

            # Count runs
            summary["total_runs"] += m.get("num_runs", 1)

            # Collect throughputs
            throughput = m.get("throughput", {})
            if isinstance(throughput, dict):
                if "avg_fps" in throughput:
                    throughputs.append(throughput["avg_fps"])
                elif "max_fps" in throughput:
                    throughputs.append(throughput["max_fps"])
            elif "throughput_fps" in m:
                throughputs.append(m["throughput_fps"])

            # Collect DLA percentages
            device_split = m.get("device_split", {})
            if "dla_percentage" in device_split:
                dla_percentages.append(device_split["dla_percentage"])

            # Collect memory overhead
            mem_overhead = m.get("memory_overhead", {})
            if "overhead_percentage" in mem_overhead:
                memory_overheads.append(mem_overhead["overhead_percentage"])

            # Collect recommendations
            recs = m.get("recommendations", [])
            for rec in recs:
                if rec not in summary["all_recommendations"]:
                    summary["all_recommendations"].append(rec)

        # Calculate averages
        if throughputs:
            summary["avg_throughput_fps"] = sum(throughputs) / len(throughputs)
            summary["max_throughput_fps"] = max(throughputs)

        if dla_percentages:
            summary["avg_dla_percentage"] = sum(dla_percentages) / len(dla_percentages)

        if memory_overheads:
            summary["avg_memory_overhead_percentage"] = sum(memory_overheads) / len(memory_overheads)

        return summary

    def compare_engines(
        self,
        baseline_metrics: dict[str, Any],
        optimized_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare metrics between two engine configurations.

        Args:
            baseline_metrics: Metrics from baseline engine
            optimized_metrics: Metrics from optimized engine

        Returns:
            Comparison with improvement indicators
        """
        from trackiq_core import safe_get

        baseline_time = safe_get(baseline_metrics, "timing", "total_time_ms", default=0.0)
        if baseline_time == 0:
            baseline_time = safe_get(baseline_metrics, "timing", "avg_total_ms", default=0.0)
        optimized_time = safe_get(optimized_metrics, "timing", "total_time_ms", default=0.0)
        if optimized_time == 0:
            optimized_time = safe_get(optimized_metrics, "timing", "avg_total_ms", default=0.0)

        baseline_throughput = safe_get(baseline_metrics, "throughput_fps", default=0.0)
        if baseline_throughput == 0:
            baseline_throughput = safe_get(baseline_metrics, "throughput", "avg_fps", default=0.0)
        optimized_throughput = safe_get(optimized_metrics, "throughput_fps", default=0.0)
        if optimized_throughput == 0:
            optimized_throughput = safe_get(optimized_metrics, "throughput", "avg_fps", default=0.0)

        baseline_overhead = safe_get(baseline_metrics, "memory_overhead", "overhead_percentage", default=0.0)
        optimized_overhead = safe_get(optimized_metrics, "memory_overhead", "overhead_percentage", default=0.0)

        # Calculate improvements
        latency_improvement = ((baseline_time - optimized_time) / baseline_time * 100) if baseline_time > 0 else 0.0
        throughput_improvement = (
            ((optimized_throughput - baseline_throughput) / baseline_throughput * 100)
            if baseline_throughput > 0
            else 0.0
        )
        overhead_reduction = (
            ((baseline_overhead - optimized_overhead) / baseline_overhead * 100) if baseline_overhead > 0 else 0.0
        )

        return {
            "latency_improvement_percent": latency_improvement,
            "throughput_improvement_percent": throughput_improvement,
            "memory_overhead_reduction_percent": overhead_reduction,
            "is_faster": optimized_time < baseline_time,
            "is_higher_throughput": optimized_throughput > baseline_throughput,
            "baseline": {
                "latency_ms": baseline_time,
                "throughput_fps": baseline_throughput,
                "memory_overhead_percent": baseline_overhead,
            },
            "optimized": {
                "latency_ms": optimized_time,
                "throughput_fps": optimized_throughput,
                "memory_overhead_percent": optimized_overhead,
            },
        }


__all__ = ["DNNPipelineAnalyzer"]
