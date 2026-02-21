"""DNN Pipeline utilities for TensorRT/DriveWorks performance analysis.

This module provides utilities for analyzing DNN inference pipelines on
NVIDIA platforms, including:
- Layer-by-layer inference timing
- DLA (Deep Learning Accelerator) vs GPU execution split
- TensorRT engine optimization metrics
- Batch inference throughput
- Memory copy overhead (host ↔ device)

Classes:
- LayerTiming: Timing information for a single DNN layer.
- MemoryTransfer: Memory transfer timing between host and device.
- InferenceRun: A single inference execution with timing breakdown.
- EngineOptimizationMetrics: Metrics from TensorRT engine optimization/build.
- DNNPipelineParser: Parser for DNN pipeline profiling data.
- DNNPipelineCalculator: Calculate aggregate statistics from DNN pipeline runs.

Users can utilize these classes to analyze and optimize DNN inference pipelines
on NVIDIA hardware, helping to identify bottlenecks and improve performance.

Authors:
    Hamna Nimra
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class LayerTiming:
    """Timing information for a single DNN layer."""

    name: str
    layer_type: str
    execution_time_ms: float
    device: str  # "GPU", "DLA0", "DLA1", etc.
    input_dims: list[int] | None = None
    output_dims: list[int] | None = None
    workspace_size_bytes: int | None = None

    @property
    def execution_time_us(self) -> float:
        """Return execution time in microseconds."""
        return self.execution_time_ms * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "layer_type": self.layer_type,
            "execution_time_ms": self.execution_time_ms,
            "execution_time_us": self.execution_time_us,
            "device": self.device,
            "input_dims": self.input_dims,
            "output_dims": self.output_dims,
            "workspace_size_bytes": self.workspace_size_bytes,
        }


@dataclass
class MemoryTransfer:
    """Memory transfer timing between host and device."""

    transfer_type: str  # "H2D" (Host to Device) or "D2H" (Device to Host)
    size_bytes: int
    duration_ms: float
    stream_id: int | None = None

    @property
    def bandwidth_gbps(self) -> float:
        """Calculate effective bandwidth in GB/s."""
        if self.duration_ms <= 0:
            return 0.0
        size_gb = self.size_bytes / (1024**3)
        duration_s = self.duration_ms / 1000
        return size_gb / duration_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "transfer_type": self.transfer_type,
            "size_bytes": self.size_bytes,
            "size_mb": self.size_bytes / (1024**2),
            "duration_ms": self.duration_ms,
            "bandwidth_gbps": self.bandwidth_gbps,
            "stream_id": self.stream_id,
        }


@dataclass
class InferenceRun:
    """A single inference execution with timing breakdown."""

    timestamp: datetime
    batch_size: int
    total_time_ms: float
    layers: list[LayerTiming]
    memory_transfers: list[MemoryTransfer] = field(default_factory=list)
    engine_name: str | None = None

    @property
    def gpu_time_ms(self) -> float:
        """Total time spent on GPU layers."""
        return sum(layer.execution_time_ms for layer in self.layers if layer.device == "GPU")

    @property
    def dla_time_ms(self) -> float:
        """Total time spent on DLA layers."""
        return sum(layer.execution_time_ms for layer in self.layers if layer.device.startswith("DLA"))

    @property
    def h2d_time_ms(self) -> float:
        """Total host-to-device transfer time."""
        return sum(t.duration_ms for t in self.memory_transfers if t.transfer_type == "H2D")

    @property
    def d2h_time_ms(self) -> float:
        """Total device-to-host transfer time."""
        return sum(t.duration_ms for t in self.memory_transfers if t.transfer_type == "D2H")

    @property
    def memory_overhead_ms(self) -> float:
        """Total memory transfer overhead."""
        return self.h2d_time_ms + self.d2h_time_ms

    @property
    def compute_time_ms(self) -> float:
        """Total compute time (GPU + DLA)."""
        return self.gpu_time_ms + self.dla_time_ms

    @property
    def dla_percentage(self) -> float:
        """Percentage of compute time on DLA."""
        total = self.compute_time_ms
        if total <= 0:
            return 0.0
        return (self.dla_time_ms / total) * 100

    @property
    def gpu_percentage(self) -> float:
        """Percentage of compute time on GPU."""
        total = self.compute_time_ms
        if total <= 0:
            return 0.0
        return (self.gpu_time_ms / total) * 100

    @property
    def throughput_fps(self) -> float:
        """Throughput in frames/inferences per second."""
        if self.total_time_ms <= 0:
            return 0.0
        return (self.batch_size * 1000) / self.total_time_ms

    @property
    def num_layers(self) -> int:
        """Number of layers in the model."""
        return len(self.layers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "batch_size": self.batch_size,
            "total_time_ms": self.total_time_ms,
            "gpu_time_ms": self.gpu_time_ms,
            "dla_time_ms": self.dla_time_ms,
            "h2d_time_ms": self.h2d_time_ms,
            "d2h_time_ms": self.d2h_time_ms,
            "memory_overhead_ms": self.memory_overhead_ms,
            "compute_time_ms": self.compute_time_ms,
            "dla_percentage": self.dla_percentage,
            "gpu_percentage": self.gpu_percentage,
            "throughput_fps": self.throughput_fps,
            "num_layers": self.num_layers,
            "engine_name": self.engine_name,
            "layers": [layer.to_dict() for layer in self.layers],
            "memory_transfers": [t.to_dict() for t in self.memory_transfers],
        }


@dataclass
class EngineOptimizationMetrics:
    """Metrics from TensorRT engine optimization/build."""

    engine_name: str
    build_time_seconds: float
    input_shapes: dict[str, list[int]]
    output_shapes: dict[str, list[int]]
    precision: str  # "FP32", "FP16", "INT8", "MIXED"
    dla_enabled: bool
    dla_cores_used: list[int] = field(default_factory=list)
    gpu_fallback_layers: int = 0
    total_layers: int = 0
    workspace_size_mb: float = 0.0
    engine_size_mb: float = 0.0

    @property
    def dla_coverage_percent(self) -> float:
        """Percentage of layers running on DLA."""
        if self.total_layers == 0:
            return 0.0
        dla_layers = self.total_layers - self.gpu_fallback_layers
        return (dla_layers / self.total_layers) * 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_name": self.engine_name,
            "build_time_seconds": self.build_time_seconds,
            "input_shapes": self.input_shapes,
            "output_shapes": self.output_shapes,
            "precision": self.precision,
            "dla_enabled": self.dla_enabled,
            "dla_cores_used": self.dla_cores_used,
            "gpu_fallback_layers": self.gpu_fallback_layers,
            "total_layers": self.total_layers,
            "dla_coverage_percent": self.dla_coverage_percent,
            "workspace_size_mb": self.workspace_size_mb,
            "engine_size_mb": self.engine_size_mb,
        }


class DNNPipelineParser:
    """Parser for DNN pipeline profiling data."""

    # Patterns for parsing TensorRT/Nsight profiler output
    LAYER_PATTERN = re.compile(r"(?P<name>\S+)\s+(?P<type>\S+)\s+(?P<time>[\d.]+)\s*ms\s+(?P<device>GPU|DLA\d*)")
    MEMORY_PATTERN = re.compile(r"(?P<type>H2D|D2H)\s+(?P<size>\d+)\s+bytes?\s+(?P<time>[\d.]+)\s*ms")
    TOTAL_TIME_PATTERN = re.compile(r"Total:\s*([\d.]+)\s*ms")
    BATCH_SIZE_PATTERN = re.compile(r"[Bb]atch\s*[Ss]ize[:\s]+(\d+)")

    @classmethod
    def parse_profiler_output(
        cls,
        content: str,
        timestamp: datetime | None = None,
    ) -> InferenceRun:
        """Parse profiler output text into an InferenceRun.

        Args:
            content: Raw profiler output text
            timestamp: Optional timestamp for the run

        Returns:
            InferenceRun with parsed metrics
        """
        if timestamp is None:
            timestamp = datetime.now()

        layers = []
        memory_transfers = []
        total_time = 0.0
        batch_size = 1

        # Parse batch size
        batch_match = cls.BATCH_SIZE_PATTERN.search(content)
        if batch_match:
            batch_size = int(batch_match.group(1))

        # Parse total time
        total_match = cls.TOTAL_TIME_PATTERN.search(content)
        if total_match:
            total_time = float(total_match.group(1))

        # Parse layer timings
        for match in cls.LAYER_PATTERN.finditer(content):
            layers.append(
                LayerTiming(
                    name=match.group("name"),
                    layer_type=match.group("type"),
                    execution_time_ms=float(match.group("time")),
                    device=match.group("device"),
                )
            )

        # Parse memory transfers
        for match in cls.MEMORY_PATTERN.finditer(content):
            memory_transfers.append(
                MemoryTransfer(
                    transfer_type=match.group("type"),
                    size_bytes=int(match.group("size")),
                    duration_ms=float(match.group("time")),
                )
            )

        # If no total time found, compute from layers + transfers
        if total_time == 0.0:
            total_time = sum(layer.execution_time_ms for layer in layers) + sum(t.duration_ms for t in memory_transfers)

        return InferenceRun(
            timestamp=timestamp,
            batch_size=batch_size,
            total_time_ms=total_time,
            layers=layers,
            memory_transfers=memory_transfers,
        )

    @classmethod
    def parse_csv_layer_timing(
        cls,
        filepath: str,
        batch_size: int = 1,
    ) -> InferenceRun:
        """Parse layer timing from CSV file.

        Expected CSV format:
        layer_name,layer_type,time_ms,device

        Args:
            filepath: Path to CSV file
            batch_size: Batch size used

        Returns:
            InferenceRun with layer timings
        """
        import csv

        layers = []
        try:
            with open(filepath) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    layers.append(
                        LayerTiming(
                            name=row.get("layer_name", row.get("name", "unknown")),
                            layer_type=row.get("layer_type", row.get("type", "unknown")),
                            execution_time_ms=float(row.get("time_ms", row.get("time", 0))),
                            device=row.get("device", "GPU"),
                        )
                    )
        except FileNotFoundError:
            raise FileNotFoundError(f"Layer timing file not found: {filepath}")

        total_time = sum(layer.execution_time_ms for layer in layers)

        return InferenceRun(
            timestamp=datetime.now(),
            batch_size=batch_size,
            total_time_ms=total_time,
            layers=layers,
        )


@dataclass
class DNNPipelineAggregateStats:
    """Aggregated statistics from multiple inference runs."""

    num_runs: int
    total_inferences: int
    batch_sizes_used: list[int]

    # Timing aggregates
    avg_total_time_ms: float
    min_total_time_ms: float
    max_total_time_ms: float
    std_total_time_ms: float

    # Throughput
    avg_throughput_fps: float
    max_throughput_fps: float

    # Device split
    avg_gpu_time_ms: float
    avg_dla_time_ms: float
    avg_dla_percentage: float

    # Memory overhead
    avg_h2d_time_ms: float
    avg_d2h_time_ms: float
    avg_memory_overhead_ms: float
    memory_overhead_percentage: float

    # Layer breakdown
    slowest_layers: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_runs": self.num_runs,
            "total_inferences": self.total_inferences,
            "batch_sizes_used": self.batch_sizes_used,
            "timing": {
                "avg_total_time_ms": self.avg_total_time_ms,
                "min_total_time_ms": self.min_total_time_ms,
                "max_total_time_ms": self.max_total_time_ms,
                "std_total_time_ms": self.std_total_time_ms,
            },
            "throughput": {
                "avg_fps": self.avg_throughput_fps,
                "max_fps": self.max_throughput_fps,
            },
            "device_split": {
                "avg_gpu_time_ms": self.avg_gpu_time_ms,
                "avg_dla_time_ms": self.avg_dla_time_ms,
                "avg_dla_percentage": self.avg_dla_percentage,
            },
            "memory_overhead": {
                "avg_h2d_time_ms": self.avg_h2d_time_ms,
                "avg_d2h_time_ms": self.avg_d2h_time_ms,
                "avg_total_ms": self.avg_memory_overhead_ms,
                "percentage_of_total": self.memory_overhead_percentage,
            },
            "slowest_layers": self.slowest_layers,
        }


class DNNPipelineCalculator:
    """Calculate aggregate statistics from DNN pipeline runs."""

    @staticmethod
    def calculate_aggregates(
        runs: list[InferenceRun],
        top_n_layers: int = 5,
    ) -> DNNPipelineAggregateStats:
        """Calculate aggregate statistics from inference runs.

        Args:
            runs: List of InferenceRun objects
            top_n_layers: Number of slowest layers to report

        Returns:
            DNNPipelineAggregateStats with computed metrics
        """
        import numpy as np

        if not runs:
            return DNNPipelineAggregateStats(
                num_runs=0,
                total_inferences=0,
                batch_sizes_used=[],
                avg_total_time_ms=0.0,
                min_total_time_ms=0.0,
                max_total_time_ms=0.0,
                std_total_time_ms=0.0,
                avg_throughput_fps=0.0,
                max_throughput_fps=0.0,
                avg_gpu_time_ms=0.0,
                avg_dla_time_ms=0.0,
                avg_dla_percentage=0.0,
                avg_h2d_time_ms=0.0,
                avg_d2h_time_ms=0.0,
                avg_memory_overhead_ms=0.0,
                memory_overhead_percentage=0.0,
                slowest_layers=[],
            )

        n = len(runs)
        total_inferences = sum(r.batch_size for r in runs)
        batch_sizes = list(set(r.batch_size for r in runs))

        # Timing stats
        total_times = [r.total_time_ms for r in runs]
        throughputs = [r.throughput_fps for r in runs]

        avg_total = np.mean(total_times)
        avg_gpu = np.mean([r.gpu_time_ms for r in runs])
        avg_dla = np.mean([r.dla_time_ms for r in runs])
        avg_dla_pct = np.mean([r.dla_percentage for r in runs])

        avg_h2d = np.mean([r.h2d_time_ms for r in runs])
        avg_d2h = np.mean([r.d2h_time_ms for r in runs])
        avg_mem_overhead = np.mean([r.memory_overhead_ms for r in runs])
        mem_overhead_pct = (avg_mem_overhead / avg_total * 100) if avg_total > 0 else 0.0

        # Find slowest layers across all runs
        from collections import defaultdict

        layer_times: dict[str, list[float]] = defaultdict(list)
        for run in runs:
            for layer in run.layers:
                key = f"{layer.name}:{layer.layer_type}"
                layer_times[key].append(layer.execution_time_ms)

        # Calculate average time per layer and sort
        layer_avgs = [
            {"name": name, "type": layer_type, "avg_time_ms": np.mean(times)}
            for k, times in layer_times.items()
            for name, layer_type in [k.split(":", 1)]
        ]
        layer_avgs.sort(key=lambda x: x["avg_time_ms"], reverse=True)
        slowest = layer_avgs[:top_n_layers]

        return DNNPipelineAggregateStats(
            num_runs=n,
            total_inferences=total_inferences,
            batch_sizes_used=batch_sizes,
            avg_total_time_ms=float(avg_total),
            min_total_time_ms=float(np.min(total_times)),
            max_total_time_ms=float(np.max(total_times)),
            std_total_time_ms=float(np.std(total_times)),
            avg_throughput_fps=float(np.mean(throughputs)),
            max_throughput_fps=float(np.max(throughputs)),
            avg_gpu_time_ms=float(avg_gpu),
            avg_dla_time_ms=float(avg_dla),
            avg_dla_percentage=float(avg_dla_pct),
            avg_h2d_time_ms=float(avg_h2d),
            avg_d2h_time_ms=float(avg_d2h),
            avg_memory_overhead_ms=float(avg_mem_overhead),
            memory_overhead_percentage=float(mem_overhead_pct),
            slowest_layers=slowest,
        )

    @staticmethod
    def analyze_batch_scaling(
        runs: list[InferenceRun],
    ) -> dict[str, Any]:
        """Analyze how performance scales with batch size.

        Args:
            runs: List of InferenceRun objects with different batch sizes

        Returns:
            Analysis of batch scaling behavior
        """
        if not runs:
            return {"error": "No runs provided"}

        # Group runs by batch size
        by_batch: dict[int, list[InferenceRun]] = {}
        for run in runs:
            if run.batch_size not in by_batch:
                by_batch[run.batch_size] = []
            by_batch[run.batch_size].append(run)

        # Calculate metrics per batch size
        batch_metrics = []
        for batch_size in sorted(by_batch.keys()):
            batch_runs = by_batch[batch_size]
            avg_time = sum(r.total_time_ms for r in batch_runs) / len(batch_runs)
            avg_throughput = sum(r.throughput_fps for r in batch_runs) / len(batch_runs)
            avg_mem_overhead = sum(r.memory_overhead_ms for r in batch_runs) / len(batch_runs)

            batch_metrics.append(
                {
                    "batch_size": batch_size,
                    "avg_latency_ms": avg_time,
                    "avg_throughput_fps": avg_throughput,
                    "avg_memory_overhead_ms": avg_mem_overhead,
                    "latency_per_sample_ms": avg_time / batch_size,
                    "num_runs": len(batch_runs),
                }
            )

        # Find optimal batch sizes
        optimal_throughput = max(batch_metrics, key=lambda x: x["avg_throughput_fps"])
        optimal_latency = min(batch_metrics, key=lambda x: x["avg_latency_ms"])
        optimal_efficiency = min(batch_metrics, key=lambda x: x["latency_per_sample_ms"])

        return {
            "batch_metrics": batch_metrics,
            "optimal_for_throughput": optimal_throughput["batch_size"],
            "optimal_for_latency": optimal_latency["batch_size"],
            "optimal_for_efficiency": optimal_efficiency["batch_size"],
            "max_throughput_fps": optimal_throughput["avg_throughput_fps"],
            "min_latency_ms": optimal_latency["avg_latency_ms"],
        }

    @staticmethod
    def compare_dla_vs_gpu(
        runs: list[InferenceRun],
    ) -> dict[str, Any]:
        """Compare DLA vs GPU execution characteristics.

        Args:
            runs: List of InferenceRun objects

        Returns:
            Comparison of DLA vs GPU execution
        """
        if not runs:
            return {"error": "No runs provided"}

        total_gpu_time = sum(r.gpu_time_ms for r in runs)
        total_dla_time = sum(r.dla_time_ms for r in runs)
        total_compute = total_gpu_time + total_dla_time

        # Count layers by device
        gpu_layers = set()
        dla_layers = set()
        for run in runs:
            for layer in run.layers:
                if layer.device == "GPU":
                    gpu_layers.add(layer.name)
                elif layer.device.startswith("DLA"):
                    dla_layers.add(layer.name)

        return {
            "total_gpu_time_ms": total_gpu_time,
            "total_dla_time_ms": total_dla_time,
            "gpu_percentage": ((total_gpu_time / total_compute * 100) if total_compute > 0 else 0),
            "dla_percentage": ((total_dla_time / total_compute * 100) if total_compute > 0 else 0),
            "gpu_layer_count": len(gpu_layers),
            "dla_layer_count": len(dla_layers),
            "gpu_layers": list(gpu_layers),
            "dla_layers": list(dla_layers),
            "recommendation": (
                "Consider DLA for power efficiency"
                if total_dla_time < total_gpu_time
                else "GPU provides better throughput"
            ),
        }


__all__ = [
    "LayerTiming",
    "MemoryTransfer",
    "InferenceRun",
    "EngineOptimizationMetrics",
    "DNNPipelineParser",
    "DNNPipelineAggregateStats",
    "DNNPipelineCalculator",
]
