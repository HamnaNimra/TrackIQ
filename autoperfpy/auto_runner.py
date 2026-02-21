"""Auto benchmark runner for AutoPerfPy.

Runs benchmarks sequentially for each (device, inference_config). Uses a
collector factory (device_type -> collector instance) so adding new hardware
is a single registration. Same runner for automatic and manual modes. No remote
execution. Automotive-focused: wires TegrastatsCollector for Jetson/DRIVE;
trackiq_core provides generic device detection and runner infrastructure.
"""

from collections.abc import Callable
from typing import Any

from trackiq_core.collectors import SyntheticCollector
from trackiq_core.hardware.devices import (
    DEVICE_TYPE_CPU,
    DEVICE_TYPE_INTEL_GPU,
    DEVICE_TYPE_NVIDIA_DRIVE,
    DEVICE_TYPE_NVIDIA_GPU,
    DEVICE_TYPE_NVIDIA_JETSON,
    DeviceProfile,
)
from trackiq_core.inference import InferenceConfig
from trackiq_core.power_profiler import PowerProfiler, detect_power_source
from trackiq_core.runners import (
    BenchmarkRunner,
    base_collector_config,
    make_run_label,
)
from trackiq_core.runners import run_auto_benchmarks as _run_auto_benchmarks_generic
from trackiq_core.runners import run_single_benchmark as _run_single_benchmark_generic
from trackiq_core.schema import Metrics


def _collector_nvidia_gpu(device: DeviceProfile, config: InferenceConfig) -> Any | None:
    """Build NVML collector for discrete NVIDIA GPU."""
    try:
        from trackiq_core.collectors import NVMLCollector

        return NVMLCollector(device_index=device.index, config=base_collector_config(config))
    except (ImportError, Exception):
        return None


def _collector_psutil(device: DeviceProfile, config: InferenceConfig) -> Any | None:  # noqa: ARG001
    """Build Psutil collector for CPU or Intel GPU."""
    del device  # unused, but kept for consistent factory signature
    try:
        from trackiq_core.collectors import PsutilCollector

        return PsutilCollector(config=base_collector_config(config))
    except (ImportError, Exception):
        return None


def _collector_tegrastats(device: DeviceProfile, config: InferenceConfig) -> Any | None:
    """Build Tegrastats collector for Jetson/DRIVE (automotive / edge AI)."""
    try:
        from autoperfpy.collectors import TegrastatsCollector

        cfg = base_collector_config(config)
        return TegrastatsCollector(mode="live", config=cfg)
    except (ImportError, Exception):
        return None


# Registry: device_type -> (device, config) -> collector or None.
# Add new automotive hardware here.
COLLECTOR_FACTORY: dict[str, Callable[[DeviceProfile, InferenceConfig], Any | None]] = {
    DEVICE_TYPE_NVIDIA_GPU: _collector_nvidia_gpu,
    DEVICE_TYPE_CPU: _collector_psutil,
    DEVICE_TYPE_INTEL_GPU: _collector_psutil,
    DEVICE_TYPE_NVIDIA_JETSON: _collector_tegrastats,
    DEVICE_TYPE_NVIDIA_DRIVE: _collector_tegrastats,
}


def _collector_for_device(device: DeviceProfile, config: InferenceConfig) -> Any:
    """Build collector for the given device via registry; fallback Synthetic."""
    builder = COLLECTOR_FACTORY.get(device.device_type)
    if builder:
        collector = builder(device, config)
        if collector is not None:
            return collector
    return SyntheticCollector(config=base_collector_config(config))


def run_single_benchmark(
    device: DeviceProfile,
    config: InferenceConfig,
    duration_seconds: float = 10.0,
    sample_interval_seconds: float = 0.2,
    quiet: bool = True,
    enable_power: bool = True,
) -> dict[str, Any]:
    """Run one benchmark for (device, config) and return result dict.

    Uses the automotive collector factory to select appropriate collector
    for the device type (including Tegrastats for Jetson/DRIVE).

    Args:
        device: Device profile to benchmark
        config: Inference configuration
        duration_seconds: Benchmark duration
        sample_interval_seconds: Interval between samples
        quiet: If True, suppress progress output

    Returns:
        Dictionary with benchmark results, metadata, and run_label
    """
    profiler = PowerProfiler(detect_power_source()) if enable_power else None
    result = _run_single_benchmark_generic(
        device=device,
        config=config,
        collector_factory=_collector_for_device,
        duration_seconds=duration_seconds,
        sample_interval_seconds=sample_interval_seconds,
        quiet=quiet,
        power_profiler=profiler,
    )
    if profiler is not None:
        _attach_power_profile(result, profiler)
    return result


def run_auto_benchmarks(
    device_config_pairs: list[tuple[DeviceProfile, InferenceConfig]],
    duration_seconds: float = 10.0,
    sample_interval_seconds: float = 0.2,
    quiet: bool = True,
    progress_callback: Callable[..., None] | None = None,
    enable_power: bool = True,
) -> list[dict[str, Any]]:
    """Run benchmarks sequentially for each (device, config).

    Uses the automotive collector factory to select appropriate collector
    for each device type (including Tegrastats for Jetson/DRIVE).

    Args:
        device_config_pairs: List of (device, config) tuples to benchmark
        duration_seconds: Duration for each benchmark
        sample_interval_seconds: Interval between samples
        quiet: If True, suppress progress output
        progress_callback: Optional callback(i, total, device, config) for progress

    Returns:
        List of result dictionaries, one per (device, config) pair
    """
    return _run_auto_benchmarks_generic(
        device_config_pairs=device_config_pairs,
        collector_factory=_collector_for_device,
        duration_seconds=duration_seconds,
        sample_interval_seconds=sample_interval_seconds,
        quiet=quiet,
        progress_callback=progress_callback,
        power_profiler_factory=((lambda: PowerProfiler(detect_power_source())) if enable_power else None),
    )


def _attach_power_profile(result: dict[str, Any], profiler: PowerProfiler) -> None:
    """Populate result summary and payload with profiler-derived power metrics."""
    summary = result.setdefault("summary", {})
    latency = summary.get("latency", {}) if isinstance(summary.get("latency"), dict) else {}
    throughput = summary.get("throughput", {}) if isinstance(summary.get("throughput"), dict) else {}
    memory = summary.get("memory", {}) if isinstance(summary.get("memory"), dict) else {}
    base_metrics = Metrics(
        throughput_samples_per_sec=float(throughput.get("mean_fps", 0.0)),
        latency_p50_ms=float(latency.get("p50_ms", 0.0)),
        latency_p95_ms=float(latency.get("p95_ms", 0.0)),
        latency_p99_ms=float(latency.get("p99_ms", 0.0)),
        memory_utilization_percent=float(memory.get("mean_percent", 0.0)),
        communication_overhead_percent=None,
        power_consumption_watts=None,
    )
    updated_metrics = profiler.to_metrics_update(base_metrics)
    summary.setdefault("power", {})["mean_w"] = updated_metrics.power_consumption_watts
    summary["power"]["peak_w"] = profiler.to_tool_payload()["power_profile"]["summary"].get("peak_power_watts")
    result["power_profile"] = profiler.to_tool_payload()["power_profile"]


__all__ = [
    "BenchmarkRunner",
    "run_single_benchmark",
    "run_auto_benchmarks",
    "COLLECTOR_FACTORY",
    "make_run_label",
    "base_collector_config",
]
