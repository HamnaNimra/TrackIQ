"""Benchmark runner for TrackIQ.

Provides BenchmarkRunner class and utilities for running benchmarks
across devices with different configurations. Uses a collector factory
pattern so applications can register their own collectors.
"""

import time
from collections.abc import Callable
from typing import Any

from trackiq_core.collectors import CollectorBase, CollectorExport, SyntheticCollector
from trackiq_core.hardware.devices import DeviceProfile, get_platform_metadata_for_device
from trackiq_core.inference import InferenceConfig
from trackiq_core.power_profiler import PowerProfiler


class BenchmarkRunner:
    """Run a collector for a fixed duration and return exported results."""

    def __init__(
        self,
        collector: CollectorBase,
        duration_seconds: float,
        sample_interval_seconds: float = 0.1,
        quiet: bool = False,
        power_profiler: PowerProfiler | None = None,
    ):
        """Initialize benchmark runner.

        Args:
            collector: Collector instance to use for sampling
            duration_seconds: Duration to run the benchmark
            sample_interval_seconds: Interval between samples
            quiet: If True, suppress progress output
        """
        self.collector = collector
        self.duration_seconds = duration_seconds
        self.sample_interval_seconds = sample_interval_seconds
        self.quiet = quiet
        self.power_profiler = power_profiler

    def run(self) -> CollectorExport:
        """Run collection for duration_seconds, sampling at sample_interval_seconds.

        Returns:
            CollectorExport with collected samples and summary statistics
        """
        self.collector.start()
        start = time.time()
        sample_count = 0
        if self.power_profiler is not None:
            self.power_profiler.start_session()

        try:
            while time.time() - start < self.duration_seconds:
                ts = time.time()
                metrics = self.collector.sample(ts)
                if self.power_profiler is not None and metrics:
                    throughput = float(metrics.get("throughput_fps", 0.0) or 0.0)
                    self.power_profiler.record_step(sample_count, throughput)
                if not self.quiet and metrics:
                    warmup = metrics.get("is_warmup", False)
                    latency = metrics.get("latency_ms", 0)
                    gpu = metrics.get("gpu_percent", 0)
                    power = metrics.get("power_w", 0)
                    marker = " [WARMUP]" if warmup else ""
                    print(
                        f"[{sample_count:4d}] Latency: {latency:6.2f}ms | "
                        f"GPU: {gpu:5.1f}% | Power: {power:5.1f}W{marker}"
                    )
                sample_count += 1
                time.sleep(self.sample_interval_seconds)
        except KeyboardInterrupt:
            if not self.quiet:
                print("\nCollection interrupted by user")

        self.collector.stop()
        if self.power_profiler is not None:
            self.power_profiler.end_session()
        return self.collector.export()


def make_run_label(device: DeviceProfile, config: InferenceConfig) -> str:
    """Generate a short label for this run (e.g. nvidia_0_fp16_bs4).

    Args:
        device: Device profile
        config: Inference configuration

    Returns:
        Human-readable run label
    """
    return f"{device.device_id}_{config.precision}_bs{config.batch_size}"


def base_collector_config(config: InferenceConfig) -> dict[str, Any]:
    """Build base collector config from inference config.

    Args:
        config: Inference configuration

    Returns:
        Dictionary suitable for passing to a collector constructor
    """
    return {
        "warmup_samples": config.warmup_runs,
        "batch_sizes": [config.batch_size],
    }


# Type alias for collector factory functions
CollectorFactory = Callable[[DeviceProfile, InferenceConfig], CollectorBase | None]


def run_single_benchmark(
    device: DeviceProfile,
    config: InferenceConfig,
    collector_factory: CollectorFactory | None = None,
    duration_seconds: float = 10.0,
    sample_interval_seconds: float = 0.2,
    quiet: bool = True,
    power_profiler: PowerProfiler | None = None,
) -> dict[str, Any]:
    """Run one benchmark for (device, config) and return result dict.

    Args:
        device: Device profile to benchmark
        config: Inference configuration
        collector_factory: Optional function to create collector for device
        duration_seconds: Benchmark duration
        sample_interval_seconds: Interval between samples
        quiet: If True, suppress progress output

    Returns:
        Dictionary with benchmark results, metadata, and run_label
    """
    # Use provided factory or fall back to synthetic
    collector = None
    if collector_factory:
        collector = collector_factory(device, config)
    if collector is None:
        collector = SyntheticCollector(config=base_collector_config(config))

    runner = BenchmarkRunner(
        collector,
        duration_seconds=duration_seconds,
        sample_interval_seconds=sample_interval_seconds,
        quiet=quiet,
        power_profiler=power_profiler,
    )
    export = runner.run()
    result = export.to_dict()
    result["platform_metadata"] = get_platform_metadata_for_device(device)
    result["inference_config"] = config.to_dict()
    result["run_label"] = make_run_label(device, config)
    return result


def run_auto_benchmarks(
    device_config_pairs: list[tuple[DeviceProfile, InferenceConfig]],
    collector_factory: CollectorFactory | None = None,
    duration_seconds: float = 10.0,
    sample_interval_seconds: float = 0.2,
    quiet: bool = True,
    progress_callback: Callable[..., None] | None = None,
    power_profiler_factory: Callable[[], PowerProfiler | None] | None = None,
) -> list[dict[str, Any]]:
    """Run benchmarks sequentially for each (device, config).

    Same runner for auto/manual modes.

    Args:
        device_config_pairs: List of (device, config) tuples to benchmark
        collector_factory: Optional function to create collector for device
        duration_seconds: Duration for each benchmark
        sample_interval_seconds: Interval between samples
        quiet: If True, suppress progress output
        progress_callback: Optional callback(i, total, device, config) for progress

    Returns:
        List of result dictionaries, one per (device, config) pair
    """
    results: list[dict[str, Any]] = []
    total = len(device_config_pairs)
    for i, (device, config) in enumerate(device_config_pairs):
        if progress_callback:
            progress_callback(i + 1, total, device, config)
        try:
            res = run_single_benchmark(
                device,
                config,
                collector_factory=collector_factory,
                duration_seconds=duration_seconds,
                sample_interval_seconds=sample_interval_seconds,
                quiet=quiet,
                power_profiler=(power_profiler_factory() if power_profiler_factory else None),
            )
            results.append(res)
        except Exception as e:
            results.append(
                {
                    "run_label": make_run_label(device, config),
                    "platform_metadata": get_platform_metadata_for_device(device),
                    "inference_config": config.to_dict(),
                    "error": str(e),
                    "samples": [],
                    "summary": {},
                }
            )
    return results
